import unittest
import numpy as np
import torch
import albumentations as A
from typing import Tuple
# Attempt to import from src, assuming PYTHONPATH is set or tests are run from root
try:
    from src.custom_transforms import SmartCropScale, build_transforms
except ImportError:
    # Fallback for cases where src is not directly in PYTHONPATH
    # This might happen if tests are run from within the src/tests directory directly
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from custom_transforms import SmartCropScale, build_transforms


def create_dummy_image(height=100, width=100, channels=3) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

def create_dummy_keypoints() -> list[Tuple[float, float]]:
    return [(10.0, 20.0), (50.0, 50.0), (90.0, 80.0)] # x, y format

class TestSmartCropScale(unittest.TestCase):

    def test_initialization_defaults(self):
        transform = SmartCropScale()
        self.assertEqual(transform.min_crop_factor, 0.5)
        self.assertEqual(transform.max_crop_factor, 0.9)
        self.assertIsNone(transform.resize_to)
        self.assertEqual(transform.p, 1.0)

    def test_initialization_custom(self):
        transform = SmartCropScale(min_crop_factor=0.3, max_crop_factor=0.7, resize_to=(50, 50), p=0.5)
        self.assertEqual(transform.min_crop_factor, 0.3)
        self.assertEqual(transform.max_crop_factor, 0.7)
        self.assertEqual(transform.resize_to, (50, 50))
        self.assertEqual(transform.p, 0.5)

    def test_probability_p0(self):
        image = create_dummy_image()
        keypoints = create_dummy_keypoints()
        transform = SmartCropScale(p=0.0)
        result = transform(image=image.copy(), keypoints=[list(kp) for kp in keypoints]) # Pass copies
        self.assertTrue(np.array_equal(result['image'], image))
        self.assertEqual(result['keypoints'], keypoints)

    def test_probability_p1(self):
        image = create_dummy_image(100,100)
        keypoints = create_dummy_keypoints() # [[10,20], [50,50], [90,80]]
        transform = SmartCropScale(p=1.0, min_crop_factor=0.5, max_crop_factor=0.5, resize_to=None) # Force 50x50 crop

        # Mock random functions to get a deterministic crop
        # Crop a 50x50 region starting at (25, 25) from a 100x100 image
        with unittest.mock.patch('random.uniform', return_value=0.5): # crop_factor = 0.5 -> 50x50
            with unittest.mock.patch('random.randint', return_value=25): # x_start=25, y_start=25
                result = transform(image=image.copy(), keypoints=[list(kp) for kp in keypoints])
        
        self.assertEqual(result['image'].shape, (50, 50, 3))
        # Original keypoints: [[10,20], [50,50], [90,80]]
        # Crop from (25,25) to (75,75)
        # (10,20) -> outside
        # (50,50) -> (50-25, 50-25) = (25,25) -> inside
        # (90,80) -> outside
        self.assertEqual(len(result['keypoints']), 1)
        if len(result['keypoints']) == 1: # Ensure it's not empty before accessing
             self.assertAlmostEqual(result['keypoints'][0][0], 25.0)
             self.assertAlmostEqual(result['keypoints'][0][1], 25.0)

    def test_point_filtering_and_translation(self):
        image = create_dummy_image(100, 100)
        # Keypoints: one inside, one outside, one on edge
        keypoints = [[30.0, 30.0], [90.0, 90.0], [50.0, 0.0]] 
        transform = SmartCropScale(min_crop_factor=0.5, max_crop_factor=0.5, resize_to=None, p=1.0)

        # Mock to crop from (25,25) to (75,75) (a 50x50 area)
        with unittest.mock.patch('random.uniform', return_value=0.5):
            with unittest.mock.patch('random.randint', return_value=25): # x_start=25, y_start=25
                result = transform(image=image.copy(), keypoints=[list(kp) for kp in keypoints])
        
        # Expected:
        # (30,30) is inside -> (30-25, 30-25) = (5,5)
        # (90,90) is outside
        # (50,0) is outside (y_start is 25)
        self.assertEqual(len(result['keypoints']), 1)
        if len(result['keypoints']) == 1:
            self.assertListEqual(list(result['keypoints'][0]), [5.0, 5.0])

    def test_no_points_in_crop(self):
        image = create_dummy_image(100, 100)
        keypoints = [[90.0, 90.0]] # Point clearly outside the intended crop
        transform = SmartCropScale(min_crop_factor=0.5, max_crop_factor=0.5, resize_to=None, p=1.0)

        with unittest.mock.patch('random.uniform', return_value=0.5): # crop 50x50
            with unittest.mock.patch('random.randint', return_value=0):   # crop from (0,0) to (50,50)
                result = transform(image=image.copy(), keypoints=[list(kp) for kp in keypoints])
        
        self.assertEqual(len(result['keypoints']), 0)

    def test_resizing_after_crop(self):
        image = create_dummy_image(100, 100)
        keypoints = [[30.0, 30.0]] # This will be (5,5) in the 50x50 crop
        target_resize = (80, 80) # Resize the 50x50 crop to 80x80
        transform = SmartCropScale(min_crop_factor=0.5, max_crop_factor=0.5, 
                                   resize_to=target_resize, p=1.0)

        # Mock to crop 50x50 from (25,25)
        with unittest.mock.patch('random.uniform', return_value=0.5):
            with unittest.mock.patch('random.randint', return_value=25):
                result = transform(image=image.copy(), keypoints=[list(kp) for kp in keypoints])

        self.assertEqual(result['image'].shape, (target_resize[0], target_resize[1], 3))
        self.assertEqual(len(result['keypoints']), 1)
        if len(result['keypoints']) == 1:
            # Original in crop: (5,5)
            # Scale factor: 80/50 = 1.6
            # Expected: (5*1.6, 5*1.6) = (8,8)
            self.assertAlmostEqual(result['keypoints'][0][0], 8.0)
            self.assertAlmostEqual(result['keypoints'][0][1], 8.0)

    def test_empty_keypoints_input(self):
        image = create_dummy_image(100, 100)
        keypoints = []
        transform = SmartCropScale(p=1.0)
        result = transform(image=image.copy(), keypoints=[])
        self.assertEqual(len(result['keypoints']), 0)
        # Image should still be processed
        self.assertNotEqual(result['image'].shape, image.shape[:2]) # It will be cropped

class TestBuildTransforms(unittest.TestCase):

    def setUp(self):
        self.dummy_image = create_dummy_image(height=120, width=160)
        self.dummy_keypoints = create_dummy_keypoints()
        self.target_input_size = (64, 64) # H, W

        self.aug_config_full = {
            'smart_crop': {'min_factor': 0.5, 'max_factor': 0.8, 'p': 1.0},
            'hflip_prob': 0.5,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 
                             'saturation': 0.2, 'hue': 0.1, 'p': 0.8}
        }
        self.aug_config_minimal = {
            'hflip_prob': 0.0, # No flip
            # 'smart_crop': None, # or missing
            # 'color_jitter': None # or missing
        }
        self.aug_config_no_smart_crop_p0 = {
            'smart_crop': {'min_factor': 0.5, 'max_factor': 0.8, 'p': 0.0},
            'hflip_prob': 0.0,
        }


    def test_build_transforms_returns_compose_objects(self):
        train_tf, val_tf = build_transforms(self.aug_config_full, self.target_input_size)
        self.assertIsInstance(train_tf, A.Compose)
        self.assertIsInstance(val_tf, A.Compose)

    def test_train_transform_with_full_config(self):
        train_tf, _ = build_transforms(self.aug_config_full, self.target_input_size)
        
        # Mock random calls within SmartCropScale if needed for more deterministic output shape of keypoints
        # For now, check general properties
        transformed_data = train_tf(image=self.dummy_image.copy(), keypoints=np.array([list(kp) for kp in self.dummy_keypoints]))
        
        img_tensor = transformed_data['image']
        kps_transformed = transformed_data['keypoints']

        self.assertIsInstance(img_tensor, torch.Tensor)
        # Expected shape (C, H, W) = (3, target_input_size[0], target_input_size[1])
        self.assertEqual(img_tensor.shape, (3, self.target_input_size[0], self.target_input_size[1]))
        
        # Keypoints should be modified by SmartCropScale (cropping and resizing)
        # Their number might change, or their values will definitely change due to scaling.
        # It's hard to predict exact values without extensive mocking of SmartCropScale's random parts.
        self.assertNotEqual(kps_transformed, self.dummy_keypoints) # Check they are not the same object or values

    def test_val_transform_output(self):
        _, val_tf = build_transforms(self.aug_config_full, self.target_input_size)
        
        transformed_data = val_tf(image=self.dummy_image.copy(), keypoints=np.array([list(kp) for kp in self.dummy_keypoints]))
        img_tensor = transformed_data['image']
        kps_transformed = transformed_data['keypoints']

        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, self.target_input_size[0], self.target_input_size[1]))

        # Keypoints in val_transform are only affected by A.Resize
        # Original image: 160x120 (W,H). Target size: 64x64 (H,W) -> so resize to (64,64) H,W
        # Keypoints are (x,y)
        # Scale_x = new_W / old_W = 64 / 160
        # Scale_y = new_H / old_H = 64 / 120
        scale_x = self.target_input_size[1] / self.dummy_image.shape[1] # target_W / old_W
        scale_y = self.target_input_size[0] / self.dummy_image.shape[0] # target_H / old_H
        
        expected_kps = []
        for x, y in self.dummy_keypoints:
            expected_kps.append([x * scale_x, y * scale_y])
        
        self.assertEqual(len(kps_transformed), len(expected_kps))
        for kp_res, kp_exp in zip(kps_transformed, expected_kps):
            self.assertAlmostEqual(kp_res[0], kp_exp[0], places=4)
            self.assertAlmostEqual(kp_res[1], kp_exp[1], places=4)

    def test_train_transform_minimal_config(self):
        # No smart crop, no color jitter, no hflip
        train_tf, _ = build_transforms(self.aug_config_minimal, self.target_input_size)
        transformed_data = train_tf(image=self.dummy_image.copy(), keypoints=[list(kp) for kp in self.dummy_keypoints])
        img_tensor = transformed_data['image']
        kps_transformed = transformed_data['keypoints']

        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, self.target_input_size[0], self.target_input_size[1]))
        
        # Keypoints should be same as val_transform in this case (only Resize, Normalize, ToTensor)
        scale_x = self.target_input_size[1] / self.dummy_image.shape[1]
        scale_y = self.target_input_size[0] / self.dummy_image.shape[0]
        expected_kps = [[x * scale_x, y * scale_y] for x,y in self.dummy_keypoints]

        self.assertEqual(len(kps_transformed), len(expected_kps))
        for kp_res, kp_exp in zip(kps_transformed, expected_kps):
            self.assertAlmostEqual(kp_res[0], kp_exp[0], places=4)
            self.assertAlmostEqual(kp_res[1], kp_exp[1], places=4)


    def test_train_transform_smart_crop_p0(self):
        # SmartCropScale is in the list, but its p=0.0, so it won't apply.
        # The first transform in the list will be SmartCropScale itself, but it will act as identity.
        # However, the `build_transforms` logic adds A.Resize if SmartCropScale is not active (p=0 or not in config).
        # Let's verify this behavior.
        
        # If 'smart_crop' in config AND p=0:
        # The SmartCropScale instance IS added to train_transforms_list.
        # Its __call__ will return image and kps unchanged.
        # Then, other transforms like HFlip, ColorJitter, Normalize, ToTensorV2 are applied.
        # Crucially, A.Resize is NOT added separately if smart_crop config exists, even if p=0.
        # SmartCropScale with p=0 does not resize. So image shape will be original until Normalize/ToTensor.
        # This is a subtle point. The current build_transforms adds SmartCropScale if config exists and p > 0.
        # If p=0 or config for smart_crop not present, A.Resize is added.
        # So with smart_crop.p = 0, A.Resize IS added first.
        
        train_tf, _ = build_transforms(self.aug_config_no_smart_crop_p0, self.target_input_size)
        
        # Check that A.Resize is indeed the first effective transform if SmartCropScale has p=0
        # The first transform in train_tf.transforms should be A.Resize if p=0
        # The logic in build_transforms:
        # if smart_crop_config and smart_crop_config.get('p', 0.0) > 0:
        #    train_transforms_list.append(SmartCropScale(... resize_to=target_input_size ...))
        # else:
        #    train_transforms_list.append(A.Resize(height=target_input_size[0], width=target_input_size[1]))
        # So if p=0, A.Resize is added.

        self.assertIsInstance(train_tf.transforms[0], A.Resize) # First transform is Resize
        
        transformed_data = train_tf(image=self.dummy_image.copy(), keypoints=[list(kp) for kp in self.dummy_keypoints])
        img_tensor = transformed_data['image']
        kps_transformed = transformed_data['keypoints']

        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, self.target_input_size[0], self.target_input_size[1]))

        # Keypoints scaled by A.Resize
        scale_x = self.target_input_size[1] / self.dummy_image.shape[1]
        scale_y = self.target_input_size[0] / self.dummy_image.shape[0]
        expected_kps = [[x * scale_x, y * scale_y] for x,y in self.dummy_keypoints]
        
        self.assertEqual(len(kps_transformed), len(expected_kps))
        for kp_res, kp_exp in zip(kps_transformed, expected_kps):
            self.assertAlmostEqual(kp_res[0], kp_exp[0], places=4)
            self.assertAlmostEqual(kp_res[1], kp_exp[1], places=4)


if __name__ == '__main__':
    # Ensure src and src/tests directories and __init__.py files exist for imports
    # This is more for running the script directly, Python's unittest discovery might handle it.
    if not os.path.exists('src'):
        os.makedirs('src')
    if not os.path.exists('src/__init__.py'):
        with open('src/__init__.py', 'w') as f: pass
    if not os.path.exists('src/tests'):
        os.makedirs('src/tests')
    if not os.path.exists('src/tests/__init__.py'):
        with open('src/tests/__init__.py', 'w') as f: pass
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
