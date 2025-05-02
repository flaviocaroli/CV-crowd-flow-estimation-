import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ShanghaiTechDataset(Dataset):
    def __init__(self, root: str, part: str = 'part_A', split: str = 'train_data', transform=None, target_size=(384, 384)):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # paths
        self.images_dir = os.path.join(root, part, split, 'images')
        self.gt_dir = os.path.join(root, part, split, 'ground_truth')

        # list all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Resize image to target size
        img = img.resize(self.target_size, Image.BILINEAR)

        # Load ground truth
        mat_name = 'GT_' + os.path.splitext(img_name)[0] + '.mat'
        mat_path = os.path.join(self.gt_dir, mat_name)
        mat = scipy.io.loadmat(mat_path)
        
        # Create density map from point annotations
        den = np.zeros((orig_height, orig_width), dtype=np.float32)
        
        # Extract points from image_info
        points = mat['image_info'][0, 0][0, 0][0]
        
        # Place points on density map
        for point in points:
            x, y = int(min(point[0], orig_width-1)), int(min(point[1], orig_height-1))
            den[y, x] = 1.0
        
        # Resize density map to match target size
        if den.shape[0] != self.target_size[1] or den.shape[1] != self.target_size[0]:
            original_count = den.sum()  # Preserve total count
            
            # Create resized density map
            resized_den = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
            
            # Simple resize by scaling coordinates
            y_ratio = self.target_size[1] / orig_height
            x_ratio = self.target_size[0] / orig_width
            
            for point in points:
                x, y = point[0] * x_ratio, point[1] * y_ratio
                x, y = int(min(x, self.target_size[0]-1)), int(min(y, self.target_size[1]-1))
                resized_den[y, x] = 1.0
            
            den = resized_den
            
            # Scale to preserve count if needed
            if original_count > 0 and den.sum() > 0:
                den = den * (original_count / den.sum())

        # Apply transform to image
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        # Convert density map to tensor
        den_tensor = torch.from_numpy(den)
        
        return img, den_tensor