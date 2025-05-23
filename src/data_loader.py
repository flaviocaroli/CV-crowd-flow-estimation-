import os
from typing import Optional, Tuple
import math
import random

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import pytorch_lightning as pl

from src.utils import create_density_map, get_device


class RandomCropWithPoints:
    def __init__(self, size, allow_overlap=True):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.allow_overlap = allow_overlap
    
    def __call__(self, image, points):
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]
            image = Image.fromarray(image)
        
        target_h, target_w = self.size
        
        # Handle small images
        if h < target_h or w < target_w:
            padded = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            padded.paste(image, (0, 0))
            return padded, points
        
        # Random crop position
        y = random.randint(0, max(0, h - target_h))
        x = random.randint(0, max(0, w - target_w))
        
        # Crop image
        cropped = image.crop((x, y, x + target_w, y + target_h))
        
        # Filter and adjust points
        if points.size == 0:
            adjusted_points = np.zeros((0, 2), dtype=np.float32)
        else:
            mask = (points[:, 0] >= x) & (points[:, 0] < x + target_w) & \
                   (points[:, 1] >= y) & (points[:, 1] < y + target_h)
            
            adjusted_points = points[mask].copy()
            if adjusted_points.size > 0:
                adjusted_points[:, 0] -= x
                adjusted_points[:, 1] -= y
            else:
                adjusted_points = np.zeros((0, 2), dtype=np.float32)
        
        return cropped, adjusted_points


class HorizontalFlipWithPoints:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, points):
        if random.random() < self.p:
            if isinstance(image, Image.Image):
                w = image.size[0]
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                w = image.shape[1]
                flipped_image = Image.fromarray(np.fliplr(image))
            
            if points.size > 0:
                flipped_points = points.copy()
                flipped_points[:, 0] = w - 1 - points[:, 0]
                return flipped_image, flipped_points
            
            return flipped_image, points
        
        return image, points


class RotationWithPoints:
    def __init__(self, degrees=10, p=0.3):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image, points):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[:2]
                image = Image.fromarray(image)
            
            rotated_image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            
            if points.size > 0:
                theta = math.radians(angle)
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                
                cx, cy = w / 2, h / 2
                centered_points = points - np.array([cx, cy])
                rotation_matrix = np.array([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
                rotated_points = centered_points @ rotation_matrix.T + np.array([cx, cy])
                
                mask = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < w) & \
                       (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < h)
                
                rotated_points = rotated_points[mask]
                if rotated_points.size == 0:
                    rotated_points = np.zeros((0, 2), dtype=np.float32)
                
                return rotated_image, rotated_points.astype(np.float32)
            
            return rotated_image, points
        
        return image, points


class ShanghaiTechDataset(Dataset):
    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",
        transform: Optional[transforms.Compose] = None,
        target_input_size: Tuple[int, int] = (384, 384),
        sigma: float = 5.0,
        target_density_map_size: Optional[Tuple[int, int]] = None,
        return_count: bool = False,
        augment: bool = True,
        augment_factor: int = 1,
    ):
        self.root = root
        self.part = part
        self.split = split
        self.target_input_size = target_input_size
        self.sigma = sigma
        self.target_density_map_size = target_density_map_size or target_input_size
        self.return_count = return_count
        self.augment = augment and (split == "train_data")
        self.augment_factor = augment_factor

        self.images_dir = os.path.join(root, part, split, "images")
        self.gt_dir = os.path.join(root, part, split, "ground_truth")

        self.image_files = sorted(f for f in os.listdir(self.images_dir)
                                  if f.endswith(".jpg"))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

        # Setup transforms
        if self.augment:
            self.geometric_transforms = [
                RandomCropWithPoints(target_input_size, allow_overlap=True),
                HorizontalFlipWithPoints(p=0.5),
                RotationWithPoints(degrees=10, p=0.3),
            ]
            
            self.color_transform = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(target_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_points(mat_path: str) -> np.ndarray:
        mat = scipy.io.loadmat(mat_path)
        try:
            points = mat["image_info"][0, 0][0, 0]['location'].astype(np.float32)
        except (IndexError, ValueError, TypeError, KeyError):
            points = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        return points

    @staticmethod
    def _scale_points(points: np.ndarray,
                      orig_wh: Tuple[int, int],
                      to_wh: Tuple[int, int]) -> np.ndarray:
        if points.ndim == 0 or points.size == 0:
            return np.array([], dtype=np.float32).reshape(0, 2)
        
        ow, oh = orig_wh
        tw, th = to_wh
        sx = tw / ow if ow > 0 else 1.0
        sy = th / oh if oh > 0 else 1.0
        
        return np.stack([points[:, 0] * sx, points[:, 1] * sy], axis=1)

    def __len__(self) -> int:
        return len(self.image_files) * self.augment_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = idx % len(self.image_files)
        
        img_name = self.image_files[real_idx]
        img_path = os.path.join(self.images_dir, img_name)
        mat_name = f"GT_{os.path.splitext(img_name)[0]}.mat"
        mat_path = os.path.join(self.gt_dir, mat_name)

        img = self._load_image(img_path)
        orig_w, orig_h = img.size
        
        if os.path.exists(mat_path):
            points = self._load_points(mat_path)
        else:
            points = np.zeros((0, 2), dtype=np.float32)

        if self.augment:
            # Apply geometric transforms
            for transform in self.geometric_transforms:
                img, points = transform(img, points)
            
            # Apply color transforms
            img = self.color_transform(img)
            img_tensor = self.to_tensor(img)
            
            # Scale to density map size if different
            if self.target_density_map_size != self.target_input_size:
                points = self._scale_points(points, self.target_input_size, self.target_density_map_size)
        else:
            img_tensor = self.image_transform(img)
            points = self._scale_points(points, (orig_w, orig_h), self.target_density_map_size)

        # Create density map
        density_map_h, density_map_w = self.target_density_map_size[1], self.target_density_map_size[0]
        density_map = create_density_map(
            centroids=points,
            img_size=(density_map_h, density_map_w),
            sigma=self.sigma,
        )

        if self.return_count:
            return img_tensor, torch.tensor([density_map.sum()], dtype=torch.float32)

        den_tensor = torch.from_numpy(density_map).unsqueeze(0)
        return img_tensor, den_tensor


class ShanghaiTechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_folder,
        part="part_A",
        validation_split=0.1,
        seed=42,
        sigma=5,
        return_count=False,
        batch_size=8,
        num_workers=4,
        device=None,
        target_input_size: Tuple[int, int] = (384, 384),
        target_density_map_size: Tuple[int, int] = (384, 384),
        augment=True,
        augment_factor=5,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.part = part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.seed = seed
        self.sigma = sigma
        self.return_count = return_count
        self.target_input_size = target_input_size
        self.target_density_map_size = target_density_map_size
        self.augment = augment
        self.augment_factor = augment_factor
        self.device = device or get_device()
        self.pin_memory = True if self.device != "mps" else False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.save_hyperparameters()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup(self, stage: Optional[str] = None):
        # Training dataset with augmentations
        dataset_for_train_split = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="train_data",
            target_input_size=self.target_input_size,
            sigma=self.sigma,
            target_density_map_size=self.target_density_map_size,
            return_count=self.return_count,
            augment=self.augment,
            augment_factor=self.augment_factor,
        )

        # Validation dataset without augmentations
        dataset_for_val_split = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="train_data",
            target_input_size=self.target_input_size,
            sigma=self.sigma,
            target_density_map_size=self.target_density_map_size,
            return_count=self.return_count,
            augment=False,
            augment_factor=1,
        )
        
        num_total_train_samples = len(dataset_for_val_split)  # Use non-augmented count
        val_size = int(num_total_train_samples * self.validation_split)
        train_size = num_total_train_samples - val_size

        train_subset, val_subset = random_split(
            dataset_for_val_split, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        self.train_dataset = Subset(dataset_for_train_split, train_subset.indices)
        self.val_dataset = Subset(dataset_for_val_split, val_subset.indices)
            
        self.test_dataset = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="test_data",
            target_input_size=self.target_input_size,
            sigma=self.sigma,
            target_density_map_size=self.target_density_map_size,
            return_count=self.return_count,
            augment=False,
            augment_factor=1,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    data_folder = "./data/ShanghaiTech"

    data_module = ShanghaiTechDataModule(
        data_folder=data_folder,
        part="part_A",
        batch_size=3,
        validation_split=0.1,
        sigma=5,
        target_density_map_size=(384, 384),
        target_input_size=(384, 384),
        return_count=False,
        augment=True,
        augment_factor=5,
    )