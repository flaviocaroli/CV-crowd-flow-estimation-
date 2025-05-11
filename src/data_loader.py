import os

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.utils import create_density_map


class ShanghaiTechDataset(Dataset):
    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",  # "train_data", "test_data" or "val_data"
        transform=None,
        target_size=(384, 384),
        sigma=5,  # Added parameter for Gaussian kernel
        density_map_size=None,  # Added parameter for output density map size
        return_count=False,  # Added parameter to toggle between density map and count
    ):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.density_map_size = density_map_size or target_size  # Default to target_size if None
        self.return_count = return_count

        # paths
        self.images_dir = os.path.join(root, part, split, "images")
        self.gt_dir = os.path.join(root, part, split, "ground_truth")

        # list all image files
        self.image_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        )
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Get original dimensions
        orig_width, orig_height = img.size

        # Resize image to target size
        img = img.resize(self.target_size, Image.BILINEAR)

        # Load ground truth
        mat_name = "GT_" + os.path.splitext(img_name)[0] + ".mat"
        mat_path = os.path.join(self.gt_dir, mat_name)
        mat = scipy.io.loadmat(mat_path)

        # Extract points from image_info
        points = mat["image_info"][0, 0][0, 0][0]

        # Scale points to match target_size
        scaled_points = []
        for point in points:
            x, y = point[0], point[1]
            # Scale coordinates
            x_scaled = x * (self.target_size[0] / orig_width)
            y_scaled = y * (self.target_size[1] / orig_height)
            scaled_points.append([x_scaled, y_scaled])
        
        scaled_points = np.array(scaled_points)

        # Create density map using utility function
        density_map = create_density_map(
            centroids=scaled_points, 
            img_size=self.target_size[::-1],  # (height, width)
            sigma=self.sigma
        )
        
        # Resize density map if needed
        if self.density_map_size != self.target_size:
            density_map_pil = Image.fromarray(density_map)
            density_map_pil = density_map_pil.resize(self.density_map_size, Image.BILINEAR)
            
            # Preserve count after resizing
            original_count = density_map.sum()
            density_map = np.array(density_map_pil)
            if density_map.sum() > 0:
                density_map = density_map * (original_count / density_map.sum())

        # Apply transform to image
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Convert density map to tensor or get count
        if self.return_count:
            count = density_map.sum()
            return img, torch.tensor([count], dtype=torch.float32)
        else:
            den_tensor = torch.from_numpy(density_map).unsqueeze(0)  # Add channel dimension
            return img, den_tensor


class ShanghaiTechDataModule:
    """
    The main datamodule. When iterated over, returns batches of (X, y) of sequence and target sequence shifted by one.
    """

    def __init__(
        self, 
        data_folder, 
        part="part_A",
        validation_split=0.1, 
        seed=42,
        sigma=5,  # Added parameter
        density_map_size=None,  # Added parameter
        return_count=False,  # Added parameter
        batch_size=8,  # Made configurable
        num_workers=4  # Made configurable
    ):
        self.data_folder = data_folder
        self.part = part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.seed = seed
        self.sigma = sigma
        self.density_map_size = density_map_size
        self.return_count = return_count

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Setup image transforms
        image_transform = transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor()
        ])

        # Load the dataset
        self.train_data = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="train_data",
            transform=image_transform,
            sigma=self.sigma,
            density_map_size=self.density_map_size,
            return_count=self.return_count
        )

        # Split the dataset into train and validation sets
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_data,
            [
                int(len(self.train_data) * (1 - self.validation_split)),
                int(len(self.train_data) * self.validation_split),
            ],
        )

        self.test_dataset = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="test_data",
            transform=image_transform,
            sigma=self.sigma,
            density_map_size=self.density_map_size,
            return_count=self.return_count
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )