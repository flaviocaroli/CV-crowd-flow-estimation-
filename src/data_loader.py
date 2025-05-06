import os

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ShanghaiTechDataset(Dataset):
    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",  # "train_data", "test_data" or "val_data"
        transform=None,
        target_size=(384, 384),
    ):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform
        self.target_size = target_size

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

        # Create density map from point annotations
        den = np.zeros((orig_height, orig_width), dtype=np.float32)

        # Extract points from image_info
        points = mat["image_info"][0, 0][0, 0][0]

        # Place points on density map
        for point in points:
            x, y = (
                int(min(point[0], orig_width - 1)),
                int(min(point[1], orig_height - 1)),
            )
            den[y, x] = 1.0

        # Resize density map to match target size
        if den.shape[0] != self.target_size[1] or den.shape[1] != self.target_size[0]:
            original_count = den.sum()  # Preserve total count

            # Create resized density map
            resized_den = np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.float32
            )

            # Simple resize by scaling coordinates
            y_ratio = self.target_size[1] / orig_height
            x_ratio = self.target_size[0] / orig_width

            for point in points:
                x, y = point[0] * x_ratio, point[1] * y_ratio
                x, y = (
                    int(min(x, self.target_size[0] - 1)),
                    int(min(y, self.target_size[1] - 1)),
                )
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


class ShanghaiTechDataModule:
    """
    The main datamodule. When iterated over, returns batches of (X, y) of sequence and target sequence shifted by one.
    """

    def __init__(self, data_folder, part, validation_split=0.1, seed=42):
        self.data_folder = data_folder
        self.batch_size = 8
        self.num_workers = 4
        self.validation_split = validation_split
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Load the dataset
        self.train_data = ShanghaiTechDataset(
            root=self.data_folder,
            part=part,
            split="train_data",
            transform=transforms.Compose(
                [transforms.Resize((384, 384)), transforms.ToTensor()]
            ),
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
            part=part,
            split="test_data",
            transform=transforms.Compose(
                [transforms.Resize((384, 384)), transforms.ToTensor()]
            ),
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
