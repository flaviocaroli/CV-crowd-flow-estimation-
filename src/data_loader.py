import os
from typing import Optional, Tuple

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from src.utils import create_density_map, show_samples_from_loaders, get_device


class ShanghaiTechDataset(Dataset):
    """
    Loads one sample (image, density map *or* count) from the ShanghaiTech
    dataset.  The density map is generated **directly** at `density_map_size`.
    """

    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",
        transform: Optional[transforms.Compose] = None,
        input_size: Tuple[int, int] = (384, 384),               # (W, H)
        sigma: float = 5.0,
        density_map_size: Optional[Tuple[int, int]] = None,      # (W, H)
        return_count: bool = False,
    ):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform or get_default_transform(input_size)
        self.input_size = input_size
        self.sigma = sigma
        self.density_map_size = density_map_size or input_size
        self.return_count = return_count

        # --- paths ---------------------------------------------------------
        self.images_dir = os.path.join(root, part, split, "images")
        self.gt_dir = os.path.join(root, part, split, "ground_truth")

        self.image_files = sorted(f for f in os.listdir(self.images_dir)
                                  if f.endswith(".jpg"))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

    # ======================================================================
    # internal helpers
    # ======================================================================
    @staticmethod
    def _load_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_points(mat_path: str) -> np.ndarray:
        mat = scipy.io.loadmat(mat_path)
        # shape (N, 2)
        return mat["image_info"][0, 0][0, 0][0].astype(np.float32)

    @staticmethod
    def _scale_points(points: np.ndarray,
                      orig_wh: Tuple[int, int],
                      to_wh: Tuple[int, int]) -> np.ndarray:
        """Scale (x, y) coordinates from original W×H to new W×H."""
        ow, oh = orig_wh
        tw, th = to_wh
        sx, sy = tw / ow, th / oh
        return np.stack([points[:, 0] * sx, points[:, 1] * sy], axis=1)

    # ======================================================================
    # main API
    # ======================================================================
    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---------- file names --------------------------------------------
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mat_name = f"GT_{os.path.splitext(img_name)[0]}.mat"
        mat_path = os.path.join(self.gt_dir, mat_name)

        # ---------- image -------------------------------------------------
        img = self._load_image(img_path)
        orig_wh = img.size                                    # (W, H)
        img = img.resize(self.input_size, Image.Resampling.BILINEAR)

        img_tensor = self.transform(img)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)

        # ---------- ground-truth points ----------------------------------
        points = self._load_points(mat_path)

        # 1️⃣ scale to *image* resolution for potential visualisation
        #    (not strictly needed for training)
        # scaled_for_image = self._scale_points(points, orig_wh, self.input_size)

        # 2️⃣ scale *again* to the density-map resolution
        scaled_for_density = self._scale_points(points, orig_wh, self.density_map_size)

        # ---------- density map ------------------------------------------
        density_map = create_density_map(
            centroids=scaled_for_density,
            img_size=self.density_map_size,   # (H, W)
            sigma=self.sigma,
        )                                           # sum == crowd count

        if self.return_count:
            return img_tensor, torch.tensor([density_map.sum()], dtype=torch.float32)

        den_tensor = torch.from_numpy(density_map).unsqueeze(0)  # (1, H, W)
        return img_tensor, den_tensor


def get_default_transform(input_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )


class ShanghaiTechDataModule(pl.LightningDataModule):
    """
    The main ShanghaiTechDataModule. When iterated over, returns batches of (X, y) of sequence and target sequence shifted by one.
    """

    def __init__(
        self,
        data_folder,
        part="part_A",
        validation_split=0.1,
        seed=42,
        sigma=5,  # Added parameter
        return_count=False,  # Added parameter
        batch_size=8,  # Made configurable
        num_workers=4,  # Made configurable
        device=None,  # Added parameter
        input_size=(384, 384),  # Added parameter
        density_map_size=(384, 384),  # Added parameter
    ):
        self.data_folder = data_folder
        self.part = part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.seed = seed
        self.sigma = sigma
        self.return_count = return_count
        self.input_size = input_size
        self.density_map_size = density_map_size
        self.device = device or get_device()
        self.pin_memory = True if self.device != "mps" else False
        self.transform: Optional[transforms.Compose] = None or get_default_transform(input_size=self.input_size)
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = True

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup(self, stage=None):
        full_dataset = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="train_data",
            transform=self.transform,
            input_size=self.input_size,
            sigma=self.sigma,
            density_map_size=self.density_map_size,
            return_count=self.return_count,
        )

        train_size = int(len(full_dataset) * (1 - self.validation_split))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.test_dataset = ShanghaiTechDataset(
            root=self.data_folder,
            part=self.part,
            split="test_data",
            transform=self.transform,
            input_size=self.input_size,
            sigma=self.sigma,
            density_map_size=self.density_map_size,
            return_count=self.return_count,
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
        batch_size=3,  # for plotting
        validation_split=0.1,
        sigma=5,
        density_map_size=(384, 384),
        input_size=(384, 384),
        return_count=False,
    )

    # Show train, val, test samples
    show_samples_from_loaders(data_module)
