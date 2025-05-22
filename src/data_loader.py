import os
from typing import Optional, Tuple, List, Dict # Added List and Dict

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset # Added Subset
from torchvision import transforms
import pytorch_lightning as pl
import albumentations as A # Added

from src.utils import create_density_map, show_samples_from_loaders, get_device
from src.custom_transforms import build_transforms # Added


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
        transform: Optional[A.Compose] = None,  # Changed type hint
        target_input_size: Tuple[int, int] = (384, 384),        # (W, H) Renamed
        sigma: float = 5.0,
        target_density_map_size: Optional[Tuple[int, int]] = None, # (W, H) Renamed
        return_count: bool = False,
    ):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform # Removed get_default_transform call
        self.target_input_size = target_input_size # Renamed
        self.sigma = sigma
        self.target_density_map_size = target_density_map_size or target_input_size # Renamed
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
        if points.ndim == 0 or points.size == 0: # Handle empty or invalid points
            return np.array([], dtype=np.float32).reshape(0, 2)
        
        ow, oh = orig_wh
        tw, th = to_wh

        # Avoid division by zero if original dimensions are zero
        sx = tw / ow if ow > 0 else 1.0
        sy = th / oh if oh > 0 else 1.0
        
        scaled_points = np.stack([points[:, 0] * sx, points[:, 1] * sy], axis=1)
        return scaled_points

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
        img_pil = self._load_image(img_path) # Load as PIL image first
        img_np = np.array(img_pil)          # Convert to NumPy array for Albumentations
        orig_w, orig_h = img_pil.size       # Original (W, H)

        # ---------- ground-truth points ----------------------------------
        # Points are (N, 2) where columns are (x, y)
        points = self._load_points(mat_path) 
        # Albumentations expects keypoints as List[tuple[float, float]] or List[List[float]]
        points_list = points.tolist() if points.size > 0 else []

        current_img_dims_for_scaling: Tuple[int, int] # Will hold (W,H) of the image points_transformed are relative to

        if self.transform:
            augmented = self.transform(image=img_np, keypoints=points_list)
            img_transformed_tensor = augmented['image'] # Should be (C, H, W) tensor by ToTensorV2
            points_transformed = np.array(augmented['keypoints']) # Back to numpy (N,2) or (0,2)
            
            # Albumentations ToTensorV2 output is (C, H, W)
            augmented_h, augmented_w = img_transformed_tensor.shape[1], img_transformed_tensor.shape[2]
            current_img_dims_for_scaling = (augmented_w, augmented_h)
        else:
            # Fallback to basic PIL resize and torchvision ToTensor if no Albumentations transform
            img_pil_resized = img_pil.resize(self.target_input_size, Image.Resampling.BILINEAR) # target_input_size is (W,H)
            img_transformed_tensor = transforms.ToTensor()(img_pil_resized) # (C, H, W)
            
            # Points need to be scaled from original image to target_input_size
            points_transformed = self._scale_points(points, 
                                                    orig_wh=(orig_w, orig_h), 
                                                    to_wh=self.target_input_size)
            current_img_dims_for_scaling = self.target_input_size # (W,H)

        # Scale points (which are relative to current_img_dims_for_scaling) to target_density_map_size
        scaled_for_density = self._scale_points(
            points_transformed,
            orig_wh=current_img_dims_for_scaling, # (W,H) of the image points_transformed are relative to
            to_wh=self.target_density_map_size      # Target (W,H) for density map
        )

        # ---------- density map ------------------------------------------
        # create_density_map expects img_size as (H, W)
        # self.target_density_map_size is (W, H)
        density_map_h, density_map_w = self.target_density_map_size[1], self.target_density_map_size[0]
        density_map = create_density_map(
            centroids=scaled_for_density,
            img_size=(density_map_h, density_map_w), 
            sigma=self.sigma,
        )

        if self.return_count:
            return img_transformed_tensor, torch.tensor([density_map.sum()], dtype=torch.float32)

        den_tensor = torch.from_numpy(density_map).unsqueeze(0)  # (1, H, W)
        return img_transformed_tensor, den_tensor


# Removed get_default_transform function as it's replaced by Albumentations pipeline.

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
        sigma=5,
        return_count=False,
        batch_size=8,
        num_workers=4,
        device=None,
        target_input_size: Tuple[int, int] = (224, 224),  # Renamed, added type hint for clarity
        target_density_map_size: Tuple[int, int] = (224, 224),  # Renamed, added type hint
        augmentation_config: Optional[Dict] = None, # Added
    ):
        super().__init__() # Added for pl.LightningDataModule best practices
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
        self.augmentation_config = augmentation_config # Stored
        self.device = device or get_device()
        self.pin_memory = True if self.device != "mps" else False
        # self.transform: Optional[A.Compose] = None # Removed
        self.train_transform: Optional[A.Compose] = None # Added
        self.val_transform: Optional[A.Compose] = None # Added
        self.allow_zero_length_dataloader_with_multiple_devices = False # Default from pl
        # self._log_hyperparams = True # This is often handled automatically or via save_hyperparameters()
        self.save_hyperparameters() # Recommended for PyTorch Lightning

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup(self, stage: Optional[str] = None): # Added type hint for stage
        # Build transforms
        aug_config = self.augmentation_config if self.augmentation_config is not None else {}
        self.train_transform, self.val_transform = build_transforms(
            config=aug_config,
            target_input_size=self.target_input_size
        )

        if stage == "fit" or stage is None:
            # Dataset for training data with training transforms
            dataset_for_train_split = ShanghaiTechDataset(
                root=self.data_folder,
                part=self.part,
                split="train_data",
                transform=self.train_transform,
                target_input_size=self.target_input_size,
                sigma=self.sigma,
                target_density_map_size=self.target_density_map_size,
                return_count=self.return_count,
            )

            # Dataset for training data with validation transforms (for val split)
            dataset_for_val_split = ShanghaiTechDataset(
                root=self.data_folder,
                part=self.part,
                split="train_data", # Still from "train_data" directory
                transform=self.val_transform, # Use validation transforms
                target_input_size=self.target_input_size,
                sigma=self.sigma,
                target_density_map_size=self.target_density_map_size,
                return_count=self.return_count,
            )
            
            num_total_train_samples = len(dataset_for_train_split)
            val_size = int(num_total_train_samples * self.validation_split)
            train_size = num_total_train_samples - val_size

            indices = torch.randperm(num_total_train_samples, generator=torch.Generator().manual_seed(self.seed)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_dataset = Subset(dataset_for_train_split, train_indices)
            self.val_dataset = Subset(dataset_for_val_split, val_indices)
            
        if stage == "test" or stage is None:
            self.test_dataset = ShanghaiTechDataset(
                root=self.data_folder,
                part=self.part,
                split="test_data",
                transform=self.val_transform, # Use validation transforms for testing
                target_input_size=self.target_input_size,
                sigma=self.sigma,
                target_density_map_size=self.target_density_map_size,
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
        target_density_map_size=(384, 384),
        target_input_size=(384, 384),
        augmentation_config={"hflip_prob": 0.5, "smart_crop": {"p": 0.5}}, # Example aug config
        return_count=False,
    )
    
    # Setup the datamodule (this will build transforms and datasets)
    data_module.setup(stage="fit") # Or data_module.setup() for all
    data_module.setup(stage="test")


    # Show train, val, test samples
    # To show samples, data_module.setup() must have been called.
    show_samples_from_loaders(data_module)
