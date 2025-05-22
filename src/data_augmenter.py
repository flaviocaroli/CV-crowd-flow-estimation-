from pathlib import Path
from typing import Tuple, List

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm


def load_points(mat_path: str) -> np.ndarray:
    """
    Load point annotations from .mat file.
    
    Args:
        mat_path: Path to the .mat file containing ground truth annotations.
        
    Returns:
        Array of shape (N, 2) containing point coordinates.
    """
    mat = scipy.io.loadmat(mat_path)
    try:
        points = mat["image_info"][0, 0][0, 0]['location'].astype(np.float32)
    except (IndexError, ValueError, TypeError):
        points = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
    return points


def save_points(points: np.ndarray, mat_path: str) -> None:
    """
    Save point annotations to .mat file in ShanghaiTech format.
    
    Args:
        points: Array of shape (N, 2) containing point coordinates.
        mat_path: Path where the .mat file will be saved.
    """
    if points.size == 0:
        points = np.zeros((0, 2), dtype=np.float64)
    elif points.ndim == 1:
        points = points.reshape(1, -1)
    
    points = points.astype(np.float64)
    
    dt = np.dtype([('location', 'O'), ('number', 'O')])
    struct_arr = np.zeros((1, 1), dtype=dt)
    struct_arr[0, 0]['location'] = points
    struct_arr[0, 0]['number'] = np.array(len(points))
    
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = struct_arr
    
    scipy.io.savemat(mat_path, {'image_info': image_info})


def extract_patches(
    image: np.ndarray, 
    patch_size: int = 384, 
    allow_overlap: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract patches from an image.
    
    Args:
        image: Input image array of shape (H, W) or (H, W, C).
        patch_size: Size of square patches to extract.
        allow_overlap: If True, use minimum overlap to cover entire image.
                      If False, extract only non-overlapping patches.
    
    Returns:
        Tuple of (patches, coordinates) where:
        - patches: List of extracted patches, each of shape (patch_size, patch_size, 3).
        - coordinates: List of (y, x) top-left coordinates for each patch.
    """
    h, w = image.shape[:2]
    
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    patches = []
    coordinates = []
    
    if h < patch_size or w < patch_size:
        padded = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
        actual_h = min(h, patch_size)
        actual_w = min(w, patch_size)
        padded[:actual_h, :actual_w] = image[:actual_h, :actual_w]
        patches.append(padded)
        coordinates.append((0, 0))
        return patches, coordinates
    
    if not allow_overlap:
        n_patches_y = h // patch_size
        n_patches_x = w // patch_size
        
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y = i * patch_size
                x = j * patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coordinates.append((y, x))
    else:
        n_patches_y = int(np.ceil(h / patch_size))
        n_patches_x = int(np.ceil(w / patch_size))
        
        if n_patches_y > 1:
            stride_y = (h - patch_size) / (n_patches_y - 1)
        else:
            stride_y = 0
            
        if n_patches_x > 1:
            stride_x = (w - patch_size) / (n_patches_x - 1)
        else:
            stride_x = 0
        
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y = int(min(i * stride_y, h - patch_size))
                x = int(min(j * stride_x, w - patch_size))
                
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coordinates.append((y, x))
    
    return patches, coordinates


def filter_points_for_patch(
    points: np.ndarray, 
    patch_coord: Tuple[int, int], 
    patch_size: int = 384
) -> np.ndarray:
    """
    Filter points that fall within a patch and adjust coordinates.
    
    Args:
        points: Array of shape (N, 2) containing point coordinates.
        patch_coord: Tuple of (y, x) representing top-left corner of patch.
        patch_size: Size of the square patch.
    
    Returns:
        Array of shape (M, 2) containing filtered and adjusted point coordinates.
    """
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    y, x = patch_coord
    
    mask = (points[:, 0] >= x) & (points[:, 0] < x + patch_size) & \
           (points[:, 1] >= y) & (points[:, 1] < y + patch_size)
    
    filtered_points = points[mask].copy()
    
    if filtered_points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    filtered_points[:, 0] -= x
    filtered_points[:, 1] -= y
    
    return filtered_points.astype(np.float32)


def process_dataset(
    input_dir: str, 
    output_dir: str, 
    patch_size: int = 384, 
    allow_overlap: bool = False
) -> None:
    """
    Process entire ShanghaiTech dataset to create patches.
    
    Args:
        input_dir: Path to input ShanghaiTech directory.
        output_dir: Path to output directory for patches.
        patch_size: Size of square patches to extract.
        allow_overlap: If True, use minimum overlap to cover entire images.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for part in ['part_A', 'part_B']:
        part_path = input_path / part
        if not part_path.exists():
            print(f"Skipping {part} - not found")
            continue
            
        print(f"\nProcessing {part}...")
        
        for split in ['train_data', 'test_data']:
            split_path = part_path / split
            if not split_path.exists():
                print(f"Skipping {split} - not found")
                continue
                
            print(f"  Processing {split}...")
            
            images_dir = split_path / 'images'
            gt_dir = split_path / 'ground_truth'
            
            out_images_dir = output_path / part / split / 'images'
            out_gt_dir = output_path / part / split / 'ground_truth'
            out_images_dir.mkdir(parents=True, exist_ok=True)
            out_gt_dir.mkdir(parents=True, exist_ok=True)
            
            image_files = sorted(list(images_dir.glob('*.jpg')))
            
            for img_path in tqdm(image_files, desc=f"    {part}/{split}"):
                image = np.array(Image.open(img_path))
                
                img_name = img_path.stem
                mat_name = f'GT_{img_name}.mat'
                mat_path = gt_dir / mat_name
                
                if mat_path.exists():
                    points = load_points(str(mat_path))
                else:
                    print(f"Warning: No ground truth found for {img_name}")
                    points = np.array([], dtype=np.float32).reshape(0, 2)
                
                patches, coordinates = extract_patches(image, patch_size, allow_overlap)
                
                for patch_idx, (patch, coord) in enumerate(zip(patches, coordinates)):
                    patch_name = f'{img_name}_patch_{patch_idx:03d}'
                    patch_img_path = out_images_dir / f'{patch_name}.jpg'
                    patch_mat_path = out_gt_dir / f'GT_{patch_name}.mat'
                    
                    Image.fromarray(patch).save(patch_img_path)
                    
                    patch_points = filter_points_for_patch(points, coord, patch_size)
                    save_points(patch_points, str(patch_mat_path))
            
            print(f"    Completed {split}: {len(image_files)} images -> {len(list(out_images_dir.glob('*.jpg')))} patches")


if __name__ == "__main__":

    input_dir = "./data/ShanghaiTech"
    output_dir = "./data/ShanghaiTech_patches"
    patch_size = 384
    overlap = True

    print("Starting augmentation...")
    
    process_dataset(input_dir, output_dir, patch_size, overlap)
    
    print("\nAugmentation complete!")