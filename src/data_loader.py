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
        self.target_size = target_size  # Target size for resizing

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
        
        # Get original dimensions for scaling the density map
        orig_width, orig_height = img.size
        
        # Resize image to target size
        img = img.resize(self.target_size, Image.BILINEAR)

        # Load ground truth
        mat_name = 'GT_' + os.path.splitext(img_name)[0] + '.mat'
        mat_path = os.path.join(self.gt_dir, mat_name)
        
        try:
            # Handle .mat file loading with extra debugging
            mat = scipy.io.loadmat(mat_path)
            
            # Print mat keys for debugging
            print(f"MAT keys: {[k for k in mat.keys() if not k.startswith('__')]}")
            
            # Look for point annotations first (more reliable)
            if 'image_info' in mat:
                try:
                    # This depends on the specific ShanghaiTech format
                    # Try different possible structures
                    if isinstance(mat['image_info'], np.ndarray) and mat['image_info'].size > 0:
                        if isinstance(mat['image_info'][0, 0], np.ndarray) and mat['image_info'][0, 0].size > 0:
                            # Format could be nested arrays
                            if len(mat['image_info'][0, 0].shape) > 1 and mat['image_info'][0, 0].shape[1] > 0:
                                # Format might be image_info[0,0][0,0][0]
                                try:
                                    points = mat['image_info'][0, 0][0, 0][0]
                                    print(f"Found points: {points.shape}")
                                except:
                                    # Another format might be direct
                                    points = mat['image_info'][0, 0]
                            else:
                                points = mat['image_info'][0, 0]
                        else:
                            points = mat['image_info'][0]
                    else:
                        points = mat['image_info']
                        
                    # Create density map from points
                    den = np.zeros((orig_height, orig_width), dtype=np.float32)
                    
                    # Process point coordinates
                    if isinstance(points, np.ndarray):
                        for point in points:
                            if isinstance(point, np.ndarray) and len(point) >= 2:
                                try:
                                    x, y = int(float(point[0])), int(float(point[1]))
                                    if 0 <= x < orig_width and 0 <= y < orig_height:
                                        den[y, x] = 1.0
                                except (ValueError, TypeError) as e:
                                    print(f"Error processing point {point}: {e}")
                                    continue
                    print(f"Created density map from points")
                except Exception as e:
                    print(f"Error extracting points: {e}")
                    # Fallback to other methods
                    den = np.ones((orig_height, orig_width), dtype=np.float32)
            # Try direct density map
            elif 'density' in mat:
                den = mat['density']
                print(f"Found density map, shape: {den.shape}, type: {den.dtype}")
            elif 'map' in mat:
                den = mat['map']
                print(f"Found map, shape: {den.shape}, type: {den.dtype}")
            else:
                # Try other fields
                for key in mat.keys():
                    if key.startswith('__'):  # Skip metadata
                        continue
                    val = mat[key]
                    if isinstance(val, np.ndarray) and val.ndim == 2:
                        den = val
                        print(f"Using {key} as density map, shape: {den.shape}, type: {den.dtype}")
                        break
                else:
                    # Last resort - create dummy density
                    print(f"Could not find density map, creating uniform map")
                    den = np.ones((orig_height, orig_width), dtype=np.float32)
                    
            # Ensure den is a proper 2D array of float32
            if not isinstance(den, np.ndarray):
                print(f"den is not an ndarray, type: {type(den)}")
                den = np.array([[float(den)]], dtype=np.float32)
            elif den.dtype == 'object':
                print(f"den has dtype object, converting elements")
                # Create a new array with proper type
                new_den = np.zeros(den.shape, dtype=np.float32)
                for i in range(den.shape[0]):
                    for j in range(den.shape[1]):
                        try:
                            # Try to extract a numeric value
                            if isinstance(den[i,j], (list, tuple)) and len(den[i,j]) > 0:
                                new_den[i,j] = float(den[i,j][0])
                            else:
                                new_den[i,j] = float(den[i,j])
                        except (ValueError, TypeError):
                            new_den[i,j] = 0.0
                den = new_den
            else:
                # Ensure the array is float32
                den = den.astype(np.float32)
                
            # Ensure den is 2D
            if den.ndim != 2:
                print(f"den has {den.ndim} dimensions, reshaping")
                if den.size == 1:
                    # Single value - create uniform map
                    total_count = float(den.item())
                    den = np.ones((orig_height, orig_width), dtype=np.float32) * (total_count / (orig_height * orig_width))
                else:
                    # Try to reshape if possible
                    den = den.reshape((orig_height, orig_width))
            
            # Resize density map to match target size
            if den.shape[0] != self.target_size[1] or den.shape[1] != self.target_size[0]:
                original_count = den.sum()
                # Create a new resized array
                resized_den = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
                
                # Simple resize by averaging (more stable than PIL for density maps)
                h_ratio = den.shape[0] / self.target_size[1]
                w_ratio = den.shape[1] / self.target_size[0]
                
                for i in range(self.target_size[1]):
                    for j in range(self.target_size[0]):
                        h_start = int(i * h_ratio)
                        h_end = int((i + 1) * h_ratio)
                        w_start = int(j * w_ratio)
                        w_end = int((j + 1) * w_ratio)
                        resized_den[i, j] = den[h_start:h_end, w_start:w_end].mean()
                
                den = resized_den
                
                # Rescale to preserve total count
                if den.sum() > 0:  # Avoid division by zero
                    den = den * (original_count / den.sum())
            
            # Apply image transform
            if self.transform:
                img = self.transform(img)
                
            # Final check to ensure proper types
            if not isinstance(img, torch.Tensor):
                print(f"Converting image to tensor")
                img = transforms.ToTensor()(img)
                
            # Ensure den is a tensor
            den_tensor = torch.from_numpy(den)
            
            return img, den_tensor
            
        except Exception as e:
            print(f"Error processing {mat_path}: {e}")
            # Create dummy data as fallback
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            # Create a dummy density map
            dummy_den = torch.zeros((self.target_size[1], self.target_size[0]), dtype=torch.float32)
            return img, dummy_den