import os
import scipy.io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ShanghaiTechDataset(Dataset):
    """
    ShanghaiTech Crowd Counting dataset loader.

    Expects a directory structure:
        root/
          part_A/ or part_B/
            train_data/ or test_data/
              images/
                *.jpg
              ground_truth/
                *.mat
    """
    def __init__(self, root: str, part: str = 'part_A', split: str = 'train_data', transform=None):
        self.root = root
        self.part = part
        self.split = split
        self.transform = transform

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
        # load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # corresponding ground-truth .mat file
        mat_name = os.path.splitext(img_name)[0] + '.mat'
        mat_path = os.path.join(self.gt_dir, mat_name)
        mat = scipy.io.loadmat(mat_path)

        # ground truth density map stored under key 'density' or 'map'
        if 'density' in mat:
            den = mat['density']
        elif 'map' in mat:
            den = mat['map']
        else:
            den = next(v for v in mat.values() if isinstance(v, (list, tuple, np.ndarray)) and getattr(v, 'ndim', 0) == 2)

        if self.transform:
            img = self.transform(img)

        return img, den.astype('float32')