from scipy.ndimage import gaussian_filter
import numpy as np
import cv2

def create_density_map(centroids, img_size, sigma=5):
    """
    Create a density map from the given centroids.

    Parameters:
    - centroids: np.ndarray of shape (N, 2) where N is the number of points
    - img_size: tuple (height, width) of the image size
    - sigma: standard deviation for Gaussian kernel

    Returns:
    - density_map: np.ndarray of shape (height, width)
    """
    height, width = img_size
    density_map = np.zeros((height, width), dtype=np.float32)

    for centroid in centroids:
        x = int(round(centroid[0]))
        y = int(round(centroid[1]))
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] += 1

    # Smooth using Gaussian filter
    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map