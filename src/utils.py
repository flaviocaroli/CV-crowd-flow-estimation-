import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def get_device():
        # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


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


def show_samples_from_loaders(data_module, sigma=5):
    """
    Show samples from train, val, and test loaders for every element in the batch.
    Display the image and its density map side by side in the same figure.
    Arrange rows as batch elements and columns as loaders.
    """
    loaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }

    # Get one batch from each loader
    batches = {}
    for loader_name, loader in loaders.items():
        for batch in loader:
            batches[loader_name] = batch
            break  # Only first batch

    # Assume all batches have the same batch size
    batch_size = next(iter(batches.values()))[0].shape[0]
    fig, axes = plt.subplots(
        nrows=batch_size,
        ncols=len(loaders),
        figsize=(4 * len(loaders), 3 * batch_size),
        dpi=100
    )

    if batch_size == 1:
        axes = np.expand_dims(axes, 0)
    if len(loaders) == 1:
        axes = np.expand_dims(axes, 1)

    loader_names = list(loaders.keys())
    for col, loader_name in enumerate(loader_names):
        imgs, densities = batches[loader_name]
        for row in range(batch_size):
            ax = axes[row, col]
            img_np = imgs[row].permute(1, 2, 0).numpy()
            density_np = densities[row, 0].numpy()
            ax.imshow(img_np)
            ax.imshow(density_np, cmap="jet", alpha=0.5)
            ax.axis("off")
            if row == 0:
                ax.set_title(loader_name)
            if col == 0:
                ax.set_ylabel(f"Sample {row}")

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

def plot_density_predictions(model, dataloader, device, num_samples=5, figsize=(15, 5)):
    """
    Iterates over a dataloader and displays input images, ground-truth density maps,
    and model predictions for each sample. Each density map has its sum annotated.
    """
    model.eval()
    shown = 0

    with torch.no_grad():
        for imgs, gt_maps in dataloader:
            if shown >= num_samples:
                break

            img = imgs[0]              # (3, H, W)
            gt_map = gt_maps[0]        # (1, H, W)

            # Inference
            input_img = img.unsqueeze(0).to(device)    # [1,3,H,W]
            pred_map = model(input_img)                # [1,1,H,W] or [1,H,W]
            pred_map = pred_map.squeeze().cpu().numpy()  # [H,W]

            # Prepare visuals
            rgb = img.permute(1, 2, 0).cpu().numpy()
            gt = gt_map.squeeze(0).cpu().numpy()
            gt_sum = gt.sum()
            pred_sum = pred_map.sum()

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            axes[0].imshow(rgb)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(gt, cmap="jet")
            axes[1].set_title("Ground-Truth Density")
            axes[1].axis("off")
            # Add GT count as a label
            axes[1].text(
                0.02, 0.95, f"Sum: {gt_sum:.2f}",
                transform=axes[1].transAxes,
                fontsize=12, color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
                verticalalignment="top", horizontalalignment="left"
            )

            axes[2].imshow(pred_map, cmap="jet")
            axes[2].set_title("Predicted Density")
            axes[2].axis("off")
            # Add Pred count as a label
            axes[2].text(
                0.02, 0.95, f"Sum: {pred_sum:.2f}",
                transform=axes[2].transAxes,
                fontsize=12, color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
                verticalalignment="top", horizontalalignment="left"
            )

            plt.tight_layout()
            plt.show()
            shown += 1

def plot_all_decoder_predictions(model, dataloader, device, figsize_per_pred=(5, 5), i=0):
    """
    Plots input image, ground-truth density, and the decoder predictions
    at each upsampling stage (for just one image).
    """
    model.eval()
    with torch.no_grad():
        for idx, el in enumerate(dataloader):
            imgs, gt_maps = el
            if idx == i:
                break
        
        img = imgs[0]          # (3, H, W)
        gt_map = gt_maps[0]    # (1, H, W)
        input_img = img.unsqueeze(0).to(device)

        # Run model to get intermediate predictions
        preds = model(input_img, return_intermediates=True)
        # preds: list of [1, C, h, w], except last: [1,1,H,W]

        # Prepare visuals
        rgb = img.permute(1, 2, 0).cpu().numpy()
        gt = gt_map.squeeze(0).cpu().numpy()
        gt_sum = gt.sum()

        # Plotting
        n_preds = len(preds)
        fig, axes = plt.subplots(1, 2 + n_preds, figsize=(figsize_per_pred[0]*(2+n_preds), figsize_per_pred[1]))
        # Input image
        axes[0].imshow(rgb)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # GT density
        axes[1].imshow(gt, cmap="jet")
        axes[1].set_title("Ground-Truth Density")
        axes[1].axis("off")
        axes[1].text(
            0.02, 0.95, f"Sum: {gt_sum:.2f}",
            transform=axes[1].transAxes,
            fontsize=12, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
            verticalalignment="top", horizontalalignment="left"
        )

        # Decoder outputs
        for i, p in enumerate(preds):
            # Pick first channel if not already 1-channel (e.g., up layers)
            pm = p[0]
            if pm.shape[0] > 1:
                pm = pm[0]
            pm = pm.cpu().numpy()
            pred_sum = pm.sum()
            axes[2+i].imshow(pm.squeeze(), cmap="jet")
            axes[2+i].set_title(f"Dec {i+1}\nSum: {pred_sum:.2f}")
            axes[2+i].axis("off")
            axes[2+i].text(
                0.02, 0.95, f"Sum: {pred_sum:.2f}",
                transform=axes[2+i].transAxes,
                fontsize=12, color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
                verticalalignment="top", horizontalalignment="left"
            )
        plt.tight_layout()
        plt.show()