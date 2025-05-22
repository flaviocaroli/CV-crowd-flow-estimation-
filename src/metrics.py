import torch
import torch.nn.functional as F


def pixelwise_mse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Mean Squared Error (MSE) between pred and target density maps.

    Args:
        pred (Tensor): Predicted density maps, shape (B, H, W) or (B, 1, H, W).
        target (Tensor): Ground truth density maps, same shape as pred.
        reduction (str): 'mean' | 'sum' | 'none'.

    Returns:
        Tensor: MSE loss. If 'none', returns per-sample MSE of shape (B,). Otherwise, scalar.
    """
    return F.mse_loss(pred, target, reduction=reduction)


def pixelwise_mae(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Mean Absolute Error (MAE).
    """
    return F.l1_loss(pred, target, reduction=reduction)


def pixelwise_rmse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Root Mean Squared Error (RMSE).
    """
    mse = pixelwise_mse(pred, target, reduction)
    return torch.sqrt(mse)


def spatial_count(density_map: torch.Tensor) -> torch.Tensor:
    """
    Sum the density map across spatial dimensions to get counts per sample.

    Args:
        density_map (Tensor): Shape (B, H, W) or (B, 1, H, W).

    Returns:
        Tensor: Counts per sample, shape (B,).
    """

    # If channel dimension exists, remove it
    if density_map.ndim == 4 and density_map.shape[1] == 1:
        x = density_map.squeeze(1)
    else:
        x = density_map
    # sum over spatial dims
    # assumes x.shape = (B, H, W)
    return x.view(x.size(0), -1).sum(dim=1)  # (B,)


def count_mae(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute error in total counts between pred and target.
    """
    c_pred = spatial_count(pred)
    c_gt = spatial_count(target)
    return F.l1_loss(c_pred, c_gt, reduction=reduction)


# New functions for count MSE and RMSE
def count_mse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE) of total counts between pred and target.
    """
    c_pred = spatial_count(pred)
    c_gt = spatial_count(target)
    return F.mse_loss(c_pred, c_gt, reduction=reduction)


def count_rmse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE) of total counts between pred and target.
    """
    mse = count_mse(pred, target, reduction)
    return torch.sqrt(mse)


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(4, 1, 384, 384)
    target = torch.randn(4, 1, 384, 384)

    print("Pixelwise MSE:", pixelwise_mse(pred, target))
    print("Pixelwise MAE:", pixelwise_mae(pred, target))
    print("Pixelwise RMSE:", pixelwise_rmse(pred, target))
    print("Spatial Count:", spatial_count(pred))
