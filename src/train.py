import os
import torch
from torch import nn
import matplotlib.pyplot as plt

# import your updated UNet‐style backbone
from tqdm import tqdm

from src.utils import get_device, get_model



def train_model(
    data_module,
    model_name="resnet50",
    epochs=10,
    lr=1e-4,
    save_path='../models/{{model_name}}_finetuned.pth',
    pretrained=True,
    device=None,
):
    
    if device is None:
        device = get_device()

    # Build model & optimizer
    model, trainable = get_model(
        model_name, pretrained=pretrained, freeze_encoder=False
    )
    model.to(device)
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()

    train_pixel_mae = []
    val_pixel_mae   = []
    train_pixel_mse = []
    val_pixel_mse   = []

    for epoch in range(1, epochs+1):
        model.train()
        running_mse = 0.0
        running_mae = 0.0
        count = 0

        for img, gt_density in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            img        = img.to(device)         # [B,3,H,W]
            gt_density = gt_density.to(device)  # [B,1,H,W]

            pred_density = model(img)           # [B,1,H,W]
            assert pred_density.shape == gt_density.shape, \
                f"Shape mismatch: {pred_density.shape} vs {gt_density.shape}"
            mse_loss = criterion(pred_density, gt_density)

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            # accumulate pixelwise metrics
            running_mse += mse_loss.item() * img.size(0)
            running_mae += torch.abs(pred_density - gt_density).sum().item()
            count += img.numel()  # total number of pixels processed

        # average over total pixels
        train_mse = running_mse / len(train_loader.dataset)
        train_mae = running_mae / count
        train_pixel_mse.append(train_mse)
        train_pixel_mae.append(train_mae)

        # Validation
        model.eval()
        val_mse_accum = 0.0
        val_mae_accum = 0.0
        val_pixels = 0
        with torch.no_grad():
            for img, gt_density in val_loader:
                img        = img.to(device)
                gt_density = gt_density.to(device)
                pred_density = model(img)

                val_mse_accum += criterion(pred_density, gt_density).item() * img.size(0)
                val_mae_accum += torch.abs(pred_density - gt_density).sum().item()
                val_pixels += img.numel()

        val_mse = val_mse_accum / len(val_loader.dataset)
        val_mae = val_mae_accum / val_pixels
        val_pixel_mse.append(val_mse)
        val_pixel_mae.append(val_mae)

        print(f"[{model_name}] Epoch {epoch}/{epochs}  "
              f"Train MSE: {train_mse:.6f}, MAE: {train_mae:.6f}  "
              f"Val   MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")

    # Save model weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path.format(model_name=model_name))
    print(f"Saved weights to {save_path.format(model_name=model_name)}")

    # Plot pixelwise metrics
    plt.figure(figsize=(8,4))
    plt.plot(train_pixel_mse, label='Train MSE')
    plt.plot(val_pixel_mse,   label='Val MSE')
    #plt.plot(train_pixel_mae, label='Train MAE')
    #plt.plot(val_pixel_mae,   label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Pixelwise Error')
    plt.title(f'{model_name} Pixelwise MSE & MAE')
    plt.legend()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{model_name}_pixelwise_errors.png')
    plt.show()

    return model