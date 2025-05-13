import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# import your updated UNet‐style backbone
from models.resnet50_backbone import ResNet50DensityBackbone
from tqdm import tqdm
from models.vgg_backbone      import VGG19BNBackbone  
from models.unet_backbone import UNetDensityBackbone  

def get_backbone(model_name, pretrained=True, freeze_encoder=False):
    """
    Instantiate a U-Net–style density backbone, optionally freeze encoder,
    and return (model, trainable_params).
    """
    if model_name == "resnet50":
        model = ResNet50DensityBackbone(pretrained=pretrained)
        # Encoder stages: enc1, enc2, enc3, enc4, enc5
        enc_params = (
            list(model.enc1.parameters()) +
            list(model.enc2.parameters()) +
            list(model.enc3.parameters()) +
            list(model.enc4.parameters()) +
            list(model.enc5.parameters())
        )
        # Decoder blocks: dec5, dec4, dec3, dec2, + final conv
        dec_params = (
            list(model.dec5.parameters()) +
            list(model.dec4.parameters()) +
            list(model.dec3.parameters()) +
            list(model.dec2.parameters()) +
            list(model.final.parameters())
        )
    elif model_name == "unet":
        model = UNetDensityBackbone(pretrained=pretrained)
        # Separate encoder and decoder parts
        enc_params = (
            list(model.inc.parameters()) +
            list(model.down1.parameters()) +
            list(model.down2.parameters()) +
            list(model.down3.parameters()) +
            list(model.down4.parameters())
        )
        dec_params = (
            list(model.up1.parameters()) +
            list(model.up2.parameters()) +
            list(model.up3.parameters()) +
            list(model.up4.parameters()) +
            list(model.outc.parameters())
        )

    elif model_name == "vgg19_bn":
        model = VGG19BNBackbone(pretrained=pretrained)
        enc_params = list(model.vgg.features.parameters())
        dec_params = list(model.decoder.parameters())

    else:
        raise ValueError(f"Unknown model_name {model_name}")

    if freeze_encoder:
        for p in enc_params:
            p.requires_grad = False
        trainable = dec_params
    else:
        trainable = enc_params + dec_params

    return model, trainable


def train_model(
    data_module,
    model_name="resnet50",
    epochs=10,
    lr=1e-4,
    save_path=f'../models/{{model_name}}_finetuned.pth',
    pretrained=True,
):
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

    # Build model & optimizer
    model, trainable = get_backbone(
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
    plt.plot(train_pixel_mae, label='Train MAE')
    plt.plot(val_pixel_mae,   label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Pixelwise Error')
    plt.title(f'{model_name} Pixelwise MSE & MAE')
    plt.legend()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{model_name}_pixelwise_errors.png')
    plt.show()
