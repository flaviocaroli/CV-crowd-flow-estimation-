import os
from .resnet50 import ResNet50Backbone
from .unet import UNet4
from .vgg19bn import VGG19BNBackbone
import torch

def get_model(model_name, pretrained=True, freeze_encoder=False, cpt=None, device=None):
    """
    Returns (model, trainable_params) for 'resnet50' or 'vgg19_bn'.
    Raises on unsupported names.
    """
    if model_name == "resnet50":
        model = ResNet50Backbone()
        enc_layers = ["inc", "down1", "down2", "down3", "down4"]
        dec_layers = ["up1", "up2", "up3", "up4", "outc"]
    elif model_name == "vgg19_bn":
        model = VGG19BNBackbone()
        enc_layers = ["enc1", "enc2", "enc3", "enc4", "enc5"]
        dec_layers = ["up1", "up2", "up3", "up4", "outc"]
    elif model_name == "unet":
        model = UNet4()
        enc_layers = ["inc", "down1", "down2", "down3", "down4"]
        dec_layers = ["up1", "up2", "up3", "outc"]
    else:
        raise ValueError(f"Unsupported backbone '{model_name}'")

    if cpt is not None:
        # Load checkpoint
        if os.path.isfile(cpt):
            print(f"Loading checkpoint '{cpt}'")
            checkpoint = torch.load(cpt, map_location=device)
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{cpt}'")

    enc_params = []
    for layer in enc_layers:
        enc_params += list(getattr(model, layer).parameters())
    dec_params = []
    for layer in dec_layers:
        dec_params += list(getattr(model, layer).parameters())

    if freeze_encoder:
        for p in enc_params:
            p.requires_grad = False
        trainable = dec_params
    else:
        trainable = enc_params + dec_params

    model.to(device)
    return model, trainable

__all__ = ["get_model"]