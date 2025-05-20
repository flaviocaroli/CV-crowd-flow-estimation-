import os
from .resnet50 import ResNet50Backbone
from .unet import UNet
from .vgg19bn import VGG19BNBackbone
import torch


def get_model(
    model_name, freeze_encoder=False, cpt=None, device=None, **kwargs
):
    """
    Returns (model, trainable_params) for 'resnet50', 'vgg19_bn', or 'unet'.
    If freeze_encoder=True, only decoder parameters remain trainable.
    """
    # Determine depth if provided
    depth = kwargs.get("depth", kwargs.get("model_depth", 4))

    # Instantiate backbone
    if model_name == "resnet50":
        model = ResNet50Backbone()
    elif model_name == "vgg19_bn":
        model = VGG19BNBackbone()
    elif model_name == "unet":
        model = UNet(
            in_channels=kwargs.get("in_channels", 3),
            base_channels=kwargs.get("base_channels", 32),
            depth=depth,
        )
    else:
        raise ValueError(f"Unsupported backbone '{model_name}'")

    # Load checkpoint if provided
    if cpt is not None:
        if os.path.isfile(cpt):
            print(f"Loading checkpoint '{cpt}'")
            checkpoint = torch.load(cpt, map_location=device)
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{cpt}'")

    # Optionally freeze encoder parameters
    if freeze_encoder:
        for name, param in model.named_parameters():
            # keep decoder params trainable (upX and outc)
            if not (name.startswith('up') or name.startswith('outc')):
                param.requires_grad = False

    # Move model to device
    model.to(device)

    return model

__all__ = ["get_model"]
