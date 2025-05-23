import os
from .resnet50 import ResNetUNet, ResnetSkip1
from .unet import UNet, UNetSkip1
from .vgg19 import VGGUNet, VGGSkip1
import torch


def get_model(
    model_name, freeze_encoder=False, cpt=None, device=None, **kwargs
):
    """
    Returns (model, trainable_params) for 'resnet50', 'vgg19_bn', or 'unet'.
    If freeze_encoder=True, only decoder parameters remain trainable.
    """
    # Determine depth if provided
    depth = kwargs.pop("depth", kwargs.get("model_depth", 4))
    in_channels = kwargs.pop("in_channels", 3)
    num_filters = kwargs.pop("num_filters", 32)
    skip_skip = kwargs.pop("skip_skip", False)

    # Instantiate backbone
    if model_name == "resnet":
        if skip_skip:
            model = ResnetSkip1(
                depth=depth,
                **kwargs,
            )
        else:
            model = ResNetUNet(
                depth=depth,
                **kwargs,
            )
    elif model_name == "vgg":
        if skip_skip:
            model = VGGSkip1(
                depth=depth,
                **kwargs,
            )
        else:
            model = VGGUNet(
                depth=depth,
                **kwargs,
            )
    elif model_name == "unet":
        if skip_skip:
            model = UNetSkip1(
                in_channels=in_channels,
                num_filters=num_filters,
                depth=depth,
                **kwargs,
            )
        else:
            model = UNet(
                in_channels=in_channels,
                num_filters=num_filters,
                depth=depth,
                **kwargs,
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
