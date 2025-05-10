import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# import your three backbone classes
from models.resnet50_backbone import ResNet50Backbone
from models.vgg_backbone      import VGG19BNBackbone
from models.alexnet_backbone  import AlexNetBackbone

def get_backbone(model_name, pretrained=True, freeze_backbone=False):
    """
    Instantiate one of your three backbones, freeze its pretrained layers if requested,
    and return it ready for fine‑tuning the head.
    """
    if model_name == "resnet50":
        model = ResNet50Backbone(pretrained=pretrained)
        # in ResNet50Backbone we replaced resnet.fc already
        backbone_params = model.resnet.layer1, model.resnet.layer2, model.resnet.layer3, model.resnet.layer4
        head_params = model.resnet.fc
    elif model_name == "vgg19_bn":
        model = VGG19BNBackbone(pretrained=pretrained)
        backbone_params = model.vgg.features
        head_params = model.vgg.classifier
    else:
        raise ValueError(f"Unknown model_name {model_name}")

    if freeze_backbone:
        # freeze all parameters in the backbone
        for p in backbone_params.parameters():
            p.requires_grad = False

    return model, head_params

def train_model(
    data_module,
    model_name="resnet50",
    epochs=10,
    lr=1e-4,
    batch_size=16,
    save_path=f'../models/{{model_name}}_finetuned.pth'
):
    # device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # get backbone + identify head parameters for optimizer
    model, head = get_backbone(model_name, pretrained=True, freeze_backbone=False)
    model = model.to(device)

    # only train head parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    train_loader = data_module.train_dataloader(batch_size=batch_size)
    val_loader   = data_module.val_dataloader(batch_size=batch_size)

    # metrics storage
    train_mae, val_mae = [], []
    train_mse, val_mse = [], []
    train_rmse, val_rmse = [], []

    for epoch in range(1, epochs+1):
        # ——— TRAIN ———
        model.train()
        t_mae = t_mse = 0.0
        for img, count_map in train_loader:
            img = img.to(device)
            # ground‑truth count as sum over density map
            count = count_map.sum(dim=(1,2)).unsqueeze(1).to(device).float()

            out = model(img)
            loss = criterion(out, count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_mae += torch.abs(out - count).sum().item()
            t_mse += ((out - count)**2).sum().item()

        # ——— VALIDATION ———
        model.eval()
        v_mae = v_mse = 0.0
        with torch.no_grad():
            for img, count_map in val_loader:
                img = img.to(device)
                count = count_map.sum(dim=(1,2)).unsqueeze(1).to(device).float()
                out = model(img)
                v_mae += torch.abs(out - count).sum().item()
                v_mse += ((out - count)**2).sum().item()

        # normalize by dataset size
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)

        train_mae.append(t_mae / n_train)
        train_mse.append(t_mse / n_train)
        train_rmse.append((t_mse / n_train)**0.5)

        val_mae.append(v_mae / n_val)
        val_mse.append(v_mse / n_val)
        val_rmse.append((v_mse / n_val)**0.5)

        print(f"[{model_name}] Epoch {epoch}/{epochs}  "
              f"Train MAE: {train_mae[-1]:.3f}  Val MAE: {val_mae[-1]:.3f}")

    # save model weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path.format(model_name=model_name))
    print(f"Saved weights to {save_path.format(model_name=model_name)}")

    # plot metrics
    plt.figure(figsize=(8,4))
    plt.plot(train_mae, label='Train MAE')
    plt.plot(val_mae, label='Val MAE')
    #plt.plot(train_mse, label='Train MSE')
    #plt.plot(val_mse, label='Val MSE')
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse,   label='Val RMSE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'{model_name} Metrics')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{model_name}_metrics.png')
    plt.show()
