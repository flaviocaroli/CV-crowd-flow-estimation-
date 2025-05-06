import torch
from torch import nn
from torch.utils.data import DataLoader
from models.resnet50_backbone import ResNet50Backbone
import matplotlib.pyplot as plt


def train_model(data_module, epochs=10, lr=1e-4, batch_size=16, save_path='../models/resnet50_finetuned.pth'):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")

    model = ResNet50Backbone().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    train_mae, val_mae = [], []
    train_mse, val_mse = [], []
    train_rmse, val_rmse = [], []

    for epoch in range(epochs):
        model.train()
        t_mae = t_mse = 0
        for batch in train_loader:
            img, count_map = batch
            img = img.to(device)
            count = count_map.sum(dim=(1, 2)).unsqueeze(1).float().to(device)
            output = model(img)
            loss = criterion(output, count)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_mae += torch.abs(output - count).sum().item()
            t_mse += ((output - count)**2).sum().item()

        model.eval()
        v_mae = v_mse = 0
        with torch.no_grad():
            for img, count_map in val_loader:
                img = img.to(device)
                count = count_map.sum(dim=(1,2)).unsqueeze(1).float().to(device)
                output = model(img)
                v_mae += torch.abs(output - count).sum().item()
                v_mse += ((output - count)**2).sum().item()

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        train_mae.append(t_mae / n_train)
        train_mse.append(t_mse / n_train)
        train_rmse.append((t_mse / n_train)**0.5)
        val_mae.append(v_mae / n_val)
        val_mse.append(v_mse / n_val)
        val_rmse.append((v_mse / n_val)**0.5)

        print(f"Epoch {epoch+1}/{epochs} - Train MAE: {train_mae[-1]:.2f} - Val MAE: {val_mae[-1]:.2f}")

    import os
    os.makedirs(os.path.dirname('outputs/metrics_plot.png'), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Saving model to {'outputs/metrics_plot.png'}")
    else:
        print(f"Results already exists at {'outputs/metrics_plot.png'}, overwriting.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Saving model to {save_path}")
    else:
        print(f"Model already exists at {save_path}, overwriting.")
    torch.save(model.state_dict(), save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_mae, label='Train MAE')
    plt.plot(val_mae, label='Val MAE')
    plt.plot(train_mse, label='Train MSE')
    plt.plot(val_mse, label='Val MSE')
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Val RMSE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Metrics')
    plt.savefig('outputs/metrics_plot.png')
    plt.show()