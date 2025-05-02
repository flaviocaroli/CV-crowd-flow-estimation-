import torch
from torch import nn
from torch.utils.data import DataLoader
from models.resnet50_backbone import ResNet50Backbone
import matplotlib.pyplot as plt


def train_model(train_dataset, val_dataset, epochs=10, lr=1e-4, batch_size=16, save_path='models/resnet50_finetuned.pth'):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")

    model = ResNet50Backbone().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_mae, val_mae = [], []
    train_mse, val_mse = [], []
    train_rmse, val_rmse = [], []

    for epoch in range(epochs):
        model.train()
        t_mae = t_mse = 0
        for img, count_map in train_loader:
            img = img.to(device)
            count = count_map.sum(dim=(1,2)).unsqueeze(1).float().to(device)
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

        n_train = len(train_dataset)
        n_val = len(val_dataset)
        train_mae.append(t_mae / n_train)
        train_mse.append(t_mse / n_train)
        train_rmse.append((t_mse / n_train)**0.5)
        val_mae.append(v_mae / n_val)
        val_mse.append(v_mse / n_val)
        val_rmse.append((v_mse / n_val)**0.5)

        print(f"Epoch {epoch+1}/{epochs} - Train MAE: {train_mae[-1]:.2f} - Val MAE: {val_mae[-1]:.2f}")

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