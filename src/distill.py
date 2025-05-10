import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import ShanghaiTechDataset
from src.finetune_metrics import count_mae_mse
from src.student_models import SmallRFNet, BigRFNet, SmallRFNetGAP, BigRFNetGAP
from models.resnet50_backbone import ResNet50Backbone
from models.vgg_backbone import VGG19Backbone

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
lr = 1e-4
epochs = 50
alpha = 0.5  # weight for distillation loss

# Load dataset
train_ds = ShanghaiTechDataset(split='train')
val_ds   = ShanghaiTechDataset(split='val')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size)

# Load teachers
teacher1 = ResNet50Backbone(pretrained=False, num_classes=1).to(device)
teacher2 = VGG19Backbone(pretrained=False, num_classes=1).to(device)
# load finetuned weights
teacher1.load_state_dict(torch.load('models/resnet50_finetuned.pth'))
teacher2.load_state_dict(torch.load('models/vgg19_finetuned.pth'))
teacher1.eval(); teacher2.eval()

# Initialize students
students = {
    'small': SmallRFNet().to(device),
    'big': BigRFNet().to(device),
    'small_gap': SmallRFNetGAP().to(device),
    'big_gap': BigRFNetGAP().to(device)
}
optimizers = {k: optim.Adam(m.parameters(), lr=lr) for k, m in students.items()}
criterion_gt = nn.MSELoss()
criterion_kd = nn.MSELoss()

# Distillation training loop
def train_student(name, model):
    optimizer = optimizers[name]
    best_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, counts in train_loader:
            imgs, counts = imgs.to(device), counts.to(device)
            # teacher predictions (soft targets)
            with torch.no_grad():
                t1 = teacher1(imgs)
                t2 = teacher2(imgs)
                soft_targets = (t1 + t2) / 2
            # student
            preds = model(imgs)
            # losses
            loss_gt = criterion_gt(preds, counts)
            loss_kd = criterion_kd(preds, soft_targets)
            loss = alpha * loss_kd + (1 - alpha) * loss_gt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        # validation
        model.eval()
        mae, mse = count_mae_mse(model, val_loader, device)
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f'models/{name}_student.pth')
        print(f"[{name}] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_ds):.4f} Val MAE: {mae:.2f}")

if __name__ == '__main__':
    for name, student in students.items():
        print(f"Training student: {name}")
        train_student(name, student)