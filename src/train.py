from src.data_loader import ShanghaiTechDataset
from src.finetune_metrics import train_model
from torchvision import transforms

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = ShanghaiTechDataset(
        root="data/ShanghaiTech",
        part="part_A",
        split="train_data",
        transform=transform
    )

    val_dataset = ShanghaiTechDataset(
        root="data/ShanghaiTech",
        part="part_A",
        split="test_data",
        transform=transform
    )

    train_model(train_dataset, val_dataset, epochs=30, lr=1e-4, batch_size=8)