import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import wandb
import os

from src.data_loader import ShanghaiTechDataModule
from src.train_lightning import LitDensityEstimator
from src.utils import get_device

def main():
    # Initialize Weights & Biases (W&B)
    wandb_project = "density_estimation"
    pl.seed_everything(42)
    device = get_device()

    # Prepare the data
    data_module = ShanghaiTechDataModule(
        data_folder="./data/ShanghaiTech",  # adjust as needed
        part="part_A",
        validation_split=0.1,
        sigma=5,
        return_count=False,
        batch_size=16,
        num_workers=4,
        input_size=(384, 384),
        density_map_size=(192, 192),
        device=device,
    )
    data_module.setup()

    # Experiment configurations
    configs = [
        {"depth": 4, "num_filters": 32},
        {"depth": 3, "num_filters": 64},
        {"depth": 3, "num_filters": 32},
        {"depth": 2, "num_filters": 64},
        {"depth": 2, "num_filters": 128},
    ]

    def make_name(cfg):
        return f"unet_depth{cfg['depth']}_nf{cfg['num_filters']}"

    for cfg in configs:
        name = make_name(cfg)
        checkpoint_dir = os.path.join("./models/checkpoints", name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Setup W&B logger
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=name,
            tags=["unet", f"depth_{cfg['depth']}", f"nf_{cfg['num_filters']}"],
        )

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=name + "_{epoch:02d}",
            save_top_k=1,
            monitor="val_mse",
            mode="min"
        )

        # LR monitor callback
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Early stopping callback
        early_stop_callback = EarlyStopping(
            monitor="val_mse",
            patience=10,            # stop if no improvement in 10 checks
            mode="min",
            verbose=True,
            min_delta=0.0001,      # minimum change to qualify as an improvement
        )

        # Initialize model
        model = LitDensityEstimator(
            model_name="unet",
            lr=1e-4,
            pretrained=True,
            freeze_encoder=False,
            depth=cfg["depth"],
            num_filters=cfg["num_filters"],
            device=device,
        )

        # Initialize Trainers
        trainer = Trainer(
            max_epochs=50,
            default_root_dir="./outputs",
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            accelerator=str(device).lower(),
            devices=1,
            log_every_n_steps=50,
        )

        # Fit model
        trainer.fit(model, datamodule=data_module)

        # Save final checkpoint
        trainer.save_checkpoint(os.path.join(checkpoint_dir, name + ".ckpt"))

        trainer.test(model, datamodule=data_module)

        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()