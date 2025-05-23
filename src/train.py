import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import wandb
import os

from src.data_loader import ShanghaiTechDataModule
from src.train_lightning import LitDensityEstimator
from src.utils import get_device, compute_receptive_field

def main():
    # Initialize Weights & Biases (W&B)
    wandb_project = "recover"
    pl.seed_everything(42)
    device = get_device()

    # Prepare the data
    data_module = ShanghaiTechDataModule(
        data_folder="./data/ShanghaiTech",  # adjust as needed
        part="part_A",
        validation_split=0.1,
        sigma=5,
        return_count=False,
        batch_size=8,
        num_workers=4,
        input_size=(384, 384),
        density_map_size=(192, 192),
        device=device,
    )
    data_module.setup()

    # Experiment configurations
    configs = [        

        # ResNet50
        #{"model":"resnet50","depth": 3, "split":"A"},
        #{"model":"resnet50","depth": 2, "split":"A"},
        #{"model":"resnet50","depth": 4, "split":"A"},
        ### VGG19
        #{"model":"vgg19_bn","depth": 4, "split":"A"},
        #{"model":"vgg19_bn","depth": 3, "split":"A"},
        #{"model":"vgg19_bn","depth": 2, "split":"A"},

                # ResNet50
        {"model":"resnet50","depth": 3, "split":"B"},
        {"model":"resnet50","depth": 2, "split":"B"},
        {"model":"resnet50","depth": 4, "split":"B"},

        ## VGG19
        {"model":"vgg19_bn","depth": 4, "split":"B"},
        {"model":"vgg19_bn","depth": 3, "split":"B"},
        {"model":"vgg19_bn","depth": 2, "split":"B"},
    ]

    def make_name(cfg):
        label = cfg.get('label')
        if label:
            return f"{label}_{cfg['model']}_depth{cfg['depth']}_split_{cfg['split']}"
        return f"{cfg['model']}_depth{cfg['depth']}_split_{cfg['split']}"

    for cfg in configs:
        name = make_name(cfg)
        try:
            checkpoint_dir = os.path.join("./models/checkpoints", name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Setup W&B logger
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=name,
                tags=[f"model_{cfg["model"]}", f"depth_{cfg['depth']}", f"split_{cfg['split']}"],
            )

            # Checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=name + "_{epoch:02d}",
                save_top_k=1,
                monitor="val/mse",
                mode="min"
            )

            # LR monitor callback
            lr_monitor = LearningRateMonitor(logging_interval="step")

            # Early stopping callback
            early_stop_callback = EarlyStopping(
                monitor="val/mse",
                patience=10,            # stop if no improvement in 10 checks
                mode="min",
                verbose=True,
                min_delta=0.00001,      # minimum change to qualify as an improvement
            )

            # Initialize model
            model = LitDensityEstimator(
                model_name=cfg["model"],
                lr=5e-4,
                device=device,
                **cfg,
            )

            # Log model architecture and receptive field to W&B
            wandb_logger.experiment.config.update({"model_architecture": str(model)})
            wandb_logger.experiment.config.update({"receptive_field": compute_receptive_field(model)})
            wandb_logger.experiment.config.update({"cfg": cfg})

            # Initialize Trainers
            trainer = Trainer(
                max_epochs=200,
                log_every_n_steps=10,
                default_root_dir="./outputs",
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                accelerator=str(device).lower(),
                devices=1,
            )

            # Fit model
            trainer.fit(model, datamodule=data_module)

            # Save final checkpoint
            trainer.save_checkpoint(os.path.join(checkpoint_dir, name + ".ckpt"))

            trainer.test(model, datamodule=data_module)

            # Finish W&B run
            wandb.finish()
        except Exception as e:
            print(f"Error in experiment {name}: {e}")
            wandb.log({"fatal_error": str(e)})
            wandb.finish(quiet=True)
            continue
    
    # Finish W&B run
    wandb.finish(quiet=True)

if __name__ == "__main__":
    main()