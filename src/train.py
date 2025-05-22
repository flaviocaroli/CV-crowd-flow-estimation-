import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import wandb
import os

from src.data_loader import ShanghaiTechDataModule
from src.train_lightning import LitDensityEstimator
from src.utils import get_device, compute_receptive_field
from src.config_utils import load_experiment_configs

def main() -> None:
    parser = argparse.ArgumentParser(description="Train experiments from a YAML configuration file.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./train_config.yaml",
        help=(
            "Path to the training configuration YAML file."
            " Pointing to a different YAML is all you need to run new experiments."
        ),
    )
    args = parser.parse_args()

    wandb_project = "density_estimation_custom_head_init"
    pl.seed_everything(42)
    device = get_device()

    configs = load_experiment_configs(args.config_path)
    print(f"Loaded {len(configs)} configurations from {args.config_path}")
    for cfg in configs:
        name = cfg["name"]
        try:
            data_module = ShanghaiTechDataModule(
                data_folder=cfg.get("data_folder", "./data/ShanghaiTech"),
                part=cfg.get("dataset_part", "part_A"),
                validation_split=cfg.get("validation_split", 0.1),
                sigma=cfg.get("sigma", 5),
                return_count=cfg.get("return_count", False),
                batch_size=cfg.get("batch_size", 8),
                num_workers=cfg.get("num_workers", 4),
                device=device,
                target_input_size=(
                    cfg.get("target_input_width", 384),
                    cfg.get("target_input_height", 384),
                ),
                target_density_map_size=(
                    cfg.get("target_density_map_width", cfg.get("target_input_width", 384)),
                    cfg.get("target_density_map_height", cfg.get("target_input_height", 384)),
                ),
            )
            data_module.setup()
            checkpoint_dir = os.path.join("./models/checkpoints", name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            wandb_logger = WandbLogger(project=wandb_project, name=name)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=name + "_{epoch:02d}",
                save_top_k=1,
                monitor="val/mse",
                mode="min",
            )

            lr_monitor = LearningRateMonitor(logging_interval="step")

            early_stop_callback = EarlyStopping(
                monitor="val/mse",
                patience=10,
                mode="min",
                verbose=True,
                min_delta=1e-5,
            )

            model = LitDensityEstimator(
                model_name=cfg.get("model_name", "unet"),
                lr=cfg.get("learning_rate", 5e-4),
                pretrained=cfg.get("pretrained", True),
                freeze_encoder=cfg.get("freeze_encoder", False),
                device=device,
                **cfg.get("model_kwargs", {}),
            )

            wandb_logger.experiment.config.update({"model_architecture": str(model)})
            wandb_logger.experiment.config.update({"receptive_field": compute_receptive_field(model)})

            trainer = Trainer(
                max_epochs=cfg.get("max_epochs", 200),
                log_every_n_steps=10,
                default_root_dir="./outputs",
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                accelerator=str(device).lower(),
                devices=1,
            )

            trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
            trainer.save_checkpoint(os.path.join(checkpoint_dir, name + ".ckpt"))
            trainer.test(model, dataloaders=data_module.test_dataloader())
            wandb.finish()
        except Exception as e:
            print(f"Error in experiment {name}: {e}")
            wandb.log({"fatal_error": str(e)})
            wandb.finish(quiet=True)
            continue

    wandb.finish(quiet=True)

if __name__ == "__main__":
    main()