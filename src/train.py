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
import torch

from src.data_loader import ShanghaiTechDataModule
from src.train_lightning import LitDensityEstimator
from src.utils import compute_receptive_field
import warnings
warnings.filterwarnings("ignore", message="Clipping input data to the valid range for imshow")

def main() -> None:
    wandb_project = "density-estimation"
    pl.seed_everything(42)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Base configuration
    base_config = {
        "model_name": "unet",
        "data_folder": "./data/ShanghaiTech",
        "dataset_part": "part_A",
        "num_workers": 2,  # Reduced to avoid conflicts
        "sigma": 5.0,
        "pretrained": True,
        "freeze_encoder": False,
        "max_epochs": 200,
        "target_input_width": 384,
        "target_input_height": 384,
        "target_density_map_width": 384,
        "target_density_map_height": 384,
        "batch_size": 8,
        "learning_rate": 0.00005,
        "validation_split": 0.1,
        "return_count": False,
        "wandb_project": wandb_project,
        "augment": False,
        "augment_factor": 8,
    }
    
    # Generate all configurations
    configs = []
    
    # Model combinations
    for model in ["vgg", "resnet"]:
        for part in ["part_A", "part_B"]:
            for depth in range(2, 5):
                    config = base_config.copy()
                    config["name"] = f"experiment_{model}_{part}_d{depth}"
                    config["model_name"] = model
                    config["dataset_part"] = part
                    config["model_kwargs"] = {
                        "base_channels": 32,
                        "depth": depth,
                        "stride_l1": 1,
                        "stride_l2": 1,
                        "dilation_l1": 1,
                        "dilation_l2": 1,
                    }
                    configs.append(config)
    
    # # Parameter sweep
    # for part in ["part_A", "part_B"]:
    #     for depth in range(2, 4):
    #         for dilation in range(1, 4):
    #             config = base_config.copy()
    #             config["name"] = f"experiment_{part}_d{depth}_dil{dilation}"
    #             config["dataset_part"] = part
    #             config["model_kwargs"] = {
    #                 "base_channels": 32,
    #                 "depth": depth,
    #                 "dilation_l1": dilation,
    #                 "dilation_l2": dilation,
    #                 "depth_dilation": 2,
    #             }
    #             configs.append(config)

    print(f"Total experiments: {len(configs)}")
    
    # Run experiments, alternating between GPUs
    for i, cfg in enumerate(configs):
        gpu_id = i % num_gpus if num_gpus > 0 else 0
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        name = cfg["name"]
        print(f"[GPU {gpu_id}] Running experiment {i+1}/{len(configs)}: {name}")
        
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            data_module = ShanghaiTechDataModule(
                data_folder=cfg.get("data_folder", "./data/ShanghaiTech"),
                part=cfg.get("dataset_part", "part_A"),
                validation_split=cfg.get("validation_split", 0.1),
                sigma=cfg.get("sigma", 5),
                return_count=cfg.get("return_count", False),
                batch_size=cfg.get("batch_size", 8),
                num_workers=cfg.get("num_workers", 2),
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

            wandb_logger = WandbLogger(project=wandb_project, name=f"{name}_gpu{gpu_id}")
            
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
                patience=20,
                mode="min",
                verbose=True,
                min_delta=1e-5,
            )

            freeze_encoder = cfg.pop("freeze_encoder", False)

            model = LitDensityEstimator(
                model_name=cfg.get("model_name", "unet"),
                lr=cfg.get("learning_rate", 5e-3),
                pretrained=cfg.get("pretrained", True),
                freeze_encoder=freeze_encoder,
                device=device,
                **cfg.get("model_kwargs", {}),
            )

            wandb_logger.experiment.config.update({
                "model_architecture": str(model),
                "receptive_field": compute_receptive_field(model),
                "gpu_id": gpu_id
            })

            trainer = Trainer(
                max_epochs=cfg.get("max_epochs", 200),
                log_every_n_steps=25,
                default_root_dir="./outputs",
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                accelerator="auto",
            )

            trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
            trainer.save_checkpoint(os.path.join(checkpoint_dir, name + ".ckpt"))
            trainer.test(model, dataloaders=data_module.test_dataloader())
            
            wandb.finish()
            print(f"[GPU {gpu_id}] Completed: {name}")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in {name}: {e}")
            wandb.finish(quiet=True)
            continue

if __name__ == "__main__":
    main()