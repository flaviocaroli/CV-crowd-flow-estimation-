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
from concurrent.futures import ProcessPoolExecutor
import time

from src.data_loader import ShanghaiTechDataModule
from src.train_lightning import LitDensityEstimator
from src.utils import compute_receptive_field
import warnings
import torch 

warnings.filterwarnings("ignore", message="Clipping input data to the valid range for imshow")

def run_single_experiment(config, gpu_id):
    """Run a single experiment on specified GPU"""
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Force device to cuda:0 since CUDA_VISIBLE_DEVICES makes the assigned GPU appear as cuda:0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    name = config["name"]
    print(f"[GPU {gpu_id}] Starting experiment: {name}")
    
    try:
        # Initialize wandb with unique name
        wandb_logger = WandbLogger(
            project=config["wandb_project"], 
            name=f"{name}_gpu{gpu_id}"
        )
        
        # Data module
        data_module = ShanghaiTechDataModule(
            data_folder=config.get("data_folder", "./data/ShanghaiTech"),
            part=config.get("dataset_part", "part_A"),
            validation_split=config.get("validation_split", 0.1),
            sigma=config.get("sigma", 5),
            return_count=config.get("return_count", False),
            batch_size=config.get("batch_size", 8),
            num_workers=config.get("num_workers", 4),
            device=device,
            target_input_size=(
                config.get("target_input_width", 384),
                config.get("target_input_height", 384),
            ),
            target_density_map_size=(
                config.get("target_density_map_width", config.get("target_input_width", 384)),
                config.get("target_density_map_height", config.get("target_input_height", 384)),
            ),
            augment=config.get("augment", True),
            augment_factor=config.get("augment_factor", 8),
        )
        data_module.setup()
        
        # Checkpoints
        checkpoint_dir = os.path.join("./models/checkpoints", name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
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
        
        # Model
        model = LitDensityEstimator(
            model_name=config.get("model_name", "unet"),
            lr=config.get("learning_rate", 5e-4),
            pretrained=config.get("pretrained", True),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
            **config.get("model_kwargs", {}),
        )
        
        wandb_logger.experiment.config.update({"model_architecture": str(model)})
        wandb_logger.experiment.config.update({"receptive_field": compute_receptive_field(model)})
        wandb_logger.experiment.config.update({"gpu_id": gpu_id})
        
        # Trainer
        trainer = Trainer(
            max_epochs=config.get("max_epochs", 200),
            log_every_n_steps=10,
            default_root_dir="./outputs",
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            accelerator="gpu",
            devices=1,  # Single GPU per process
        )
        
        # Train
        trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        trainer.save_checkpoint(os.path.join(checkpoint_dir, name + ".ckpt"))
        trainer.test(model, dataloaders=data_module.test_dataloader())
        
        print(f"[GPU {gpu_id}] Completed experiment: {name}")
        wandb.finish()
        return f"Success: {name}"
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in experiment {name}: {e}")
        if 'wandb_logger' in locals():
            wandb.log({"fatal_error": str(e)})
        wandb.finish(quiet=True)
        return f"Failed: {name} - {str(e)}"

def run_experiment_batch(configs, gpu_id):
    """Run a batch of experiments on a single GPU"""
    results = []
    for config in configs:
        result = run_single_experiment(config, gpu_id)
        results.append(result)
        time.sleep(2)  # Brief pause between experiments
    return results

def main() -> None:
    import torch
    
    # Check GPU availability
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Warning: Less than 2 GPUs available. Falling back to single GPU.")
        num_gpus = 1
    else:
        num_gpus = 2
        print(f"Using {num_gpus} GPUs")
    
    wandb_project = "density-estimation"
    pl.seed_everything(42)

    # Base configuration
    base_config = {
        "model_name": "unet",
        "data_folder": "./data/ShanghaiTech",
        "dataset_part": "part_A",
        "num_workers": 4,
        "sigma": 5.0,
        "pretrained": True,
        "freeze_encoder": False,
        "max_epochs": 150,
        "target_input_width": 224,
        "target_input_height": 224,
        "target_density_map_width": 224,
        "target_density_map_height": 224,
        "batch_size": 8,
        "learning_rate": 0.00005,
        "validation_split": 0.1,
        "return_count": False,
        "wandb_project": wandb_project,
        "augment": False,
        "augment_factor": 8,
    }
    
    # Generate all experiment configurations
    configs = []
    
    # Model and dataset combinations
    for model in ["vgg", "resnet"]:
        for part in ["part_A", "part_B"]:
            for depth in range(2, 4):
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
    
    # Parameter sweep experiments
    for part in ["part_A", "part_B"]:
        for depth in range(2, 4):
            for stride in range(1, 4):
                for dilation in range(1, 4):
                    config = base_config.copy()
                    config["name"] = f"experiment_{part}_d{depth}_s{stride}_dil{dilation}"
                    config["dataset_part"] = part
                    config["model_kwargs"] = {
                        "base_channels": 32,
                        "depth": depth,
                        "stride_l1": stride,
                        "stride_l2": stride,
                        "dilation_l1": dilation,
                        "dilation_l2": dilation,
                    }
                    configs.append(config)

    print(f"Total experiments: {len(configs)}")
    
    if num_gpus == 1:
        # Single GPU fallback
        for config in configs:
            run_single_experiment(config, 0)
    else:
        # Multi-GPU execution
        # Split experiments between GPUs (alternating assignment for load balancing)
        gpu_configs = [configs[i::num_gpus] for i in range(num_gpus)]
        
        print(f"GPU 0: {len(gpu_configs[0])} experiments")
        print(f"GPU 1: {len(gpu_configs[1])} experiments")
        
        # Run experiments in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [
                executor.submit(run_experiment_batch, gpu_configs[i], i) 
                for i in range(num_gpus)
            ]
            
            # Collect results
            all_results = []
            for i, future in enumerate(futures):
                try:
                    results = future.result()
                    print(f"\nGPU {i} completed {len(results)} experiments:")
                    for result in results:
                        print(f"  {result}")
                    all_results.extend(results)
                except Exception as e:
                    print(f"GPU {i} batch failed: {e}")
        
        print(f"\nTotal completed experiments: {len(all_results)}")

if __name__ == "__main__":
    main()