import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

# Added imports
import argparse
from src.config_utils import get_model_config
from src.data_loader import ShanghaiTechDataModule

import wandb
from src.metrics import (
    count_mae,
    count_mse,
    count_rmse,
    pixelwise_mae,
    pixelwise_rmse,
)
from src.models import get_model
from src.utils import plot_dec_steps_batch

class MSEWithNegPenaltyLoss(nn.Module):
    def __init__(self, neg_penalty_weight=10.0):
        super().__init__()
        self.neg_penalty_weight = neg_penalty_weight

    def forward(self, preds, targets):
        # MSE sum loss
        mse_loss = torch.sum((preds - targets) ** 2)
        
        # Negative predictions penalty (sum of negative parts)
        neg_penalty = torch.sum(torch.relu(-preds))  # sum of abs of negative values
        
        # Total loss = MSE + weighted negative penalty
        total_loss = mse_loss + self.neg_penalty_weight * neg_penalty
        return total_loss

class LitDensityEstimator(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet50",
        lr: float = 1e-4,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        device: torch.device | None = None,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = get_model(
            model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            **model_kwargs
        )
        self.criterion = MSEWithNegPenaltyLoss(neg_penalty_weight=10.0)
        self.model.to(device)

    def forward(self, x, return_intermediates: bool = False):
        return self.model(x, return_intermediates=return_intermediates)

    def training_step(self, batch, batch_idx):
        img, gt = batch
        pred = self(img)
        loss = self.criterion(pred, gt)
        mae = pixelwise_mae(pred, gt)
        rmse = pixelwise_rmse(pred, gt)

        self.log('train/mse', loss, on_epoch=True, prog_bar=True)
        self.log('train/mae', mae, on_epoch=True)
        self.log('train/rmse', rmse, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, name="val"):
        img, gt = batch
        preds = self(img, return_intermediates=True)
        pred = preds[-1]  # Get the final output
        loss = self.criterion(pred, gt)
        mae = pixelwise_mae(pred, gt)
        rmse = pixelwise_rmse(pred, gt)

        # count metrics
        count_mae_val = count_mae(pred, gt)
        count_mse_val = count_mse(pred, gt)
        count_rmse_val = count_rmse(pred, gt)

        self.log(f'{name}/mse', loss, on_epoch=True, prog_bar=True)
        self.log(f'{name}/mae', mae, on_epoch=True)
        self.log(f'{name}/rmse', rmse, on_epoch=True)

        # Log count metrics
        self.log(f'{name}/count_mae', count_mae_val, on_epoch=True)
        self.log(f'{name}/count_mse', count_mse_val, on_epoch=True)
        self.log(f'{name}/count_rmse', count_rmse_val, on_epoch=True)

        # Log images to W&B for the first batch only
        if batch_idx == 0 and isinstance(self.logger, pl_loggers.WandbLogger) and self.current_epoch % 10 == 0:
            fig = plot_dec_steps_batch(img, gt, preds)
            # Log figure
            self.logger.experiment.log({
            f"{name}_batch_images_epoch_{self.current_epoch}": wandb.Image(fig)
            })
            plt.close(fig)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(self.parameters(), lr=self.lr)
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
        # Return compatible config for PyTorch Lightning
        return OptimizerLRSchedulerConfig({ 
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a density estimation model.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment_A",
        help="Name of the experiment to run, as defined in the config file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./train_config.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    config = get_model_config(yaml_path=args.config_path, experiment_name=args.experiment_name)

    # Extract data-related parameters
    data_folder = config.get('data_folder', './data/ShanghaiTech')
    part = config.get('dataset_part', 'part_A')
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    target_input_width = config.get('target_input_width', 384)
    target_input_height = config.get('target_input_height', 384)
    target_input_size_datamodule = (target_input_width, target_input_height) 

    sigma = config.get('sigma', 5.0)
    augmentation_cfg = config.get('augmentation')

    # Extract model-related parameters
    model_name = config.get('model_name', 'unet') # default to 'unet' as per previous contexts
    lr = config.get('learning_rate', 1e-4)
    pretrained = config.get('pretrained', True)
    freeze_encoder = config.get('freeze_encoder', False)
    model_specific_kwargs = config.get('model_kwargs', {})

    data_module = ShanghaiTechDataModule(
        data_folder=data_folder,
        part=part,
        batch_size=batch_size,
        num_workers=num_workers,
        target_input_size=target_input_size_datamodule, # (W,H)
        target_density_map_size=target_input_size_datamodule, # Assuming (W,H) and same as input
        sigma=sigma,
    )

    # Instantiate LitDensityEstimator
    model = LitDensityEstimator(
        model_name=model_name,
        lr=lr,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        **model_specific_kwargs
    )

    # Instantiate Trainer
    # For simplicity, logger is omitted, PyTorch Lightning will use a default TensorBoardLogger or CSVLogger.
    # wandb_logger = pl_loggers.WandbLogger(project="density_estimation", name=args.experiment_name)
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 10),
        accelerator="auto",
        devices="auto",
        # logger=wandb_logger # Uncomment to use WandB
    )

    # Run Training and Testing
    print(f"Starting training for experiment: {args.experiment_name} with config:")
    import yaml
    print(yaml.dump(config))

    trainer.fit(model, datamodule=data_module)
    print("Training finished.")

    print("Starting testing...")
    trainer.test(model, datamodule=data_module)
    print("Testing finished.")