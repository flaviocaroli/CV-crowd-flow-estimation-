import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

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
        self.criterion = nn.MSELoss()
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
        if batch_idx == 0 and isinstance(self.logger, pl_loggers.WandbLogger):
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