import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .DenoisingDiffusionProcess import *


class PixelDiffusion(pl.LightningModule):
    def __init__(
        self,
        generated_channels,
        train_dataset=None,
        valid_dataset=None,
        num_timesteps=1000,
        batch_size=1,
        lr=1e-3,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size = batch_size

        self.model = DenoisingDiffusionProcess(
            num_timesteps=num_timesteps,
            generated_channels=generated_channels,
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0, 1).mul_(2)).sub_(1)

    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)

    def training_step(self, batch, batch_idx):
        images = batch
        loss = self.model.p_loss(self.input_T(images))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        loss = self.model.p_loss(self.input_T(images))

        self.log("val_loss", loss)

        return loss

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
        else:
            return None

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            return None
        
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            return None

    def configure_optimizers(self):
        # Cosine Annealing LR Scheduler

        optimizer = torch.optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters(),
                )
            ),
            lr=self.lr,
        )
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
        """

        return {"optimizer": optimizer}


class PixelDiffusionConditional(PixelDiffusion):
    def __init__(
        self,
        generated_channels,
        condition_channels,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        batch_size=1,
        lr=1e-3,
        num_diffusion_steps_prediction=200,
        cylindrical_padding=False
    ):
        pl.LightningModule.__init__(self)
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_diffusion_steps_prediction = num_diffusion_steps_prediction
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=generated_channels,
            condition_channels=condition_channels,
            cylindrical_padding=cylindrical_padding
        )

    @torch.no_grad()
    def forward(self, batch, *args, **kwargs):
        input, _, _ = batch
        return self.output_T(self.model(self.input_T(input), *args, **kwargs))

    def training_step(self, batch, batch_idx):
        input, output, _ = batch
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input, output, _ = batch
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        input, output, _ = batch
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        input, _, _ = batch

        # set up DDIM sampler: 
        sampler = DDIM_Sampler(self.num_diffusion_steps_prediction, self.model.num_timesteps)
        return self.output_T(self.model(self.input_T(input), sampler=sampler))
        
    
    def config(self):
        cfg = {
            "model_name": "PixelDiffusionConditional",
            "model_hparam": {
                "generated_channels": self.generated_channels,
                "condition_channels": self.condition_channels,
                "batch_size": self.batch_size,
                "lr": self.lr,
            },
        }
        return cfg
