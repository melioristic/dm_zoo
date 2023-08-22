import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision

from .vae import ConvVAE

class VAE(pl.LightningModule):
    def __init__(self,
        inp_shape,
        data_type = "input",
        train_dataset = None,
        valid_dataset = None,
        channel_mult = [1,2], 
        dim = None,
        batch_size = 128,
        lr = 1e-3,
        lr_scheduler_name="ReduceLROnPlateau",
        num_workers = 1,
        beta = 1.0
        ):
        super().__init__()

        self.beta = beta
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        if train_dataset!=None:
            len_train = self.train_dataset.stop-self.train_dataset.start
            len_val = self.valid_dataset.stop-self.valid_dataset.start

            self.train_kld_weight = batch_size/len_train
            self.val_kld_weight = batch_size/len_val
        else:
            len_train = None
            len_val = None
            self.train_kld_weight = None
            self.val_kld_weight = None



        self.lr = lr
        self.batch_size = batch_size
        self.data_type = data_type
        self.model = ConvVAE(
            inp_shape=inp_shape,
            channel_mult=channel_mult,
            dim=dim
        )

    @torch.no_grad()
    def forward(self, batch, *args):
        if self.data_type == "input":
            x, _ = batch
        elif self.data_type == "output":
            _, x = batch
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if self.data_type == "input":
            x, _ = batch
        elif self.data_type == "output":
            _, x = batch
        r, x, mu, log_var = self.model(x)

        loss = self.model.loss_function(r, x, mu, log_var, self.beta, self.train_kld_weight)

        self.log("train_loss", loss["loss"], prog_bar=True, on_epoch=True)
        self.log("train_recon_loss", loss["recon_loss"], prog_bar=True, on_epoch=True)
        self.log("train_KLD_loss", loss["KLD_loss"] , prog_bar=True, on_epoch=True)

        return loss["loss"] 

    def validation_step(self, batch, batch_idx):
        # standard loss:
        if self.data_type == "input":
            x, _ = batch
        elif self.data_type == "output":
            _, x = batch
        
        r, x, mu, log_var = self.model(x)

        loss = self.model.loss_function(r, x, mu, log_var, self.beta, self.val_kld_weight)

        self.log("val_loss", loss["loss"], prog_bar=True, on_epoch=True)
        self.log("val_recon_loss", loss["recon_loss"], prog_bar=True, on_epoch=True)
        self.log("val_KLD_loss", loss["KLD_loss"] , prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            n_images = 5
            grid = torchvision.utils.make_grid(torch.concat([r[:n_images], x[:n_images]], dim=0), nrow=n_images) # plot the first n_images images.
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

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
        scheduler = self._get_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def _get_scheduler(self, optimizer):
        # for experimental purposes only. 
        # All epoch related things are in respect to the "1x longer" epoch length.
        return getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(optimizer=optimizer)
        
    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None