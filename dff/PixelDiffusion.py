import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import math
from .DenoisingDiffusionProcess import *




class PixelDiffusionConditional(pl.LightningModule):
    def __init__(
        self,
        args,
        generated_channels,
        conditioning_channels,
        loss_fn,
        sampler,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
    ):
        pl.LightningModule.__init__(self)
        
        self.lr_scheduler_name = args.lr_scheduler_name
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.num_workers=args.num_workers
        self.num_diffusion_steps_inference=args.num_diffusion_steps_inference

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset        

        self.model = DenoisingDiffusionConditionalProcess(
            args,
            generated_channels=generated_channels,
            conditioning_channels=conditioning_channels,
            loss_fn=loss_fn,
            sampler=sampler
        )

    def _get_scheduler(self, optimizer):
        # for experimental purposes only. All epoch related things are in respect to the "1x longer" epoch length.
        match self.lr_scheduler_name:
            case "Constant":
                return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1)
            case "ReduceLROnPlateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 30, factor=0.2, min_lr=1e-8)
            case "StepLR":
                return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.2)
            case "CosineAnnealingLR":
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=360, eta_min=1e-6)
            case "CosineAnnealingWarmRestarts":
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=120, T_mult=2, eta_min=1e-6)
            case "CosineAnnealingWarmupRestarts":
                return CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=240, cycle_mult=2, max_lr=self.lr, min_lr=1e-6, warmup_steps=5, gamma=0.5)
            case _:
                raise ValueError("Invalid argument passed to scheduler configuration.")

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
        
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None
        
    def val_dataloader(self):
        if self.valid_dataset is not None:
            # indices = np.random.choice(np.arange(len(self.valid_dataset)), size=64, replace=False)
            return DataLoader(
                self.valid_dataset,
                num_workers=self.num_workers,
                # sampler=SubsetRandomSampler(indices=indices),
                batch_size=self.batch_size,
                # shuffle=False,
            )
        else:
            return None
    
    @torch.no_grad()
    def forward(self, batch, *args, **kwargs):
        input, _ = batch
        return self.output_T(self.model(self.input_T(input), *args, **kwargs))

    def training_step(self, batch, batch_idx):
        input, output = batch
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log("train_loss", loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        input, output = batch
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        input, target = batch
        # set up DDIM sampler: 
        sampler = DDIM_Sampler(self.num_diffusion_steps_inference, self.model.num_timesteps)
        return self.output_T(self.model(self.input_T(input), sampler=sampler))

    def validation_step(self, batch, batch_idx):
        input, output = batch
        # standard loss:
        loss = self.model.p_loss(self.input_T(output), self.input_T(input))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        # full reconstruction loss:
        sampler = DDIM_Sampler(self.num_diffusion_steps_inference, self.model.num_timesteps)
        prediction = self.output_T(self.model(self.input_T(input), sampler=sampler))
        reconstruction_loss = F.mse_loss(prediction, output)
        self.log("val_loss_new", reconstruction_loss, prog_bar=True, on_epoch=True)
        return loss       
    
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
            "monitor": "val_loss_new"
        }
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0, 1).mul_(2)).sub_(1)

    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    """
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
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConditionalDiffusionModel")
        parser.add_argument("--lr_scheduler_name", type=str, default="Constant", 
                            choices= ["Constant", "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", 
                                      "CosineAnnealingWarmRestarts", "CosineAnnealingWarmupRestarts"]) 
        parser.add_argument("--num_diffusion_steps_inference", type=int, default=200) 
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=int, default=1E-3)     
        parser.add_argument("--num_workers", type=int, default=0)
        return parent_parser


from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr