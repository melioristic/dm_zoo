import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *
from .backbones.unet_palette import UNet_Palette


class DenoisingDiffusionConditionalProcess(nn.Module):
    def __init__(
        self,
        config,
        generated_channels,
        conditioning_channels,
        loss_fn,
        sampler
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = conditioning_channels
        self.num_timesteps = config.num_diffusion_steps
        self.loss_fn = loss_fn

        # Forward Process
        self.forward_process = GaussianForwardProcess(
            num_timesteps=self.num_timesteps,
            schedule=config.noise_schedule,
        )

        # Neural Network Backbone
        if config.unet_type == "UnetConvNextBlock":
            self.model = UnetConvNextBlock(
                dim=config.num_channels_base,
                dim_mults=config.dims_mults,
                channels=self.generated_channels + self.condition_channels,
                out_dim=self.generated_channels,
                with_time_emb=True,
                use_cyclical_padding=config.use_cyclical_padding
            )
        elif config.unet_type == "UNet_Palette":
            self.model = UNet_Palette(
                in_channel=self.generated_channels+self.condition_channels,
                inner_channel=config.num_channels_base, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                out_channel=self.generated_channels,
                res_blocks=2,  # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                attn_res = [16,], # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                dropout=0.2, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                channel_mults=config.dims_mults, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                use_checkpoint=False,
                use_fp16=False,
                num_head_channels=32, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
            )
        else:
            raise NotImplementedError(f"Invalid type of UNet backbone: {config.unet_type}")
        # defaults to a DDPM sampler if None is provided
        
        self.sampler = (
            DDPM_Sampler(num_timesteps=self.num_timesteps)
            if sampler is None
            else sampler
        )

    @torch.no_grad()
    def forward(self, condition, sampler=None, verbose=False):
        """
        forward() function triggers a complete inference cycle

        A custom sampler can be provided as an argument!
        """

        # read dimensions
        b, c, h, w = condition.shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        x_t = torch.randn(
            [b, self.generated_channels, h, w],
            device=device,
        )

        for i in (
            tqdm(
                it,
                desc="diffusion sampling",
                total=num_timesteps,
            )
            if verbose
            else it
        ):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_input = torch.cat([x_t, condition], 1).to(device)
            z_t = self.model(model_input, t)  # prediction of noise
            x_t = self.sampler(x_t, t, z_t)  # prediction of next state

        return x_t

    def p_loss(self, output, condition):
        """
        Assumes output and input are in [-1,+1] range
        """
        b, c, h, w = output.shape
        device = output.device

        # loss for training

        # input is the optional condition
        t = torch.randint(
            0,
            self.forward_process.num_timesteps,
            (b,),
            device=device,
        ).long()
        output_noisy, noise = self.forward_process(output, t, return_noise=True)

        # reverse pass
        model_input = torch.cat([output_noisy, condition], 1).to(device)
        noise_hat = self.model(model_input, t)

        # apply loss
        return self.loss_fn(noise, noise_hat)