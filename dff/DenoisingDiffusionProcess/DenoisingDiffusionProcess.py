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
        args,
        generated_channels,
        conditioning_channels,
        loss_fn,
        sampler
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = conditioning_channels
        self.num_timesteps = args.num_diffusion_steps
        self.loss_fn = loss_fn

        # Forward Process
        self.forward_process = GaussianForwardProcess(
            num_timesteps=self.num_timesteps,
            schedule=args.noise_schedule,
        )

        # Neural Network Backbone
        if args.unet_type == "UnetConvNextBlock":
            self.model = UnetConvNextBlock(
                dim=args.num_channels_base,
                dim_mults=args.dims_mults,
                channels=self.generated_channels + self.condition_channels,
                out_dim=self.generated_channels,
                with_time_emb=True,
                use_cyclical_padding=args.use_cyclical_padding
            )
        elif args.unet_type == "UNet_Palette":
            self.model = UNet_Palette(
                in_channel=self.generated_channels+self.condition_channels,
                inner_channel=args.num_channels_base, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                out_channel=self.generated_channels,
                res_blocks=2,  # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                attn_res = [16,], # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                dropout=0.2, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                channel_mults=args.dims_mults, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
                use_checkpoint=False,
                use_fp16=False,
                num_head_channels=32, # from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/config/inpainting_celebahq.json
            )
        else:
            raise NotImplementedError("Invalid type of UNet backbone")
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DiffusionProcess")
        parser.add_argument("--unet_type", type=str, default="UnetConvNextBlock") 
        parser.add_argument("--noise_schedule", type=str, default="linear") 
        parser.add_argument("--use_cyclical_padding", type=bool, default=False)
        parser.add_argument("--num_diffusion_steps", type=int, default=1000)           
        parser.add_argument("--dims_mults", type=tuple, default=(1,2,4,8))
        parser.add_argument("--num_channels_base", type=int, default=64)
        return parent_parser