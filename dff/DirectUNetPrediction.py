
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenoisingDiffusionProcess.forward import *
from .DenoisingDiffusionProcess.samplers import *
from .DenoisingDiffusionProcess.backbones.unet_convnext import *
from .DenoisingDiffusionProcess.backbones.unet_palette import UNet_Palette

class DirectUNetPrediction(nn.Module):
    def __init__(
        self,
        config,
        generated_channels,
        condition_channels,
        loss_fn
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.loss_fn = loss_fn

            # Neural Network Backbone
        if config.unet_type == "UnetConvNextBlock":
            self.model = UnetConvNextBlock(
                dim=config.num_channels_base,
                dim_mults=config.dims_mults,
                channels=condition_channels,
                out_dim=self.generated_channels,
                with_time_emb=False,
                    use_cyclical_padding=config.use_cyclical_padding
            )
        elif config.unet_type == "UNet_Palette":
            self.model = UNet_Palette(
                in_channel=self.condition_channels,
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

        return self.model(condition)

    def p_loss(self, output, condition):
        """
        Assumes output and input are in [-1,+1] range
        """
        b, c, h, w = output.shape
        device = output.device

        # apply loss
        return self.loss_fn(output, self.model(condition.to(device)))
