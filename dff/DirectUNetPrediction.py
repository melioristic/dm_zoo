
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenoisingDiffusionProcess.forward import *
from .DenoisingDiffusionProcess.samplers import *
from .DenoisingDiffusionProcess.backbones.unet_convnext import *

class DirectUNetPrediction(nn.Module):
    def __init__(
        self,
        generated_channels,
        condition_channels,
        loss_fn=F.mse_loss,
        cylindrical_padding=False
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.loss_fn = loss_fn

        # Neural Network Backbone
        self.model = UnetConvNextBlock(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=condition_channels,
            out_dim=self.generated_channels,
            with_time_emb=False,
            cylindrical_padding=cylindrical_padding
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

        return self.model(condition)

    def p_loss(self, output, condition):
        """
        Assumes output and input are in [-1,+1] range
        """
        b, c, h, w = output.shape
        device = output.device

        # apply loss
        return self.loss_fn(output, self.model(condition.to(device)))
