# Code from https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/main/snowification/diffusion/model/unet_convnext.py

import math
import torch
import torch.nn as nn
from inspect import isfunction
from einops import rearrange
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    """
    Based on transformer-like embedding from 'Attention is all you need'
    Note: 10,000 corresponds to the maximum sequence length
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)  # ideally would use cyclical padding here as well.


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)  # ideally would use cyclical padding here as well.


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True, use_cyclical_padding=False):
        super().__init__()
        
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = GeneralizedConv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, use_cyclical_padding=use_cyclical_padding, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            GeneralizedConv2d(in_channels=dim, out_channels=dim_out * mult, kernel_size=3, padding=1, use_cyclical_padding=use_cyclical_padding),
            nn.GELU(),
            GeneralizedConv2d(in_channels=dim_out * mult, out_channels=dim_out, kernel_size=3, padding=1, use_cyclical_padding=use_cyclical_padding),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        # print(x.shape, h.shape)
        return h + self.res_conv(x)
    

class Conv2dPadCyclical(nn.Module):
    """
    2d convolution that applies "cylindrical" padding: Zero pad height, cyclical pad width.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1, value=0):
        super(Conv2dPadCyclical, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups)
        self.value = value
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, pad=(self.padding, self.padding, 0, 0), mode="circular")
        x = F.pad(x, pad=(0, 0, self.padding, self.padding), mode="constant", value=self.value)
        x = self.conv(x)
        return x

class GeneralizedConv2d(nn.Module):
    """
    2d convolution, add functionality to run cylindrical padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_cyclical_padding, groups=1, pad_value=0):
        super(GeneralizedConv2d, self).__init__()
        if use_cyclical_padding:
            self.conv = Conv2dPadCyclical(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, groups=groups, value=pad_value) 
        else:
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
    
    def forward(self, x):
        return self.conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (h c) x y -> b h c (x y)",
                h=self.heads,
            ),
            qkv,
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out,
            "b h c (x y) -> b (h c) x y",
            h=self.heads,
            x=h,
            y=w,
        )
        return self.to_out(out)


# Main Model


class UnetConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        output_mean_scale=False,
        residual=False,
        use_cyclical_padding=False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        print("Cyclical Padding ? ", use_cyclical_padding)
        self.output_mean_scale = output_mean_scale

        dims = [
            channels,
            *map(lambda m: dim * m, dim_mults),
        ]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            dim=dim_in,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                            norm=ind != 0,
                            use_cyclical_padding=use_cyclical_padding
                        ),
                        ConvNextBlock(
                            dim=dim_out,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                            use_cyclical_padding=use_cyclical_padding
                        ),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(dim_out),
                            )
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim, use_cyclical_padding=use_cyclical_padding)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim, use_cyclical_padding=use_cyclical_padding)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            dim=dim_out * 2,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                            use_cyclical_padding=use_cyclical_padding
                        ),
                        ConvNextBlock(
                            dim=dim_in,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                            use_cyclical_padding=use_cyclical_padding
                        ),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(dim_in),
                            )
                        ),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1),  # kernel is 1x1 so padding doesn't matter anyway
            # nn.Tanh() # ADDED
        )

    def forward(self, x, time=None):
        orig_x = x
        t = None
        if time is not None and exists(self.time_mlp):
            t = self.time_mlp(time)

        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)
        h = []

        for (
            convnext,
            convnext2,
            attn,
            downsample,
        ) in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1, 2, 3], keepdim=True)
            out = out - original_mean + out_mean

        return out
