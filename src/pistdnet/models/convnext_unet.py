"""ConvNeXt-U-Net building blocks used by PI-STDNet."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm supporting channels-first tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            orig_dtype = x.dtype
            x_float = x.float()
            mean = x_float.mean(1, keepdim=True)
            var = (x_float - mean).pow(2).mean(1, keepdim=True)
            x_norm = (x_float - mean) / torch.sqrt(var + 1e-4)
            return (self.weight[:, None, None].float() * x_norm + self.bias[:, None, None].float()).to(orig_dtype)
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths per sample as in the original ConvNeXt-style block."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=torch.float32, device=x.device)
    random_tensor.bernoulli_(keep_prob)
    return x / keep_prob * random_tensor.to(x.dtype)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block used in the encoder and decoder."""

    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x))) * self.gamma
        return residual + drop_path(x.permute(0, 3, 1, 2), self.drop_path_rate, self.training)


class AddCoords2d(nn.Module):
    """Append normalized y/x coordinate channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat([x, y_coords, x_coords], dim=1)


class UpSampleBlock(nn.Module):
    """Decoder block with bilinear upsampling and ConvNeXt refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.block = ConvNeXtBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.block(self.reduce(torch.cat([x, skip], dim=1)))

