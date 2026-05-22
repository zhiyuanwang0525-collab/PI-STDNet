"""Physics Attention Module (PAM) and related attention baselines."""

from __future__ import annotations

import torch
from torch import nn


class MultiScalePhysicsAttention(nn.Module):
    """Generate a spatial physics-guided attention map from MAAS and debris cues."""

    def __init__(self, n_frames: int = 4, channels_per_frame: int = 11):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = channels_per_frame
        self.frame_conv = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_feats, dbz_frames = [], []
        for t in range(self.n_frames):
            base = t * self.cpf
            dbz = x[:, base : base + 1]
            az_s = x[:, base + 8 : base + 9]
            az_m = x[:, base + 9 : base + 10]
            az_l = x[:, base + 10 : base + 11]
            tds = x[:, base + 7 : base + 8]
            vdiff = torch.abs(x[:, base + 1 : base + 2] - x[:, (t - 1) * self.cpf + 1 : (t - 1) * self.cpf + 2]) if t > 0 else torch.zeros_like(az_m)
            frame_feats.append(self.frame_conv(torch.cat([az_s, az_m, az_l, vdiff, tds], dim=1)))
            dbz_frames.append(dbz)
        dbz_max = torch.stack(dbz_frames, dim=1).max(dim=1)[0]
        hard_valid = (dbz_max > 0.27).float()
        return torch.sigmoid(self.fusion(torch.stack(frame_feats, dim=0).max(dim=0)[0])) * hard_valid, hard_valid


class SENetAttention(nn.Module):
    """SE-Net attention baseline with the same output interface as PAM."""

    def __init__(self, n_frames: int = 4, channels_per_frame: int = 11, reduction: int = 16):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = channels_per_frame
        in_channels = n_frames * channels_per_frame
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels), nn.Sigmoid())
        self.spatial_proj = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _, _ = x.shape
        dbz_max = torch.stack([x[:, t * self.cpf : t * self.cpf + 1] for t in range(self.n_frames)], dim=1).max(dim=1)[0]
        hard_valid = (dbz_max > 0.15).float()
        weights = self.fc(self.global_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return torch.sigmoid(self.spatial_proj(x * weights)) * hard_valid, hard_valid


class CBAMAttention(nn.Module):
    """CBAM attention baseline with the same output interface as PAM."""

    def __init__(self, n_frames: int = 4, channels_per_frame: int = 11, reduction: int = 16):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = channels_per_frame
        in_channels = n_frames * channels_per_frame
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(nn.Linear(in_channels, in_channels // reduction), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels))
        self.spatial_conv = nn.Sequential(nn.Conv2d(2, 16, 7, padding=3), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 7, padding=3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _, _ = x.shape
        dbz_max = torch.stack([x[:, t * self.cpf : t * self.cpf + 1] for t in range(self.n_frames)], dim=1).max(dim=1)[0]
        hard_valid = (dbz_max > 0.15).float()
        avg_feat = self.avg_pool(x).view(batch, channels)
        max_feat = self.max_pool(x).view(batch, channels)
        channel_attn = torch.sigmoid(self.channel_mlp(avg_feat) + self.channel_mlp(max_feat)).view(batch, channels, 1, 1)
        x_ca = x * channel_attn
        spatial_input = torch.cat([x_ca.mean(dim=1, keepdim=True), x_ca.max(dim=1, keepdim=True)[0]], dim=1)
        return torch.sigmoid(self.spatial_conv(spatial_input)) * hard_valid, hard_valid


def create_attention_module(attn_type: str = "physics", n_frames: int = 4, channels_per_frame: int = 11) -> nn.Module:
    """Create the attention module selected by an ablation config."""
    if attn_type == "physics":
        return MultiScalePhysicsAttention(n_frames, channels_per_frame)
    if attn_type == "se":
        return SENetAttention(n_frames, channels_per_frame)
    if attn_type == "cbam":
        return CBAMAttention(n_frames, channels_per_frame)
    raise ValueError(f"Unknown attention type: {attn_type}")

