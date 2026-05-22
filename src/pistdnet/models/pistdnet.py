"""PI-STDNet model, losses, and EMA helper."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .convnext_unet import AddCoords2d, ConvNeXtBlock, LayerNorm, UpSampleBlock
from .pam import create_attention_module


DEFAULT_MODEL_CONFIG = {
    "input_shape": [120, 240],
    "n_frames": 4,
    "channels_per_frame": 11,
    "n_sweeps": 2,
    "dims": [32, 64, 128, 256],
    "depths": [3, 3, 9, 3],
    "drop_path_rate": 0.2,
    "topk": 16,
    "attention_type": "physics",
    "label_smoothing": 0.1,
}


class RotationStatisticsExtractor(nn.Module):
    """Extract compact rotation statistics used by the classifier head."""

    def __init__(self, n_frames: int = 4, n_sweeps: int = 2, channels_per_frame: int = 11):
        super().__init__()
        self.n_frames = n_frames
        self.n_sweeps = n_sweeps
        self.cpf = channels_per_frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        all_stats = []
        for sweep in range(self.n_sweeps):
            base = sweep * self.n_frames * self.cpf
            for scale_off in [8, 9, 10]:
                all_stats.append(torch.stack([x[:, base + t * self.cpf + scale_off : base + t * self.cpf + scale_off + 1].view(batch, -1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base + t * self.cpf + 9 : base + t * self.cpf + 10] > 0.5).float().mean(dim=(1, 2, 3)) for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base + t * self.cpf + 1 : base + t * self.cpf + 2] - x[:, base + (t - 1) * self.cpf + 1 : base + (t - 1) * self.cpf + 2]).abs().view(batch, -1).max(dim=1)[0] for t in range(1, self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([x[:, base + t * self.cpf + 7 : base + t * self.cpf + 8].view(batch, -1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
        return torch.stack(all_stats, dim=1)


class PhysicsGuidedTopKPool(nn.Module):
    """Original physics-guided top-k pooling helper retained for ablations."""

    def __init__(self, feat_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.topk_proj = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim // 2), nn.GELU())
        self.fuse = nn.Linear(feat_dim + feat_dim // 2, feat_dim)

    def forward(self, feat_map: torch.Tensor, physics_attn: torch.Tensor) -> torch.Tensor:
        batch, dim, height, width = feat_map.shape
        attn_flat = F.interpolate(physics_attn, size=(height, width), mode="bilinear", align_corners=True).view(batch, -1)
        feat_flat = feat_map.view(batch, dim, -1)
        topk_idx = torch.topk(attn_flat, min(self.k, height * width), dim=1)[1].unsqueeze(1).expand(-1, dim, -1)
        topk_feats = torch.gather(feat_flat, 2, topk_idx)
        pooled = self.topk_proj(torch.cat([topk_feats.mean(dim=2), topk_feats.max(dim=2)[0]], dim=1))
        global_feat = self.global_proj(feat_map.mean(dim=(2, 3)))
        return self.fuse(torch.cat([pooled, global_feat], dim=1))


class PI_STDNet(nn.Module):
    """Physics-guided ConvNeXt-U-Net for TorNet tornado detection."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_MODEL_CONFIG, **(config or {})}
        self.config = cfg
        self.input_shape = tuple(cfg["input_shape"])
        self.n_frames = int(cfg["n_frames"])
        self.channels_per_frame = int(cfg["channels_per_frame"])
        self.n_sweeps = int(cfg["n_sweeps"])
        single_sweep_ch = self.n_frames * self.channels_per_frame

        self.physics_attn_module = create_attention_module(cfg.get("attention_type", "physics"), self.n_frames, self.channels_per_frame)
        self.add_coords = AddCoords2d()
        self.sweep_fuse = nn.Sequential(nn.Conv2d(single_sweep_ch * self.n_sweeps + 2, single_sweep_ch, 3, padding=1), LayerNorm(single_sweep_ch), nn.GELU())
        self.rot_stats = RotationStatisticsExtractor(self.n_frames, self.n_sweeps, self.channels_per_frame)

        dims = list(cfg["dims"])
        depths = list(cfg["depths"])
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(single_sweep_ch, dims[0], 4, stride=4), LayerNorm(dims[0]))])
        self.downsample_layers.extend([nn.Sequential(LayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)) for i in range(3)])

        dp_rates = [x.item() for x in torch.linspace(0, float(cfg["drop_path_rate"]), sum(depths))]
        self.stages = nn.ModuleList([nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[sum(depths[:i]) + j]) for j in range(depths[i])]) for i in range(4)])

        self.up1 = UpSampleBlock(dims[3], dims[2], dims[2])
        self.up2 = UpSampleBlock(dims[2], dims[1], dims[1])
        self.up3 = UpSampleBlock(dims[1], dims[0], dims[0])
        self.spatial_head = nn.Sequential(nn.Conv2d(dims[0], 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

        self.gmp_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.rot_proj = nn.Sequential(nn.Linear(6 * self.n_sweeps, 128), nn.LayerNorm(128), nn.GELU())
        cls_in_dim = dims[1] + dims[3] + 128
        self.cls_head = nn.Sequential(nn.LayerNorm(cls_in_dim), nn.Linear(cls_in_dim, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, 1))
        self.aux_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.aux_head = nn.Sequential(nn.Flatten(), nn.Linear(dims[0] * 16, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rot_stats = self.rot_stats(x)
        single_ch = self.n_frames * self.channels_per_frame
        sweep_enhanced, physics_attns, hard_valids = [], [], []
        for sweep in range(self.n_sweeps):
            x_sweep = x[:, sweep * single_ch : (sweep + 1) * single_ch, :, :]
            pa, hv = self.physics_attn_module(x_sweep)
            sweep_enhanced.append(x_sweep * (1.0 + pa))
            physics_attns.append(pa)
            hard_valids.append(hv)

        if len(physics_attns) == 1:
            physics_attn = physics_attns[0] * hard_valids[0]
        else:
            physics_attn = (0.7 * physics_attns[0] + 0.3 * physics_attns[1]) * torch.stack(hard_valids, dim=0).max(dim=0)[0]

        feat = self.sweep_fuse(self.add_coords(torch.cat(sweep_enhanced, dim=1)))
        skips = []
        for i in range(4):
            feat = self.stages[i](self.downsample_layers[i](feat))
            skips.append(feat)

        cls_feat = torch.cat([self.gmp_pool(skips[1]).view(x.shape[0], -1), skips[3].mean(dim=(2, 3)), self.rot_proj(rot_stats)], dim=1)
        cls_logit = self.cls_head(cls_feat).squeeze(-1)
        x_decoded = self.up3(self.up2(self.up1(skips[3], skips[2]), skips[1]), skips[0])
        hard_valid = torch.stack(hard_valids, dim=0).max(dim=0)[0]
        spatial_logits = (F.interpolate(self.spatial_head(x_decoded), size=self.input_shape, mode="bilinear", align_corners=True) - 5.0 * (1.0 - hard_valid)).clamp(-20.0, 20.0)
        return cls_logit, self.aux_head(self.aux_pool(x_decoded)).squeeze(-1), spatial_logits, physics_attn


PI_STDNet_V8 = PI_STDNet


class FocalLoss(nn.Module):
    """Binary focal loss used by the original training script."""

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.ls) + 0.5 * self.ls if self.ls > 0 else targets
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return ((self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


class V8Loss(nn.Module):
    """Combined classification, auxiliary, and physics consistency loss."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=0.70, gamma=2.0, label_smoothing=label_smoothing)

    def forward(self, cls_logit: torch.Tensor, aux_logit: torch.Tensor, spatial_logits: torch.Tensor, physics_attn: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        loss_main = self.focal(cls_logit, targets)
        loss_aux = self.focal(aux_logit, targets)
        pos_mask = (targets > 0.5).float()
        if pos_mask.sum() > 0:
            loss_consist = (F.mse_loss(torch.sigmoid(spatial_logits.float()).view(spatial_logits.size(0), -1), physics_attn.detach().float().view(physics_attn.size(0), -1), reduction="none").mean(dim=1) * pos_mask).sum() / (pos_mask.sum() + 1e-7)
        else:
            loss_consist = torch.tensor(0.0, device=targets.device)
        return 0.5 * loss_main + 0.25 * loss_aux + 2.0 * loss_consist, loss_main.item()


class EMA:
    """Exponential moving average wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]

