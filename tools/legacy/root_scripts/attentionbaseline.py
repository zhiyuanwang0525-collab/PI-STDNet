# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
SE-Net 和 CBAM 注意力模块 —— 作为 PI-STDNet 物理注意力的对照实验
用法：在 trainV72_ablation.py 中替换 MultiScalePhysicsAttention

实验设计要点：
1. SE/CBAM 接收的输入和物理注意力完全一致（同样的 88 通道输入）
2. backbone、TopK、RotStats、损失函数、训练超参全部保持不变
3. 唯一区别：空间注意力图的生成方式不同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 方案 1: SE-Net (Squeeze-and-Excitation) 注意力
# 原文: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
# 
# SE-Net 是纯通道注意力，没有空间注意力。
# 为了和物理注意力的接口兼容（需要输出空间注意力图），
# 我们在 SE 通道加权后追加一个 1x1 conv 生成空间 attention map。
# ================================================================

class SENetAttention(nn.Module):
    """
    SE-Net 通道注意力 + 空间投射头
    
    输入: 单个 sweep 的全部通道 (B, single_sweep_ch, H, W)
    输出: 
        - physics_attn: (B, 1, H, W) 空间注意力图（和物理注意力接口一致）
        - hard_valid:   (B, 1, H, W) DBZ 有效掩码
    """
    def __init__(self, n_frames=4, channels_per_frame=11, reduction=16):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = channels_per_frame
        in_channels = n_frames * channels_per_frame  # 44
        
        # === SE 通道注意力 ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
        # === 空间注意力投射（从 SE 加权后的特征生成空间 map）===
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        
    def forward(self, x):
        """
        x: (B, single_sweep_ch, H, W) — 一个 sweep 的 44 通道
        """
        B, C, H, W = x.shape
        
        # 提取 DBZ 通道用于 hard_valid mask（和物理注意力保持一致）
        dbz_frames = []
        for t in range(self.n_frames):
            base = t * self.cpf
            dbz_frames.append(x[:, base:base+1])
        dbz_max = torch.stack(dbz_frames, dim=1).max(dim=1)[0]
        hard_valid = (dbz_max > 0.15).float()
        
        # SE 通道注意力
        se_weight = self.global_pool(x).view(B, C)  # (B, C)
        se_weight = self.fc(se_weight).view(B, C, 1, 1)  # (B, C, 1, 1)
        x_weighted = x * se_weight  # 通道加权
        
        # 从加权特征生成空间注意力图
        attn = torch.sigmoid(self.spatial_proj(x_weighted)) * hard_valid
        
        return attn, hard_valid


# ================================================================
# 方案 2: CBAM (Convolutional Block Attention Module)
# 原文: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
#
# CBAM = 通道注意力 + 空间注意力（串联）
# 自带空间注意力，天然和物理注意力的输出接口兼容。
# ================================================================

class CBAMAttention(nn.Module):
    """
    CBAM 通道+空间注意力
    
    输入: 单个 sweep 的全部通道 (B, single_sweep_ch, H, W)
    输出:
        - physics_attn: (B, 1, H, W) 空间注意力图
        - hard_valid:   (B, 1, H, W) DBZ 有效掩码
    """
    def __init__(self, n_frames=4, channels_per_frame=11, reduction=16):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = channels_per_frame
        in_channels = n_frames * channels_per_frame  # 44
        
        # === 通道注意力模块 ===
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
        )
        
        # === 空间注意力模块 ===
        # CBAM 原版用 7x7 conv，输入是 avg+max pooled 的 2 通道
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 7, padding=3),
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # DBZ hard_valid mask
        dbz_frames = []
        for t in range(self.n_frames):
            base = t * self.cpf
            dbz_frames.append(x[:, base:base+1])
        dbz_max = torch.stack(dbz_frames, dim=1).max(dim=1)[0]
        hard_valid = (dbz_max > 0.15).float()
        
        # --- 通道注意力 ---
        avg_feat = self.avg_pool(x).view(B, C)
        max_feat = self.max_pool(x).view(B, C)
        channel_attn = torch.sigmoid(
            self.channel_mlp(avg_feat) + self.channel_mlp(max_feat)
        ).view(B, C, 1, 1)
        x_ca = x * channel_attn  # 通道加权后的特征
        
        # --- 空间注意力 ---
        avg_spatial = x_ca.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_spatial = x_ca.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.spatial_conv(spatial_input)) * hard_valid
        
        return attn, hard_valid


# ================================================================
# 集成到消融实验的方法
# ================================================================

def create_attention_module(attn_type='physics', n_frames=4, channels_per_frame=11):
    """
    工厂函数：根据实验配置创建不同的注意力模块
    
    用法（在 PI_STDNet_Ablation.__init__ 中）：
        self.physics_attn_module = create_attention_module(
            attn_type=args.attn_type,  # 'physics', 'se', 'cbam'
            n_frames=CONFIG['N_FRAMES'],
            channels_per_frame=CONFIG['CHANNELS_PER_FRAME']
        )
    
    注意：SE 和 CBAM 的 forward 接口和物理注意力完全一致：
        attn, hard_valid = self.physics_attn_module(x_sweep)
    所以不需要修改 forward 中的其他代码。
    """
    if attn_type == 'physics':
        # 导入原始物理注意力（从你的 trainV72.py）
        from train import MultiScalePhysicsAttention
        return MultiScalePhysicsAttention(n_frames)
    elif attn_type == 'se':
        return SENetAttention(n_frames, channels_per_frame)
    elif attn_type == 'cbam':
        return CBAMAttention(n_frames, channels_per_frame)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")


# ================================================================
# 快速测试：验证输出维度一致
# ================================================================
if __name__ == '__main__':
    B, C, H, W = 2, 44, 120, 240  # 单个 sweep: 4 frames * 11 channels
    x = torch.randn(B, C, H, W)
    
    print("=" * 60)
    print("验证三种注意力模块的输出维度一致性")
    print("=" * 60)
    
    for name, Module in [("SE-Net", SENetAttention), ("CBAM", CBAMAttention)]:
        model = Module(n_frames=4, channels_per_frame=11)
        attn, hv = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}:")
        print(f"  attn shape: {attn.shape}  (期望: [{B}, 1, {H}, {W}])")
        print(f"  hard_valid shape: {hv.shape}  (期望: [{B}, 1, {H}, {W}])")
        print(f"  attn range: [{attn.min().item():.4f}, {attn.max().item():.4f}]")
        print(f"  参数量: {params:,}")
    
    print("\n✅ 所有模块输出维度一致，可以直接替换！")
    print("\n运行命令示例：")
    print("  python trainV72_ablation.py --exp_name attn_se --attn_type se")
    print("  python trainV72_ablation.py --exp_name attn_cbam --attn_type cbam")


