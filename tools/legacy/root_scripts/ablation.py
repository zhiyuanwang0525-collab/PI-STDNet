# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
PI-STDNet V8.3 严谨消融实验脚本
====================================
与满血版 trainV8fix.py 的唯一差异：通过 --disable_* 开关禁用单个模块。
所有超参数、训练策略、数据加载、EMA、学习率调度 100% 对齐满血版。

消融组别：
  1. python ablation_v83.py --exp_name full          # 消融baseline（满血，应≈0.39）
  2. python ablation_v83.py --exp_name no_physics_input --disable_physics_inputs
  3. python ablation_v83.py --exp_name no_physics_attn --disable_physics_attn
  4. python ablation_v83.py --exp_name no_topk --disable_topk
  5. python ablation_v83.py --exp_name no_rot_stats --disable_rot_stats
"""

import os, sys, argparse, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, numpy as np, pandas as pd, xarray as xr
import scipy.ndimage as ndimage, random, logging, math, gc
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================= 1. 消融开关 =================
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='full')
parser.add_argument('--disable_physics_inputs', action='store_true')
parser.add_argument('--disable_physics_attn', action='store_true')
parser.add_argument('--disable_topk', action='store_true')
parser.add_argument('--disable_rot_stats', action='store_true')
args = parser.parse_args()

# ================= 2. CONFIG — 严格对齐 V8.3 满血版 =================
CONFIG = {
    'CATALOG_PATH': '/path/to/TorNet/catalog.csv',
    'DATA_ROOT': r'/path/to/TorNet',
    'BATCH_SIZE': 32, 'ACCUM_STEPS': 2,
    'NUM_WORKERS': 12, 'PREFETCH_FACTOR': 4,  # 消融实验适当降低防OOM
    'USE_AMP': True,
    'LEARNING_RATE': 5e-4,   # ← 和满血版一致
    'WEIGHT_DECAY': 0.05,    # ← 和满血版一致
    'NUM_EPOCHS': 15,        # 消融用15epoch节省时间（足够对比趋势）
    'WARMUP_EPOCHS': 5,      # ← 和满血版一致
    'DIMS': [32, 64, 128, 256], 'DEPTHS': [3, 3, 9, 3],
    'INPUT_SHAPE': (120, 240), 'N_FRAMES': 4, 'N_SWEEPS': 2, 'TOPK': 16,
    'LABEL_SMOOTHING': 0.1, 'DROP_PATH_RATE': 0.2,
    # 输入通道数：禁用物理输入时6通道，否则11通道
    'CHANNELS_PER_FRAME': 6 if args.disable_physics_inputs else 11,
}

# ================= 3. 日志 =================
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
fh = logging.FileHandler(f"ablation_{args.exp_name}.log", mode='a', encoding='utf-8')
fh.setFormatter(log_formatter)
logger.addHandler(fh)

logger.info(f"🧪 消融实验: {args.exp_name}")
logger.info(f"  disable_physics_inputs={args.disable_physics_inputs}")
logger.info(f"  disable_physics_attn={args.disable_physics_attn}")
logger.info(f"  disable_topk={args.disable_topk}")
logger.info(f"  disable_rot_stats={args.disable_rot_stats}")
logger.info(f"  CHANNELS_PER_FRAME={CONFIG['CHANNELS_PER_FRAME']}")
logger.info(f"  关键超参: α=0.70, sampler_TOR=7.0, mask=0.27, warmup=5, lr=5e-4")

CHANNEL_MIN_MAX = {
    'DBZ': (-30.0, 80.0), 'VEL': (-50.0, 50.0), 'KDP': (-2.0, 5.0),
    'RHOHV': (0.2, 1.05), 'ZDR': (-5.0, 8.0), 'WIDTH': (0.0, 10.0),
    'SHEAR': (-10.0, 10.0), 'DEBRIS': (0.0, 1.0),
}

# ================= 4. 物理特征（和满血版完全相同） =================
class PhysicalFeatureAmplifier:
    @staticmethod
    def compute_shear(vel):
        return np.nan_to_num(np.gradient(np.nan_to_num(vel, nan=0.0), axis=1), nan=0.0)
    @staticmethod
    def compute_debris(dbz, rhohv):
        return np.clip((np.nan_to_num(dbz, nan=-30.0) - (-30)) / 110.0, 0, 1) * \
               (1.0 - np.clip(np.nan_to_num(rhohv, nan=0.0), 0, 1))
    @staticmethod
    def compute_multiscale_anomaly_shear(vel, dbz):
        vel = np.nan_to_num(vel, nan=0.0)
        eroded_mask = ndimage.binary_erosion((dbz > -29.0), iterations=2)
        results = []
        for kernel in [3, 5, 7]:
            half = kernel // 2
            padded = np.pad(vel, ((half, half), (0, 0)), mode='wrap')
            az_shear = np.zeros_like(vel)
            for i in range(vel.shape[0]):
                window = padded[i:i+kernel, :]
                az_shear[i] = window.max(axis=0) - window.min(axis=0)
            anomaly = np.clip(az_shear - ndimage.uniform_filter(az_shear, size=15), 0, None)
            results.append(anomaly * eroded_mask)
        return results

# ================= 5. 数据集（和满血版相同，按开关控制通道） =================
class V8Augmentation:
    def __init__(self, mode='train'): self.mode = mode
    def __call__(self, data):
        if self.mode != 'train': return data
        C, H, W = data.shape
        if random.random() > 0.5: data = np.flip(data, axis=2).copy()
        if random.random() > 0.5: data = np.flip(data, axis=1).copy()
        cpf = CONFIG['CHANNELS_PER_FRAME']
        safe_channels = [0, 2, 3, 4, 5]
        if random.random() > 0.5:
            for s in range(CONFIG['N_SWEEPS']):
                for t in range(CONFIG['N_FRAMES']):
                    base = s * (CONFIG['N_FRAMES'] * cpf) + t * cpf
                    for c_offset in safe_channels:
                        c = base + c_offset
                        if c < C:
                            data[c] = data[c] * random.uniform(0.95, 1.05) + random.uniform(-0.02, 0.02)
        if random.random() > 0.5:
            eh, ew = random.randint(10, 25), random.randint(20, 50)
            ey, ex = random.randint(0, max(1, H - eh)), random.randint(0, max(1, W - ew))
            data[:, ey:ey+eh, ex:ex+ew] = 0.0
        return data

class TorNetDatasetAblation(Dataset):
    def __init__(self, catalog_path, root_dir, mode='train'):
        self.root_dir, self.mode = root_dir, mode
        df = pd.read_csv(catalog_path)
        self.catalog = df[df['type'] == mode].reset_index(drop=True)
        self.augmentor = V8Augmentation(mode=mode)
        self.labels = (self.catalog['category'] == 'TOR').astype(int).values
        self.cat_ids = np.zeros(len(self.catalog), dtype=int)
        self.cat_ids[self.catalog['category'] == 'TOR'] = 2
        wrn_mask = (self.catalog['category'] == 'WRN') | (self.catalog['category'] == 'Tornado Warning')
        self.cat_ids[wrn_mask] = 1
        n_tor = (self.cat_ids == 2).sum()
        n_wrn = (self.cat_ids == 1).sum()
        n_nul = (self.cat_ids == 0).sum()
        logger.info(f"📦 [{mode.upper()}] TOR: {n_tor} | WRN: {n_wrn} | NUL: {n_nul}")

    def __len__(self): return len(self.catalog)
    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
        filepath = os.path.join(self.root_dir, f"tornet_{year}", row['filename'])
        if not os.path.exists(filepath): filepath = os.path.join(self.root_dir, row['filename'])
        try:
            with xr.open_dataset(filepath, engine='netcdf4', cache=False) as ds:
                def safe_read(key, fill_val, sweep_idx=0):
                    if key not in ds: return np.full((4, 120, 240), fill_val, dtype=np.float32)
                    raw = ds[key].values
                    data = raw[..., sweep_idx] if raw.ndim == 4 and raw.shape[-1] > sweep_idx else raw[..., 0] if raw.ndim == 4 else raw
                    return np.nan_to_num(data, nan=fill_val).astype(np.float32)
                def norm(arr, key):
                    mn, mx = CHANNEL_MIN_MAX[key]
                    return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

                all_channels = []
                for sweep_idx in range(CONFIG['N_SWEEPS']):
                    dbz, vel = safe_read('DBZ', -30.0, sweep_idx), safe_read('VEL', 0.0, sweep_idx)
                    kdp, rhohv = safe_read('KDP', 0.0, sweep_idx), safe_read('RHOHV', 0.0, sweep_idx)
                    zdr, width = safe_read('ZDR', 0.0, sweep_idx), safe_read('WIDTH', 0.0, sweep_idx)
                    frames_idx = list(range(min(dbz.shape[0], CONFIG['N_FRAMES'])))
                    while len(frames_idx) < CONFIG['N_FRAMES']: frames_idx.append(frames_idx[-1])
                    for f_idx in frames_idx:
                        f_dbz, f_vel = dbz[f_idx], vel[f_idx]
                        base_ch = [norm(f_dbz, 'DBZ'), norm(f_vel, 'VEL'), norm(kdp[f_idx], 'KDP'),
                                   norm(rhohv[f_idx], 'RHOHV'), norm(zdr[f_idx], 'ZDR'), norm(width[f_idx], 'WIDTH')]
                        if args.disable_physics_inputs:
                            all_channels.extend(base_ch)
                        else:
                            anomaly_shears = PhysicalFeatureAmplifier.compute_multiscale_anomaly_shear(f_vel, f_dbz)
                            all_channels.extend(base_ch + [
                                norm(PhysicalFeatureAmplifier.compute_shear(f_vel), 'SHEAR'),
                                np.clip(PhysicalFeatureAmplifier.compute_debris(f_dbz, rhohv[f_idx]), 0, 1),
                                np.clip(anomaly_shears[0] / 20.0, 0.0, 1.0),
                                np.clip(anomaly_shears[1] / 20.0, 0.0, 1.0),
                                np.clip(anomaly_shears[2] / 20.0, 0.0, 1.0),
                            ])
                data = np.clip(np.nan_to_num(np.stack(all_channels, axis=0).astype(np.float32)), -1.0, 2.0)
                return torch.from_numpy(self.augmentor(data.copy())), \
                       torch.tensor(self.labels[idx], dtype=torch.float32), \
                       torch.tensor(self.cat_ids[idx], dtype=torch.int8)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.catalog) - 1))

# ================= 6. 网络组件（和满血版完全相同） =================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps, self.data_format = eps, data_format
    def forward(self, x):
        if self.data_format == "channels_first":
            orig_dtype = x.dtype
            u = x.float().mean(1, keepdim=True)
            s = (x.float() - u).pow(2).mean(1, keepdim=True)
            x = (self.weight[:, None, None].float() * ((x.float() - u) / torch.sqrt(s + 1e-4)) + self.bias[:, None, None].float())
            return x.to(orig_dtype)
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training: return x
    keep = 1 - drop_prob
    r = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=torch.float32, device=x.device).bernoulli_(keep)
    return x / keep * r.to(x.dtype)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1, self.pwconv2 = nn.Linear(dim, 4*dim), nn.Linear(4*dim, dim)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path_rate = drop_path_rate
    def forward(self, x):
        res = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)).permute(0,2,3,1)))) * self.gamma
        return res + drop_path(x.permute(0,3,1,2), self.drop_path_rate, self.training)

class AddCoords2d(nn.Module):
    def forward(self, x):
        B,C,H,W = x.shape
        yc = torch.linspace(-1,1,H,device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xc = torch.linspace(-1,1,W,device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, yc, xc], dim=1)

class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.block = ConvNeXtBlock(out_ch)
    def forward(self, x, skip):
        return self.block(self.reduce(torch.cat([F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True), skip], dim=1)))

# ================= 7. 物理模块（按开关控制） =================
class MultiScalePhysicsAttention(nn.Module):
    def __init__(self, n_frames=4):
        super().__init__()
        self.n_frames = n_frames
        self.cpf = CONFIG['CHANNELS_PER_FRAME']
        self.frame_conv = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
    def forward(self, x):
        frame_feats, dbz_frames = [], []
        for t in range(self.n_frames):
            base = t * self.cpf
            dbz = x[:, base:base+1]
            az_s, az_m, az_l = x[:, base+8:base+9], x[:, base+9:base+10], x[:, base+10:base+11]
            tds = x[:, base+7:base+8]
            vdiff = torch.abs(x[:, base+1:base+2] - x[:, (t-1)*self.cpf+1:(t-1)*self.cpf+2]) if t > 0 else torch.zeros_like(az_m)
            frame_feats.append(self.frame_conv(torch.cat([az_s, az_m, az_l, vdiff, tds], dim=1)))
            dbz_frames.append(dbz)
        dbz_max = torch.stack(dbz_frames, dim=1).max(dim=1)[0]
        # ← V8.3: mask=0.27
        return torch.sigmoid(self.fusion(torch.stack(frame_feats, dim=0).max(dim=0)[0])) * (dbz_max > 0.27).float(), (dbz_max > 0.27).float()

class RotationStatisticsExtractor(nn.Module):
    def __init__(self, n_frames=4, n_sweeps=2):
        super().__init__()
        self.n_frames, self.n_sweeps, self.cpf = n_frames, n_sweeps, CONFIG['CHANNELS_PER_FRAME']
    def forward(self, x):
        B = x.shape[0]
        all_stats = []
        for s in range(self.n_sweeps):
            base = s * self.n_frames * self.cpf
            for scale_off in [8, 9, 10]:
                all_stats.append(torch.stack([x[:, base+t*self.cpf+scale_off:base+t*self.cpf+scale_off+1].view(B,-1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base+t*self.cpf+9:base+t*self.cpf+10] > 0.5).float().mean(dim=(1,2,3)) for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base+t*self.cpf+1:base+t*self.cpf+2] - x[:, base+(t-1)*self.cpf+1:base+(t-1)*self.cpf+2]).abs().view(B,-1).max(dim=1)[0] for t in range(1, self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([x[:, base+t*self.cpf+7:base+t*self.cpf+8].view(B,-1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
        return torch.stack(all_stats, dim=1)

class PhysicsGuidedTopKPool(nn.Module):
    def __init__(self, feat_dim, k=16):
        super().__init__()
        self.k = k
        self.topk_proj = nn.Sequential(nn.Linear(feat_dim*2, feat_dim), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim//2), nn.GELU())
        self.fuse = nn.Linear(feat_dim + feat_dim//2, feat_dim)
    def forward(self, feat_map, physics_attn):
        B, D, Hf, Wf = feat_map.shape
        attn_flat = F.interpolate(physics_attn, size=(Hf,Wf), mode='bilinear', align_corners=True).view(B,-1)
        feat_flat = feat_map.view(B, D, -1)
        topk_idx = torch.topk(attn_flat, min(self.k, Hf*Wf), dim=1)[1].unsqueeze(1).expand(-1, D, -1)
        topk_feats = torch.gather(feat_flat, 2, topk_idx)
        return self.fuse(torch.cat([
            self.topk_proj(torch.cat([topk_feats.mean(2), topk_feats.max(2)[0]], dim=1)),
            self.global_proj(feat_map.mean(dim=(2,3)))
        ], dim=1))

# ================= 8. 消融主架构 =================
class PI_STDNet_Ablation(nn.Module):
    def __init__(self):
        super().__init__()
        single_sweep_ch = CONFIG['N_FRAMES'] * CONFIG['CHANNELS_PER_FRAME']
        dims, depths = CONFIG['DIMS'], CONFIG['DEPTHS']

        # 物理注意力：仅在有物理输入且未禁用注意力时启用
        self.use_physics_attn = (not args.disable_physics_attn) and (not args.disable_physics_inputs)
        if self.use_physics_attn:
            self.physics_attn_module = MultiScalePhysicsAttention(CONFIG['N_FRAMES'])

        # 旋转统计：仅在有物理输入且未禁用旋转统计时启用
        self.use_rot_stats = (not args.disable_rot_stats) and (not args.disable_physics_inputs)
        if self.use_rot_stats:
            self.rot_stats = RotationStatisticsExtractor(CONFIG['N_FRAMES'], CONFIG['N_SWEEPS'])
            self.rot_proj = nn.Sequential(nn.Linear(6 * CONFIG['N_SWEEPS'], 128), nn.LayerNorm(128), nn.GELU())

        # TopK池化
        self.use_topk = (not args.disable_topk) and self.use_physics_attn
        if self.use_topk:
            self.topk_pool = PhysicsGuidedTopKPool(dims[1], k=CONFIG['TOPK'])

        self.add_coords = AddCoords2d()
        self.sweep_fuse = nn.Sequential(
            nn.Conv2d(single_sweep_ch * CONFIG['N_SWEEPS'] + 2, single_sweep_ch, 3, padding=1),
            LayerNorm(single_sweep_ch), nn.GELU())

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(single_sweep_ch, dims[0], 4, stride=4), LayerNorm(dims[0]))])
        self.downsample_layers.extend([nn.Sequential(LayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i+1], 2, stride=2)) for i in range(3)])
        dp_rates = [x.item() for x in torch.linspace(0, CONFIG['DROP_PATH_RATE'], sum(depths))]
        self.stages = nn.ModuleList([nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[sum(depths[:i])+j]) for j in range(depths[i])]) for i in range(4)])

        self.up1 = UpSampleBlock(dims[3], dims[2], dims[2])
        self.up2 = UpSampleBlock(dims[2], dims[1], dims[1])
        self.up3 = UpSampleBlock(dims[1], dims[0], dims[0])
        self.spatial_head = nn.Sequential(nn.Conv2d(dims[0], 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

        # 分类头输入维度
        cls_in_dim = dims[3]  # 全局特征（始终存在）
        if self.use_topk:
            cls_in_dim += dims[1]
        else:
            # 用GMP替代TopK，保持维度一致
            self.gmp_fallback = nn.AdaptiveMaxPool2d((1, 1))
            cls_in_dim += dims[1]
        if self.use_rot_stats:
            cls_in_dim += 128

        self.cls_head = nn.Sequential(nn.LayerNorm(cls_in_dim), nn.Linear(cls_in_dim, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, 1))
        self.aux_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.aux_head = nn.Sequential(nn.Flatten(), nn.Linear(dims[0]*16, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        B = x.shape[0]
        single_ch = CONFIG['N_FRAMES'] * CONFIG['CHANNELS_PER_FRAME']

        # 旋转统计
        rot_feats = self.rot_proj(self.rot_stats(x)) if self.use_rot_stats else None

        # 物理注意力 + sweep增强
        sweep_enhanced, physics_attns, hard_valids = [], [], []
        for s in range(CONFIG['N_SWEEPS']):
            x_sweep = x[:, s*single_ch:(s+1)*single_ch]
            if self.use_physics_attn:
                pa, hv = self.physics_attn_module(x_sweep)
                sweep_enhanced.append(x_sweep * (1.0 + pa))
                physics_attns.append(pa)
                hard_valids.append(hv)
            else:
                sweep_enhanced.append(x_sweep)

        if self.use_physics_attn:
            physics_attn = (0.7 * physics_attns[0] + 0.3 * physics_attns[1]) * torch.stack(hard_valids, dim=0).max(dim=0)[0]
        else:
            physics_attn = None

        # 主干
        feat = self.sweep_fuse(self.add_coords(torch.cat(sweep_enhanced, dim=1)))
        skips = []
        for i in range(4):
            feat = self.stages[i](self.downsample_layers[i](feat))
            skips.append(feat)

        # 分类特征
        cls_features = [skips[3].mean(dim=(2, 3))]
        if self.use_topk:
            cls_features.append(self.topk_pool(skips[1], physics_attn))
        else:
            cls_features.append(self.gmp_fallback(skips[1]).view(B, -1))
        if self.use_rot_stats:
            cls_features.append(rot_feats)

        cls_logit = self.cls_head(torch.cat(cls_features, dim=1)).squeeze(-1)

        # 解码器
        x_decoded = self.up3(self.up2(self.up1(skips[3], skips[2]), skips[1]), skips[0])
        spatial_raw = F.interpolate(self.spatial_head(x_decoded), size=CONFIG['INPUT_SHAPE'], mode='bilinear', align_corners=True)
        if self.use_physics_attn:
            hv_combined = torch.stack(hard_valids, dim=0).max(dim=0)[0]
            spatial_logits = (spatial_raw - 5.0 * (1.0 - hv_combined)).clamp(-20.0, 20.0)
        else:
            spatial_logits = spatial_raw.clamp(-20.0, 20.0)

        # 如果没有物理注意力，返回一个dummy用于loss计算
        if physics_attn is None:
            physics_attn = torch.zeros(B, 1, CONFIG['INPUT_SHAPE'][0], CONFIG['INPUT_SHAPE'][1], device=x.device)

        return cls_logit, self.aux_head(self.aux_pool(x_decoded)).squeeze(-1), spatial_logits, physics_attn

# ================= 9. 损失（对齐V8.3: α=0.70） =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.70, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing
    def forward(self, logits, targets):
        targets = targets * (1 - self.ls) + 0.5 * self.ls if self.ls > 0 else targets
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return ((self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()

class V8Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.70, gamma=2.0, label_smoothing=CONFIG['LABEL_SMOOTHING'])
    def forward(self, cls_logit, aux_logit, spatial_logits, physics_attn, targets):
        loss_main = self.focal(cls_logit, targets)
        loss_aux = self.focal(aux_logit, targets)
        pos_mask = (targets > 0.5).float()
        if self.use_consist and pos_mask.sum() > 0:
            with torch.amp.autocast('cuda', enabled=False):
                loss_consist = (F.mse_loss(
                    torch.sigmoid(spatial_logits.float()).view(spatial_logits.size(0), -1),
                    physics_attn.detach().float().view(physics_attn.size(0), -1),
                    reduction='none').mean(dim=1) * pos_mask).sum() / (pos_mask.sum() + 1e-7)
        else:
            loss_consist = torch.tensor(0.0, device=targets.device)
        return 0.5 * loss_main + 0.25 * loss_aux + 2.0 * loss_consist, loss_main.item()

# ================= 10. EMA（和满血版完全相同） =================
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.backup[n], p.data = p.data.clone(), self.shadow[n]
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = self.backup[n]

# ================= 11. 评估 =================
def evaluate_metrics(probs, labels, mask, view_name):
    if mask.sum() == 0: return 0
    p, l = probs[mask], labels[mask]
    best_csi, best_thresh, best_tp, best_fp, best_fn = 0, 0.5, 0, 0, 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (p > thresh).astype(int)
        tp = ((preds==1)&(l==1)).sum()
        fp = ((preds==1)&(l==0)).sum()
        fn = ((preds==0)&(l==1)).sum()
        csi = tp / (tp + fn + fp + 1e-7)
        if csi > best_csi:
            best_csi, best_thresh = csi, thresh
            best_tp, best_fp, best_fn = tp, fp, fn
    pod = best_tp / (best_tp + best_fn + 1e-7)
    far = best_fp / (best_tp + best_fp + 1e-7)
    logger.info(f"  👉 [{view_name}] CSI: {best_csi:.4f} @Thresh={best_thresh:.2f} | POD: {pod:.3f} | FAR: {far:.3f} | TP={best_tp} FP={best_fp}")
    return best_csi

# ================= 12. 主循环（对齐满血版的lr调度+EMA+梯度累积） =================
def main():
    # 锁定全局种子，确保每个消融版本面临的数据增强和初始化100%一致
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 既然为了绝对复现，建议把 benchmark 关掉，开启 deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TorNetDatasetAblation(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='train')
    test_ds = TorNetDatasetAblation(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='test')

    # ← V8.3: sampler权重 TOR=7.0
    sample_weights = np.zeros(len(train_ds))
    sample_weights[train_ds.cat_ids == 2] = 7.0
    sample_weights[train_ds.cat_ids == 1] = 3.0
    sample_weights[train_ds.cat_ids == 0] = 1.0
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler,
                              num_workers=CONFIG['NUM_WORKERS'], pin_memory=True,
                              prefetch_factor=CONFIG['PREFETCH_FACTOR'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False,
                             num_workers=2, pin_memory=True)

    model = PI_STDNet_Ablation().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])

    # 损失：需要知道是否使用consistency
    criterion = V8Loss().to(device)
    criterion.use_consist = (not args.disable_physics_attn) and (not args.disable_physics_inputs)

    scaler = GradScaler(enabled=CONFIG['USE_AMP'])
    ema = EMA(model)

    best_nc_csi, best_wc_csi = 0.0, 0.0
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  模型参数量: {total_params/1e6:.2f}M")

    for epoch in range(CONFIG['NUM_EPOCHS']):
        # ← 和满血版相同的lr调度
        if epoch < CONFIG['WARMUP_EPOCHS']:
            lr = CONFIG['LEARNING_RATE'] * (epoch + 1) / CONFIG['WARMUP_EPOCHS']
        else:
            lr = 1e-6 + 0.5 * (CONFIG['LEARNING_RATE'] - 1e-6) * \
                 (1 + math.cos(math.pi * (epoch - CONFIG['WARMUP_EPOCHS']) / (CONFIG['NUM_EPOCHS'] - CONFIG['WARMUP_EPOCHS'])))
        for pg in optimizer.param_groups: pg['lr'] = lr

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{args.exp_name}] Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")

        for step, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                loss, _ = criterion(*model(images), labels)
                loss = loss / CONFIG['ACCUM_STEPS']
            scaler.scale(loss).backward()
            if (step + 1) % CONFIG['ACCUM_STEPS'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); ema.update()
            train_loss += loss.item() * CONFIG['ACCUM_STEPS']
            pbar.set_postfix({'loss': f"{loss.item()*CONFIG['ACCUM_STEPS']:.4f}"})

        # 验证
        ema.apply_shadow(); model.eval()
        all_probs, all_labels, all_cats = [], [], []
        with torch.no_grad():
            for images, labels, cats in tqdm(test_loader, desc="Testing"):
                with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                    cls_logit, aux_logit, _, _ = model(images.to(device, non_blocking=True))
                all_probs.extend((0.6*torch.sigmoid(cls_logit) + 0.4*torch.sigmoid(aux_logit)).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_cats.extend(cats.numpy())
        ema.restore()

        all_probs, all_labels, all_cats = np.array(all_probs), np.array(all_labels), np.array(all_cats)
        logger.info(f"\n======== [{args.exp_name}] Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | LR: {lr:.2e} ========")

        mask_nc = np.ones_like(all_labels, dtype=bool)
        nc_csi = evaluate_metrics(all_probs, all_labels, mask_nc, "NC")
        mask_wc = (all_cats == 2) | (all_cats == 1)
        wc_csi = evaluate_metrics(all_probs, all_labels, mask_wc, "WC")

        if wc_csi > best_wc_csi:
            best_wc_csi, best_nc_csi = wc_csi, nc_csi
            ema.apply_shadow()
            torch.save({'model_state_dict': model.state_dict(), 'nc_csi': nc_csi, 'wc_csi': wc_csi},
                       f"ablation_{args.exp_name}_best.pth")
            ema.restore()
            logger.info(f"🏆 [{args.exp_name}] 新纪录! NC={best_nc_csi:.4f} WC={best_wc_csi:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 [{args.exp_name}] 最终结果: NC CSI={best_nc_csi:.4f} | WC CSI={best_wc_csi:.4f}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()


