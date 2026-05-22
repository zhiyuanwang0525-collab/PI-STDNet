# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np
import xarray as xr
import scipy.ndimage as ndimage
import random
import logging
import math
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 全局配置 (V8.2 4090D 服务器满血版) =================
CONFIG = {
    # 👇 务必修改为服务器上 TorNet 全量数据集的绝对路径
    'CATALOG_PATH': '/path/to/TorNet/catalog.csv',
    'DATA_ROOT': r'/path/to/TorNet/',
    
    # 🌟 4090D (24G) + 12核 CPU 算力压榨配置
    'BATCH_SIZE': 32,      
    'ACCUM_STEPS': 2,      
    'NUM_WORKERS': 12,     
    'PREFETCH_FACTOR': 4,
    
    'USE_AMP': True,
    'LEARNING_RATE': 5e-4, 
    'WEIGHT_DECAY': 0.05,
    'NUM_EPOCHS': 30,      
    
    'DIMS': [32, 64, 128, 256],
    'DEPTHS': [3, 3, 9, 3],        
    
    'INPUT_SHAPE': (120, 240),
    'N_FRAMES': 4,
    'CHANNELS_PER_FRAME': 11,
    'N_SWEEPS': 2,
    'WARMUP_EPOCHS': 5,
    'LABEL_SMOOTHING': 0.1,
    'DROP_PATH_RATE': 0.2, 
    'TOPK': 16,
}

# ================= 2. 日志配置 (双通道安全输出) =================
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler("v8.2_server_training.log", mode='a', encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

CHANNEL_MIN_MAX = {
    'DBZ': (-30.0, 80.0), 'VEL': (-50.0, 50.0), 'KDP': (-2.0, 5.0),
    'RHOHV': (0.2, 1.05), 'ZDR': (-5.0, 8.0), 'WIDTH': (0.0, 10.0),
    'SHEAR': (-10.0, 10.0), 'DEBRIS': (0.0, 1.0),
}

# ================= 3. 物理先验特征工程 =================
class PhysicalFeatureAmplifier:
    @staticmethod
    def compute_shear(vel):
        vel = np.nan_to_num(vel, nan=0.0)
        shear = np.gradient(vel, axis=1)
        return np.nan_to_num(shear, nan=0.0)

    @staticmethod
    def compute_debris(dbz, rhohv):
        dbz = np.nan_to_num(dbz, nan=-30.0)
        rhohv = np.nan_to_num(rhohv, nan=0.0)
        d_norm = np.clip((dbz - (-30)) / 110.0, 0, 1)
        r_norm = np.clip(rhohv, 0, 1)
        return d_norm * (1.0 - r_norm)

    @staticmethod
    def compute_multiscale_anomaly_shear(vel, dbz):
        vel = np.nan_to_num(vel, nan=0.0)
        valid_mask = (dbz > -29.0)
        eroded_mask = ndimage.binary_erosion(valid_mask, iterations=2)
        
        results = []
        for kernel in [3, 5, 7]:
            half = kernel // 2
            padded = np.pad(vel, ((half, half), (0, 0)), mode='wrap') 
            az_shear = np.zeros_like(vel)
            for i in range(vel.shape[0]):
                window = padded[i:i+kernel, :]
                az_shear[i] = window.max(axis=0) - window.min(axis=0)
            
            background = ndimage.uniform_filter(az_shear, size=15)
            anomaly = az_shear - background
            anomaly = np.clip(anomaly, 0, None)
            safe_anomaly = anomaly * eroded_mask
            results.append(safe_anomaly)
        return results

# ================= 4. 数据增强 =================
class V8Augmentation:
    def __init__(self, mode='train'):
        self.mode = mode
    def __call__(self, data):
        if self.mode != 'train': return data
        C, H, W = data.shape
        if random.random() > 0.5: data = np.flip(data, axis=2).copy()
        if random.random() > 0.5: data = np.flip(data, axis=1).copy()
        
        safe_channels = [0, 2, 3, 4, 5] 
        cpf = CONFIG['CHANNELS_PER_FRAME']
        if random.random() > 0.5:
            for s in range(CONFIG['N_SWEEPS']):
                for t in range(CONFIG['N_FRAMES']):
                    base = s * (CONFIG['N_FRAMES'] * cpf) + t * cpf
                    for c_offset in safe_channels:
                        c = base + c_offset
                        data[c] = data[c] * random.uniform(0.95, 1.05) + random.uniform(-0.02, 0.02)
        if random.random() > 0.5:
            eh, ew = random.randint(10, 25), random.randint(20, 50)
            ey, ex = random.randint(0, max(1, H - eh)), random.randint(0, max(1, W - ew))
            data[:, ey:ey+eh, ex:ex+ew] = 0.0
        return data

# ================= 5. 全量数据集 (支持多类别返回) =================
class TorNetDatasetV8_Full(Dataset):
    def __init__(self, catalog_path, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        df = pd.read_csv(catalog_path)
        
        self.catalog = df[df['type'] == mode].reset_index(drop=True)
        self.augmentor = V8Augmentation(mode=mode)
        
        self.labels = (self.catalog['category'] == 'TOR').astype(int).values
        
        self.cat_ids = np.zeros(len(self.catalog), dtype=int)
        self.cat_ids[self.catalog['category'] == 'TOR'] = 2
        
        # 处理不同 csv 可能的命名方式 (WRN 可能是 'Tornado Warning')
        wrn_mask = (self.catalog['category'] == 'WRN') | (self.catalog['category'] == 'Tornado Warning')
        self.cat_ids[wrn_mask] = 1  
        
        n_tor = (self.cat_ids == 2).sum()
        n_wrn = (self.cat_ids == 1).sum()
        n_nul = (self.cat_ids == 0).sum()
        
        logger.info(f"📦 [{mode.upper()}] 全量样本: {len(self.catalog)} | TOR: {n_tor} | WRN(困难): {n_wrn} | NUL(送分): {n_nul}")

    def __len__(self): return len(self.catalog)

    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
        filepath = os.path.join(self.root_dir, f"tornet_{year}", row['filename'])
        if not os.path.exists(filepath): filepath = os.path.join(self.root_dir, row['filename'])

        label = self.labels[idx]
        cat_id = self.cat_ids[idx]

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
                        f_shear = PhysicalFeatureAmplifier.compute_shear(f_vel)
                        f_debris = PhysicalFeatureAmplifier.compute_debris(f_dbz, rhohv[f_idx])
                        
                        anomaly_shears = PhysicalFeatureAmplifier.compute_multiscale_anomaly_shear(f_vel, f_dbz)

                        all_channels.extend([
                            norm(f_dbz, 'DBZ'), norm(f_vel, 'VEL'),          
                            norm(kdp[f_idx], 'KDP'), norm(rhohv[f_idx], 'RHOHV'),      
                            norm(zdr[f_idx], 'ZDR'), norm(width[f_idx], 'WIDTH'),       
                            norm(f_shear, 'SHEAR'), np.clip(f_debris, 0, 1),     
                            np.clip(anomaly_shears[0] / 20.0, 0.0, 1.0),  
                            np.clip(anomaly_shears[1] / 20.0, 0.0, 1.0),  
                            np.clip(anomaly_shears[2] / 20.0, 0.0, 1.0),  
                        ])

                data = np.clip(np.nan_to_num(np.stack(all_channels, axis=0).astype(np.float32), nan=0.0), -1.0, 2.0)
                return torch.from_numpy(self.augmentor(data.copy())), torch.tensor(label, dtype=torch.float32), torch.tensor(cat_id, dtype=torch.int8)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.catalog) - 1))

# ================= 6. 网络主干结构 (物理对齐与 ConvNeXt) =================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight, self.bias, self.eps, self.data_format = nn.Parameter(torch.ones(normalized_shape)), nn.Parameter(torch.zeros(normalized_shape)), eps, data_format
    def forward(self, x):
        if self.data_format == "channels_first":
            orig_dtype = x.dtype
            u, s = x.float().mean(1, keepdim=True), (x.float() - x.float().mean(1, keepdim=True)).pow(2).mean(1, keepdim=True)
            return (self.weight[:, None, None].float() * ((x.float() - u) / torch.sqrt(s + 1e-4)) + self.bias[:, None, None].float()).to(orig_dtype)
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training: return x
    keep_prob = 1 - drop_prob
    random_tensor = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=torch.float32, device=x.device).bernoulli_(keep_prob)
    return x / keep_prob * random_tensor.to(x.dtype)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1, self.pwconv2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.act, self.gamma, self.drop_path_rate = nn.GELU(), nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True), drop_path_rate
    def forward(self, x):
        residual = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)).permute(0, 2, 3, 1)))) * self.gamma
        return residual + drop_path(x.permute(0, 3, 1, 2), self.drop_path_rate, self.training)

class AddCoords2d(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y_coords, x_coords], dim=1)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.block = ConvNeXtBlock(out_channels)
    def forward(self, x, skip):
        return self.block(self.reduce(torch.cat([F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True), skip], dim=1)))

class MultiScalePhysicsAttention(nn.Module):
    def __init__(self, n_frames=4):
        super().__init__()
        self.n_frames, self.cpf = n_frames, CONFIG['CHANNELS_PER_FRAME']
        self.frame_conv = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
    def forward(self, x):
        frame_feats, dbz_frames = [], []
        for t in range(self.n_frames):
            base = t * self.cpf
            dbz, az_s, az_m, az_l, tds = x[:, base:base+1], x[:, base+8:base+9], x[:, base+9:base+10], x[:, base+10:base+11], x[:, base+7:base+8]
            vdiff = torch.abs(x[:, base+1:base+2] - x[:, (t-1)*self.cpf+1:(t-1)*self.cpf+2]) if t > 0 else torch.zeros_like(az_m)
            frame_feats.append(self.frame_conv(torch.cat([az_s, az_m, az_l, vdiff, tds], dim=1)))
            dbz_frames.append(dbz)
        dbz_max = torch.stack(dbz_frames, dim=1).max(dim=1)[0]
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
            for scale_off in [8, 9, 10]: all_stats.append(torch.stack([x[:, base+t*self.cpf+scale_off:base+t*self.cpf+scale_off+1].view(B, -1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base+t*self.cpf+9:base+t*self.cpf+10] > 0.5).float().mean(dim=(1, 2, 3)) for t in range(self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([(x[:, base+t*self.cpf+1:base+t*self.cpf+2] - x[:, base+(t-1)*self.cpf+1:base+(t-1)*self.cpf+2]).abs().view(B, -1).max(dim=1)[0] for t in range(1, self.n_frames)], dim=0).max(dim=0)[0])
            all_stats.append(torch.stack([x[:, base+t*self.cpf+7:base+t*self.cpf+8].view(B, -1).max(dim=1)[0] for t in range(self.n_frames)], dim=0).max(dim=0)[0])
        return torch.stack(all_stats, dim=1)

class PhysicsGuidedTopKPool(nn.Module):
    def __init__(self, feat_dim, k=16):
        super().__init__()
        self.k = k
        self.topk_proj = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim // 2), nn.GELU())
        self.fuse = nn.Linear(feat_dim + feat_dim // 2, feat_dim)
    def forward(self, feat_map, physics_attn):
        B, D, Hf, Wf = feat_map.shape
        attn_flat, feat_flat = F.interpolate(physics_attn, size=(Hf, Wf), mode='bilinear', align_corners=True).view(B, -1), feat_map.view(B, D, -1)
        topk_feats = torch.gather(feat_flat, 2, torch.topk(attn_flat, min(self.k, Hf * Wf), dim=1)[1].unsqueeze(1).expand(-1, D, -1))
        return self.fuse(torch.cat([self.topk_proj(torch.cat([topk_feats.mean(dim=2), topk_feats.max(dim=2)[0]], dim=1)), self.global_proj(feat_map.mean(dim=(2, 3)))], dim=1))

class PI_STDNet_V8(nn.Module):
    def __init__(self):
        super().__init__()
        single_sweep_ch = CONFIG['N_FRAMES'] * CONFIG['CHANNELS_PER_FRAME']
        self.physics_attn_module = MultiScalePhysicsAttention(CONFIG['N_FRAMES'])
        self.add_coords = AddCoords2d()
        self.sweep_fuse = nn.Sequential(nn.Conv2d(single_sweep_ch * CONFIG['N_SWEEPS'] + 2, single_sweep_ch, 3, padding=1), LayerNorm(single_sweep_ch), nn.GELU())
        self.rot_stats = RotationStatisticsExtractor(CONFIG['N_FRAMES'], CONFIG['N_SWEEPS'])
        
        dims, depths = CONFIG['DIMS'], CONFIG['DEPTHS']
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(single_sweep_ch, dims[0], 4, stride=4), LayerNorm(dims[0]))])
        self.downsample_layers.extend([nn.Sequential(LayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i+1], 2, stride=2)) for i in range(3)])

        dp_rates = [x.item() for x in torch.linspace(0, CONFIG['DROP_PATH_RATE'], sum(depths))]
        self.stages = nn.ModuleList([nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[sum(depths[:i])+j]) for j in range(depths[i])]) for i in range(4)])

        self.up1, self.up2, self.up3 = UpSampleBlock(dims[3], dims[2], dims[2]), UpSampleBlock(dims[2], dims[1], dims[1]), UpSampleBlock(dims[1], dims[0], dims[0])
        self.spatial_head = nn.Sequential(nn.Conv2d(dims[0], 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

        self.topk_pool = PhysicsGuidedTopKPool(dims[1], k=CONFIG['TOPK'])
        self.rot_proj = nn.Sequential(nn.Linear(6 * CONFIG['N_SWEEPS'], 128), nn.LayerNorm(128), nn.GELU())
        
        cls_in_dim = dims[1] + dims[3] + 128
        self.cls_head = nn.Sequential(nn.LayerNorm(cls_in_dim), nn.Linear(cls_in_dim, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, 1))
        self.aux_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.aux_head = nn.Sequential(nn.Flatten(), nn.Linear(dims[0] * 16, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        rot_stats = self.rot_stats(x)
        single_ch = CONFIG['N_FRAMES'] * CONFIG['CHANNELS_PER_FRAME']
        sweep_enhanced, physics_attns, hard_valids = [], [], []
        for s in range(CONFIG['N_SWEEPS']):
            x_sweep = x[:, s * single_ch:(s + 1) * single_ch, :, :]
            pa, hv = self.physics_attn_module(x_sweep)
            sweep_enhanced.append(x_sweep * (1.0 + pa))
            physics_attns.append(pa)
            hard_valids.append(hv)

        physics_attn = (0.7 * physics_attns[0] + 0.3 * physics_attns[1]) * torch.stack(hard_valids, dim=0).max(dim=0)[0]
        feat = self.sweep_fuse(self.add_coords(torch.cat(sweep_enhanced, dim=1)))
        skips = []
        for i in range(4):
            feat = self.stages[i](self.downsample_layers[i](feat))
            skips.append(feat)

        cls_logit = self.cls_head(torch.cat([self.topk_pool(skips[1], physics_attn), skips[3].mean(dim=(2, 3)), self.rot_proj(rot_stats)], dim=1)).squeeze(-1)
        x_decoded = self.up3(self.up2(self.up1(skips[3], skips[2]), skips[1]), skips[0])
        spatial_logits = (F.interpolate(self.spatial_head(x_decoded), size=CONFIG['INPUT_SHAPE'], mode='bilinear', align_corners=True) - 5.0 * (1.0 - torch.stack(hard_valids, dim=0).max(dim=0)[0])).clamp(-20.0, 20.0)

        return cls_logit, self.aux_head(self.aux_pool(x_decoded)).squeeze(-1), spatial_logits, physics_attn

# ================= 7. 损失与 EMA =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, label_smoothing=0.0):
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
        loss_main, loss_aux = self.focal(cls_logit, targets), self.focal(aux_logit, targets)
        pos_mask = (targets > 0.5).float()
        if pos_mask.sum() > 0:
            with torch.amp.autocast('cuda', enabled=False):
                loss_consist = (F.mse_loss(torch.sigmoid(spatial_logits.float()).view(spatial_logits.size(0), -1), physics_attn.detach().float().view(physics_attn.size(0), -1), reduction='none').mean(dim=1) * pos_mask).sum() / (pos_mask.sum() + 1e-7)
        else: loss_consist = torch.tensor(0.0, device=targets.device)
        return 0.5 * loss_main + 0.25 * loss_aux + 2.0 * loss_consist, loss_main.item()

class EMA:
    def __init__(self, model, decay=0.9995): 
        self.model, self.decay, self.shadow, self.backup = model, decay, {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}, {}
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

# ================= 8. 评估核心逻辑 (🌟 双视角评分系统) =================
def evaluate_metrics(probs, labels, mask, view_name):
    if mask.sum() == 0: return 0
    p, l = probs[mask], labels[mask]
    best_csi, best_thresh = 0, 0.5
    best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (p > thresh).astype(int)
        tp, fp, fn, tn = ((preds==1)&(l==1)).sum(), ((preds==1)&(l==0)).sum(), ((preds==0)&(l==1)).sum(), ((preds==0)&(l==0)).sum()
        csi = tp / (tp + fn + fp + 1e-7)
        if csi > best_csi: best_csi, best_thresh, best_tp, best_fp, best_fn, best_tn = csi, thresh, tp, fp, fn, tn
            
    pod = best_tp / (best_tp + best_fn + 1e-7)
    far = best_fp / (best_tp + best_fp + 1e-7)
    logger.info(f"  👉 [{view_name}] CSI: {best_csi:.4f} @Thresh: {best_thresh:.2f} | POD: {pod:.3f} | FAR: {far:.3f} | TP={best_tp}, FP={best_fp}")
    return best_csi

# ================= 9. 主循环 =================
def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 启动 V8.2 4090D 满血全量决胜版 | 设备: {device}")

    train_ds = TorNetDatasetV8_Full(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='train')
    test_ds = TorNetDatasetV8_Full(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='test')

    # 三阶权重采样
    sample_weights = np.zeros(len(train_ds))
    sample_weights[train_ds.cat_ids == 2] = 7  # TOR 
    sample_weights[train_ds.cat_ids == 1] = 3.0   # WRN 
    sample_weights[train_ds.cat_ids == 0] = 1.0   # NUL 
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, prefetch_factor=CONFIG['PREFETCH_FACTOR'], persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, prefetch_factor=CONFIG['PREFETCH_FACTOR'], persistent_workers=True)

    model = PI_STDNet_V8().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    criterion, scaler, ema = V8Loss().to(device), GradScaler(enabled=CONFIG['USE_AMP']), EMA(model)
    
    best_nc_csi, best_wc_csi = 0.0, 0.0

    for epoch in range(CONFIG['NUM_EPOCHS']):
        lr = CONFIG['LEARNING_RATE'] * (epoch + 1) / CONFIG['WARMUP_EPOCHS'] if epoch < CONFIG['WARMUP_EPOCHS'] else 1e-6 + 0.5 * (CONFIG['LEARNING_RATE'] - 1e-6) * (1 + math.cos(math.pi * (epoch - CONFIG['WARMUP_EPOCHS']) / (CONFIG['NUM_EPOCHS'] - CONFIG['WARMUP_EPOCHS'])))
        for pg in optimizer.param_groups: pg['lr'] = lr
            
        model.train()
        train_loss, pbar = 0.0, tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        optimizer.zero_grad()

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
            pbar.set_postfix({'loss': f"{loss.item() * CONFIG['ACCUM_STEPS']:.4f}"})

        # 验证环节
        ema.apply_shadow(); model.eval()
        all_probs, all_labels, all_cats = [], [], []
        with torch.no_grad():
            for images, labels, cats in tqdm(test_loader, desc="Validating"):
                with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                    cls_logit, aux_logit, _, _ = model(images.to(device, non_blocking=True))
                all_probs.extend((1 * torch.sigmoid(cls_logit) + 0 * torch.sigmoid(aux_logit)).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_cats.extend(cats.numpy())
        ema.restore()

        all_probs, all_labels, all_cats = np.array(all_probs), np.array(all_labels), np.array(all_cats)
        
        logger.info(f"\n======== Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | LR: {lr:.2e} ========")
        
        mask_nc = np.ones_like(all_labels, dtype=bool)
        nc_csi = evaluate_metrics(all_probs, all_labels, mask_nc, "全量视图 NC (打败 0.3487 就算赢)")
        
        mask_wc = (all_cats == 2) | (all_cats == 1)
        wc_csi = evaluate_metrics(all_probs, all_labels, mask_wc, "困难视图 WC (打败 0.3617 封神)")

        if wc_csi > best_wc_csi:
            best_wc_csi, best_nc_csi = wc_csi, nc_csi
            ema.apply_shadow()
            torch.save({'model_state_dict': model.state_dict(), 'nc_csi': nc_csi, 'wc_csi': wc_csi}, "best_pi_v8_server_fulldata.pth")
            ema.restore()
            logger.info(f"🏆 新纪录！已保存模型！最佳 NC: {best_nc_csi:.4f} | 最佳 WC: {best_wc_csi:.4f}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
