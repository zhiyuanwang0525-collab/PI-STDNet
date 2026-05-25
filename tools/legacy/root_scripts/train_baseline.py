# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
import numpy as np
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ⚠️ 无缝借用你的 SOTA 数据集和评估标准
try:
    from trainfix import CONFIG, TorNetDatasetV8_Full, evaluate_metrics
except ImportError:
    logger.error("请确保 trainfix.py 与本脚本在同一目录！")
    exit(1)

# ================= 1. 严格复刻官方 Baseline 架构 =================

class ReferenceCoordConv2d(nn.Module):
    """复刻官方的坐标卷积：在特征图上拼接 X 和 Y 坐标"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, y_coords, x_coords], dim=1)
        return self.relu(self.conv(x))

class ReferenceVggBlock(nn.Module):
    """复刻官方代码中的 VggBlock (包含多个 CoordConv 和 1个 MaxPool)"""
    def __init__(self, in_channels, out_channels, n_convs, drop_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            current_in = in_channels if i == 0 else out_channels
            self.layers.append(ReferenceCoordConv2d(current_in, out_channels))
            
        self.pool = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        return self.drop(x)

class ReferenceTornadoCNN(nn.Module):
    """严格对应官方的 TornadoLikelihood + TornadoClassifier 的前向传播"""
    def __init__(self, in_channels):
        super().__init__()
        self.blk1 = ReferenceVggBlock(in_channels, 64, n_convs=2)   
        self.blk2 = ReferenceVggBlock(64, 128, n_convs=2)           
        self.blk3 = ReferenceVggBlock(128, 256, n_convs=3)          
        self.blk4 = ReferenceVggBlock(256, 512, n_convs=3)          
        
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1) 
        )

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        
        likelihood_2d = self.head(x) 
        logits_1d = F.max_pool2d(likelihood_2d, kernel_size=likelihood_2d.size()[2:]) 
        
        return logits_1d.view(-1) 

# ================= 2. 主训练循环 =================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 启动官方 VGG-CNN Baseline 满血加速训练 | 设备: {device}")

    # 1. 挂载数据 (启用 i9-14900K 狂暴并发读取模式)
    train_ds = TorNetDatasetV8_Full(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='train')
    test_ds = TorNetDatasetV8_Full(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='test')

    sample_weights = np.zeros(len(train_ds))
    sample_weights[train_ds.cat_ids == 2] = 7.0
    sample_weights[train_ds.cat_ids == 1] = 3.0
    sample_weights[train_ds.cat_ids == 0] = 1.0
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 🏎️ 提速核心：多线程 + 锁页内存 + 预取
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler, 
        num_workers=12, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
        num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )

    # 2. 模型、优化器与 AMP 缩放器
    in_channels = CONFIG['N_FRAMES'] * CONFIG['CHANNELS_PER_FRAME'] * CONFIG['N_SWEEPS']
    model = ReferenceTornadoCNN(in_channels=in_channels).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss() 
    scaler = GradScaler() # 🏎️ 提速核心：混合精度防溢出

    epochs = 15 
    best_wc_csi = 0.0

    # 3. 训练与评估
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels, _ in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # 🏎️ 提速核心：开启 4070S Tensor Core 混合精度
            with torch.amp.autocast('cuda'):
                logits = model(images)
                smoothed_labels = labels * 0.8 + 0.1
                loss = criterion(logits, smoothed_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 评估测试集 (同理开启 AMP 极速验证)
        model.eval()
        all_probs, all_labels, all_cats = [], [], []
        with torch.no_grad():
            for images, labels, cats in tqdm(test_loader, desc="Validating Baseline"):
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_cats.extend(cats.numpy())

        all_probs, all_labels, all_cats = np.array(all_probs), np.array(all_labels), np.array(all_cats)
        
        logger.info(f"\n======== Reference Baseline Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} ========")
        
        mask_nc = np.ones_like(all_labels, dtype=bool)
        nc_csi = evaluate_metrics(all_probs, all_labels, mask_nc, "CNN Baseline - 全量视图 NC")
        
        mask_wc = (all_cats == 2) | (all_cats == 1)
        wc_csi = evaluate_metrics(all_probs, all_labels, mask_wc, "CNN Baseline - 困难视图 WC")

        if wc_csi > best_wc_csi:
            best_wc_csi = wc_csi
            torch.save(model.state_dict(), "best_reference_cnn_baseline.pth")
            logger.info(f"🟢 Baseline 新高: {best_wc_csi:.4f} (可以直接写入对比大表！)")

if __name__ == '__main__':
    # 避免 Windows 下多进程报错
    import multiprocessing
    multiprocessing.freeze_support()
    main()


