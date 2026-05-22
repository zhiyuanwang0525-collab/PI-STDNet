# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
PI-STDNet 论文补充图生成脚本

生成两张图:
  Fig.1 - 研究动机图: 龙卷 vs 非龙卷强风暴的 DBZ/VEL 对比
  Fig.3 - 多尺度方位切变物理原理图: 3/5/7 gate 切变 + 背景减除效果

用法:
  python generate_remaining_figures.py \
      --model_path best_pi_v8_server_fulldata.pth \
      --catalog_path /path/to/TorNet/catalog.csv \
      --data_root /path/to/TorNet \
      --save_dir paper_figures_v2
"""

import os
import argparse
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pandas as pd
import scipy.ndimage as ndimage
from tqdm import tqdm

from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG, PhysicalFeatureAmplifier


# ========== 色表 (和 extract_paper_figures_v2 保持一致) ==========
def get_dbz_cmap():
    return plt.cm.turbo, mcolors.Normalize(vmin=-30, vmax=80)

def get_vel_cmap():
    return plt.cm.coolwarm, mcolors.Normalize(vmin=-50, vmax=50)

def get_shear_cmap():
    """方位切变色表: 白→黄→橙→红→深红"""
    colors = ['#FFFFFF', '#FFF9C4', '#FFE082', '#FFB300', '#FF6F00', '#D50000', '#880000']
    cmap = mcolors.LinearSegmentedColormap.from_list('shear', colors, N=256)
    return cmap


# ========== 扇形绘图 (复用) ==========
def plot_polar(ax, data_2d, title, cmap, norm, show_cbar=False, cbar_label=''):
    H, W = data_2d.shape
    az = np.linspace(np.radians(-30), np.radians(30), H + 1)
    rng = np.linspace(0, 60, W + 1)
    R, Theta = np.meshgrid(rng, az)
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    
    masked_data = np.ma.masked_invalid(data_2d)
    mesh = ax.pcolormesh(X, Y, masked_data, cmap=cmap, norm=norm, shading='auto')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    
    r_max = 60
    theta_l, theta_r = np.radians(-30), np.radians(30)
    ax.plot([0, r_max*np.sin(theta_l)], [0, r_max*np.cos(theta_l)], 'k-', lw=0.5)
    ax.plot([0, r_max*np.sin(theta_r)], [0, r_max*np.cos(theta_r)], 'k-', lw=0.5)
    arc = np.linspace(theta_l, theta_r, 100)
    ax.plot(r_max*np.sin(arc), r_max*np.cos(arc), 'k-', lw=0.5)
    ax.axis('off')
    
    if show_cbar:
        cb = plt.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return mesh


# ================================================================
# Fig.1: 研究动机图
# 选一个 TP (龙卷) 和一个 TN_wrn (非龙卷强风暴)
# 并排展示 DBZ + VEL，说明反射率相似但旋转特征不同
# ================================================================
def generate_fig1_motivation(args):
    print("\n" + "=" * 60)
    print("🎨 生成 Fig.1: 研究动机对比图")
    print("=" * 60)
    
    test_ds = TorNetDatasetV8_Full(args.catalog_path, args.data_root, mode='test')
    
    # 加载模型用于筛选高置信样本
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PI_STDNet_V8().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    tor_case = None  # 龙卷样本
    wrn_case = None  # 非龙卷强风暴
    
    print("  搜索最佳对比 case...")
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Fig.1 scan"):
            if tor_case is not None and wrn_case is not None:
                break
            
            image, label, cat = test_ds[i]
            label_val = label.item()
            cat_val = cat.item()
            
            # 模型预测
            img_batch = image.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=True):
                cls_logit, aux_logit, _, _ = model(img_batch)
            prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).item()
            
            # 读取原始数据
            row = test_ds.catalog.iloc[i]
            year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
            nc_path = os.path.join(args.data_root, f"tornet_{year}", row['filename'])
            if not os.path.exists(nc_path):
                nc_path = os.path.join(args.data_root, row['filename'])
            
            try:
                with xr.open_dataset(nc_path, engine='netcdf4', cache=False) as ds:
                    dbz = ds['DBZ'].values
                    vel = ds['VEL'].values
                    t_idx = min(3, dbz.shape[0] - 1)
                    dbz_05 = np.nan_to_num(dbz[t_idx, :, :, 0], nan=-30.0)
                    vel_05 = np.nan_to_num(vel[t_idx, :, :, 0], nan=0.0)
            except:
                continue
            
            # 选高置信 TP (龙卷, prob > 0.8)
            if tor_case is None and label_val == 1 and prob > 0.8:
                # 要求有明显的高反射率 + 强速度对偶（VEL max - min 大）
                vel_range = vel_05.max() - vel_05.min()
                if dbz_05.max() > 50 and vel_range > 40:
                    tor_case = {
                        'dbz': dbz_05, 'vel': vel_05, 'prob': prob,
                        'vel_range': vel_range,
                        'file': os.path.basename(row['filename'])
                    }
                    print(f"  ✅ TOR case: P={prob:.3f} | VEL range={vel_range:.1f} | {row['filename']}")
            
            # 选高反射率但非龙卷的 WRN (prob < 0.5, 正确拒绝)
            if wrn_case is None and label_val == 0 and cat_val == 1 and prob < 0.5:
                if dbz_05.max() > 50:  # 同样要求高反射率
                    wrn_case = {
                        'dbz': dbz_05, 'vel': vel_05, 'prob': prob,
                        'file': os.path.basename(row['filename'])
                    }
                    print(f"  ✅ WRN case: P={prob:.3f} | {row['filename']}")
    
    if tor_case is None or wrn_case is None:
        print("  ⚠️ 未找到合适的对比样本")
        return
    
    # 绘制 Fig.1: 2行 × 2列 (DBZ, VEL) × (TOR, WRN)
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.25)
    
    dbz_cmap, dbz_norm = get_dbz_cmap()
    vel_cmap, vel_norm = get_vel_cmap()
    
    # Row 1: 龙卷样本
    ax_dbz_tor = fig.add_subplot(gs[0, 0])
    plot_polar(ax_dbz_tor, tor_case['dbz'], f'(a) Tornadic Storm — DBZ', dbz_cmap, dbz_norm)
    
    ax_vel_tor = fig.add_subplot(gs[0, 1])
    m_vel = plot_polar(ax_vel_tor, tor_case['vel'], f'(b) Tornadic Storm — VEL', vel_cmap, vel_norm)
    
    # 在VEL图上标注速度对偶区域
    ax_vel_tor.text(0.5, 0.02, f'Confirmed Tornado (P={tor_case["prob"]:.3f})',
                    transform=ax_vel_tor.transAxes, fontsize=10,
                    ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#2E7D32', alpha=0.8))
    
    # Row 2: 非龙卷强风暴
    ax_dbz_wrn = fig.add_subplot(gs[1, 0])
    m_dbz = plot_polar(ax_dbz_wrn, wrn_case['dbz'], f'(c) Non-Tornadic Storm — DBZ', dbz_cmap, dbz_norm)
    
    ax_vel_wrn = fig.add_subplot(gs[1, 1])
    plot_polar(ax_vel_wrn, wrn_case['vel'], f'(d) Non-Tornadic Storm — VEL', vel_cmap, vel_norm)
    
    ax_vel_wrn.text(0.5, 0.02, f'Warning, No Tornado (P={wrn_case["prob"]:.3f})',
                    transform=ax_vel_wrn.transAxes, fontsize=10,
                    ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#C62828', alpha=0.8))
    
    # 右侧色条
    cbar_ax1 = fig.add_subplot(gs[0, 2])
    plt.colorbar(plt.cm.ScalarMappable(norm=dbz_norm, cmap=dbz_cmap), cax=cbar_ax1)
    cbar_ax1.set_ylabel('dBZ', fontsize=11)
    
    cbar_ax2 = fig.add_subplot(gs[1, 2])
    plt.colorbar(plt.cm.ScalarMappable(norm=vel_norm, cmap=vel_cmap), cax=cbar_ax2)
    cbar_ax2.set_ylabel('m/s', fontsize=11)
    
    save_path = os.path.join(args.save_dir, "Fig1_motivation.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Fig.1 saved: {save_path}")


# ================================================================
# Fig.3: 多尺度方位切变物理原理图
# 展示同一个 VEL 场经过 3/5/7 gate 提取后的切变图
# 以及背景减除前后的对比
# ================================================================
def generate_fig3_shear(args):
    print("\n" + "=" * 60)
    print("🎨 生成 Fig.3: 多尺度方位切变物理原理图")
    print("=" * 60)
    
    test_ds = TorNetDatasetV8_Full(args.catalog_path, args.data_root, mode='test')
    
    # 找一个有明显旋转的龙卷样本
    print("  搜索适合展示的龙卷样本...")
    best_case = None
    
    for i in tqdm(range(min(len(test_ds), 5000)), desc="Fig.3 scan"):
        image, label, cat = test_ds[i]
        if label.item() != 1:
            continue
        
        row = test_ds.catalog.iloc[i]
        year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
        nc_path = os.path.join(args.data_root, f"tornet_{year}", row['filename'])
        if not os.path.exists(nc_path):
            nc_path = os.path.join(args.data_root, row['filename'])
        
        try:
            with xr.open_dataset(nc_path, engine='netcdf4', cache=False) as ds:
                vel_raw = ds['VEL'].values
                dbz_raw = ds['DBZ'].values
                t_idx = min(3, vel_raw.shape[0] - 1)
                vel = np.nan_to_num(vel_raw[t_idx, :, :, 0], nan=0.0)
                dbz = np.nan_to_num(dbz_raw[t_idx, :, :, 0], nan=-30.0)
        except:
            continue
        
        # 计算各尺度切变，选切变最大值最高的样本
        anomaly_shears = PhysicalFeatureAmplifier.compute_multiscale_anomaly_shear(vel, dbz)
        max_shear = max(a.max() for a in anomaly_shears)
        
        if best_case is None or max_shear > best_case['max_shear']:
            # 同时计算原始 (非背景减除) 的方位切变用于对比
            raw_shears = []
            eroded_mask = ndimage.binary_erosion((dbz > -29.0), iterations=2)
            for kernel in [3, 5, 7]:
                half = kernel // 2
                padded = np.pad(vel, ((half, half), (0, 0)), mode='wrap')
                az_shear = np.zeros_like(vel)
                for r in range(vel.shape[0]):
                    window = padded[r:r+kernel, :]
                    az_shear[r] = window.max(axis=0) - window.min(axis=0)
                raw_shears.append(az_shear * eroded_mask)
            
            best_case = {
                'vel': vel, 'dbz': dbz,
                'raw_shears': raw_shears,
                'anomaly_shears': anomaly_shears,
                'max_shear': max_shear,
                'file': os.path.basename(row['filename'])
            }
    
    if best_case is None:
        print("  ⚠️ 未找到合适样本")
        return
    
    print(f"  选中样本: {best_case['file']} (max anomaly shear = {best_case['max_shear']:.2f})")
    
    # ====== 绘制 Fig.3 ======
    # 布局: 主图2行×4列 + 中间箭头行 + 底部色条行
    
    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(4, 4, height_ratios=[1, 0.08, 1, 0.06], wspace=0.12, hspace=0.08)
    
    vel_cmap, vel_norm = get_vel_cmap()
    shear_cmap = get_shear_cmap()
    
    # 确定切变色表范围
    all_raw = np.concatenate([s.flatten() for s in best_case['raw_shears']])
    all_anom = np.concatenate([s.flatten() for s in best_case['anomaly_shears']])
    raw_vmax = np.percentile(all_raw[all_raw > 0], 98) if (all_raw > 0).any() else 20
    anom_vmax = np.percentile(all_anom[all_anom > 0], 98) if (all_anom > 0).any() else 15
    
    raw_norm = mcolors.Normalize(vmin=0, vmax=raw_vmax)
    anom_norm = mcolors.Normalize(vmin=0, vmax=anom_vmax)
    
    kernel_names = ['k=3 (Small-scale)', 'k=5 (TVS-scale)', 'k=7 (Meso-scale)']
    
    # ----- Row 0: VEL + Raw shears -----
    ax_vel = fig.add_subplot(gs[0, 0])
    plot_polar(ax_vel, best_case['vel'], '(a) Radial Velocity', vel_cmap, vel_norm)
    
    for j in range(3):
        ax = fig.add_subplot(gs[0, j+1])
        shear_display = best_case['raw_shears'][j].copy()
        shear_display[shear_display == 0] = np.nan
        plot_polar(ax, shear_display, f'({"bcd"[j]}) Raw AzShear {kernel_names[j]}',
                   shear_cmap, raw_norm)
    
    # ----- Row 1: 箭头标注行 -----
    for j in range(1, 4):
        ax_arrow = fig.add_subplot(gs[1, j])
        ax_arrow.axis('off')
        ax_arrow.annotate('', xy=(0.5, 0.0), xytext=(0.5, 1.0),
                         arrowprops=dict(arrowstyle='->', color='#D50000', lw=2.5),
                         transform=ax_arrow.transAxes)
    
    # 中间行左侧放标注文字
    ax_label = fig.add_subplot(gs[1, 0])
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, 'Background\nSubtraction', fontsize=14, fontweight='bold',
                  ha='center', va='center', color='#D50000',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', 
                           edgecolor='#D50000', alpha=0.9))
    
    # ----- Row 2: DBZ + Anomaly shears -----
    ax_dbz = fig.add_subplot(gs[2, 0])
    dbz_cmap, dbz_norm_cm = get_dbz_cmap()
    plot_polar(ax_dbz, best_case['dbz'], '(e) Reflectivity (DBZ)', dbz_cmap, dbz_norm_cm)
    
    for j in range(3):
        ax = fig.add_subplot(gs[2, j+1])
        anom_display = best_case['anomaly_shears'][j].copy()
        anom_display[anom_display == 0] = np.nan
        plot_polar(ax, anom_display, f'({"fgh"[j]}) Anomaly AzShear {kernel_names[j]}',
                   shear_cmap, anom_norm)
    
    # ----- Row 3: 色条 (按列对应关系排列) -----
    # Col 0: VEL + DBZ 共用 → 放两个小色条
    cbar_ax0 = fig.add_subplot(gs[3, 0])
    cb0 = plt.colorbar(plt.cm.ScalarMappable(norm=vel_norm, cmap=vel_cmap),
                       cax=cbar_ax0, orientation='horizontal')
    cb0.set_label('Velocity (m/s) / Reflectivity (dBZ)', fontsize=10)
    cb0.ax.tick_params(labelsize=8)
    
    # Col 1: Raw Shear 色条
    cbar_ax1 = fig.add_subplot(gs[3, 1])
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=raw_norm, cmap=shear_cmap),
                       cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('Raw Azimuthal Shear (m/s)', fontsize=10)
    cb1.ax.tick_params(labelsize=8)
    
    # Col 2-3: Anomaly Shear 色条 (跨两列)
    cbar_ax23 = fig.add_subplot(gs[3, 2:4])
    cb23 = plt.colorbar(plt.cm.ScalarMappable(norm=anom_norm, cmap=shear_cmap),
                        cax=cbar_ax23, orientation='horizontal')
    cb23.set_label('Anomaly Azimuthal Shear (m/s)', fontsize=10)
    cb23.ax.tick_params(labelsize=8)
    
    save_path = os.path.join(args.save_dir, "Fig3_multiscale_shear.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Fig.3 saved: {save_path}")


# ================================================================
# Table V: 计算效率统计 (不需要训练)
# ================================================================
def compute_efficiency(args):
    print("\n" + "=" * 60)
    print("📊 计算模型效率统计 (Table V)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PI_STDNet_V8().to(device)
    model.eval()
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total Parameters:     {total_params:>12,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
    
    # 推理时间
    dummy_input = torch.randn(1, 88, 120, 240).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    import time
    times = []
    with torch.no_grad():
        for _ in range(100):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Inference Time:       {mean_time:.2f} ± {std_time:.2f} ms/sample")
    print(f"  Throughput:           {1000/mean_time:.1f} samples/sec")
    
    # 尝试计算 FLOPs (需要 thop 或 fvcore)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"  FLOPs:                {flops/1e9:.2f} G")
    except ImportError:
        print("  FLOPs:                需要安装 thop (pip install thop)")
    
    # 保存结果
    result_path = os.path.join(args.save_dir, "efficiency_stats.txt")
    with open(result_path, 'w') as f:
        f.write(f"Parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Inference: {mean_time:.2f} ± {std_time:.2f} ms\n")
        f.write(f"Throughput: {1000/mean_time:.1f} samples/sec\n")
    print(f"  ✅ 统计结果保存: {result_path}")


# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_pi_v8_GWP.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures1,3')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    generate_fig1_motivation(args)
    generate_fig3_shear(args)
    compute_efficiency(args)
