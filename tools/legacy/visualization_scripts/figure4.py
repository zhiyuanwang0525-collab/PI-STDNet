# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
PI-STDNet V8.2 论文可视化图生成脚本 (TGRS 顶刊终极版)

核心升级点:
1. 采用无衬线学术字体 (Arial/Helvetica)，字体加粗且加大，满足高分辨率印刷要求。
2. 极其紧凑的排版布局 (GridSpec wspace=0.03)，消除视觉缝隙。
3. 概率、类别和注意力最大值采用带背板的“仪表盘”风格设计，提升科技感。
4. 底部色条采用极简的内向刻度线设计，避免喧宾夺主。
5. 统一输出 PDF 矢量图格式，保证放入 LaTeX 时无限放大不失真。

用法:
  python grab3_2.py \
      --model_path best_pi_v8_server_fulldata.pth \
      --catalog_path /path/to/TorNet/catalog.csv \
      --data_root /path/to/TorNet/ \
      --save_dir paper_figures_final \
      --max_per_type 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm

# ========== 顶刊学术排版全局配置 ==========
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.5,      # 边框加粗
    'axes.titlesize': 16,       # 标题字号加大
    'axes.titleweight': 'bold', # 标题加粗
    'figure.dpi': 300,          # 强制高分辨率
})

# ========== 导入模型与数据加载器 ==========
# 请确保你的 train.py 或 train_compare.py 在同一目录下，且包含 PI_STDNet_V8 和相关 Dataset
try:
    from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG
except ImportError:
    print("⚠️ 警告: 无法从 train.py 导入模型，请确保同目录下有该文件。此处为占位。")
    # 如果导入失败，你可以替换为你实际存放模型架构的文件名
    pass


# ========== 色表配置 ==========
def get_dbz_cmap():
    """TorNet 论文风格的 DBZ 色表"""
    cmap = plt.cm.turbo
    norm = mcolors.Normalize(vmin=-30, vmax=80)
    return cmap, norm

def get_vel_cmap():
    """TorNet 论文风格的 VEL 色表 (红蓝冷暖)"""
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=-50, vmax=50)
    return cmap, norm

def get_attn_cmap_focused(data, top_percent=10):
    """聚焦式注意力可视化: 气象常用暖色调"""
    colors = ['#FFFFFF', '#FFF5CC', '#FFE066', '#FFB300', '#FF6F00', '#D50000', '#880000']
    cmap = mcolors.LinearSegmentedColormap.from_list('attn_met', colors, N=256)
    
    valid = data[data > 0]
    if len(valid) > 0:
        threshold = np.percentile(valid, 100 - top_percent)
        vmin = threshold
        vmax = valid.max()
    else:
        vmin, vmax = 0, 0.1
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def mask_attention_top(data, top_percent=15):
    """将注意力图中低于阈值的区域设为 NaN (透明)"""
    valid = data[data > 0]
    if len(valid) == 0:
        return np.full_like(data, np.nan)
    threshold = np.percentile(valid, 100 - top_percent)
    masked = data.copy()
    masked[data < threshold] = np.nan
    return masked


# ========== 雷达扇形绘图基础函数 ==========
def plot_polar_channel(ax, data_2d, title, cmap, norm, show_cbar=False, cbar_label=''):
    H, W = data_2d.shape
    az = np.linspace(np.radians(-30), np.radians(30), H + 1)
    rng = np.linspace(0, 60, W + 1)
    R, Theta = np.meshgrid(rng, az)
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    
    masked_data = np.ma.masked_invalid(data_2d)
    mesh = ax.pcolormesh(X, Y, masked_data, cmap=cmap, norm=norm, shading='auto')
    
    if title:
        ax.set_title(title, pad=12)
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    
    # 绘制扇形黑边框
    r_max = 60
    theta_left, theta_right = np.radians(-30), np.radians(30)
    ax.plot([0, r_max * np.sin(theta_left)], [0, r_max * np.cos(theta_left)], 'k-', lw=0.8)
    ax.plot([0, r_max * np.sin(theta_right)], [0, r_max * np.cos(theta_right)], 'k-', lw=0.8)
    arc_theta = np.linspace(theta_left, theta_right, 100)
    ax.plot(r_max * np.sin(arc_theta), r_max * np.cos(arc_theta), 'k-', lw=0.8)
    ax.axis('off')
    
    if show_cbar:
        cb = plt.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
    
    return mesh


# ========== EF Rating 兼容提取 ==========
def get_ef_rating(catalog_df, idx):
    ef_candidates = ['ef_number', 'ef_rating', 'EF', 'ef', 'mag', 'ef_scale', 'EF_RATING', 'tornado_rating']
    for col in ef_candidates:
        if col in catalog_df.columns:
            val = catalog_df.iloc[idx][col]
            if pd.notna(val):
                val_str = str(val).strip().upper().replace('EF', '').replace('-', '').replace(' ', '')
                try:
                    return int(float(val_str))
                except (ValueError, TypeError):
                    pass
    return -1


# ========== 主执行流程 ==========
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    print("📦 正在加载 PI-STDNet_V8 模型...")
    model = PI_STDNet_V8().to(device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f"⚠️ 找不到权重 {args.model_path}，将使用随机权重测试绘图。")
    model.eval()
    
    # 加载数据
    print("📦 加载测试集数据...")
    if not os.path.exists(args.catalog_path):
        print(f"❌ 找不到 Catalog 文件: {args.catalog_path}")
        return
        
    test_ds = TorNetDatasetV8_Full(args.catalog_path, args.data_root, mode='test')
    df_full = pd.read_csv(args.catalog_path)
    test_catalog = df_full[df_full['type'] == 'test'].reset_index(drop=True)
    
    collectors = {
        'TP_strong': [], 'TP_weak': [], 'TN_wrn': [], 'FP': [], 'FN': []
    }
    
    print(f"\n🔍 全速扫描测试集 (共 {len(test_ds)} 样本)...")
    with torch.no_grad():
        for i in tqdm(range(min(len(test_ds), 2000)), desc="Scanning"): # 设定个扫描上限防卡死
            all_done = all(len(v) >= args.max_per_type for v in collectors.values())
            if all_done:
                break
            
            image, label_tensor, cat_tensor = test_ds[i]
            label = label_tensor.item()
            cat_id = cat_tensor.item()
            
            img_batch = image.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=True):
                # 请根据你的模型实际返回的四个值进行修改
                cls_logit, aux_logit, spatial_logits, physics_attn = model(img_batch)
            
            prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).item()
            ef_rating = get_ef_rating(test_catalog, i) if i < len(test_catalog) else -1
            
            optimal_thresh = 0.60  # 你论文里算出的最佳阈值
            pred = 1 if prob >= optimal_thresh else 0
            
            case_type = None
            if pred == 1 and label == 1:
                if prob > 0.85: case_type = 'TP_strong'
                else: case_type = 'TP_weak'
            elif pred == 0 and label == 0 and cat_id == 1:
                if 0.15 < prob < 0.55: case_type = 'TN_wrn'
            elif pred == 1 and label == 0:
                case_type = 'FP'
            elif pred == 0 and label == 1:
                case_type = 'FN'
            
            if case_type is None or len(collectors[case_type]) >= args.max_per_type:
                continue
            
            # 提取与还原特征
            pa_full = F.interpolate(physics_attn, size=CONFIG['INPUT_SHAPE'], mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            sp_full = torch.sigmoid(spatial_logits[0, 0]).cpu().numpy()
            
            # 加载原始 nc 文件中的物理数据
            row = test_ds.catalog.iloc[i]
            year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
            nc_path = os.path.join(args.data_root, f"tornet_{year}", row['filename'])
            if not os.path.exists(nc_path):
                nc_path = os.path.join(args.data_root, row['filename'])
            
            try:
                with xr.open_dataset(nc_path, engine='netcdf4', cache=False) as ds:
                    dbz_raw = ds['DBZ'].values
                    vel_raw = ds['VEL'].values
                    t_idx = min(3, dbz_raw.shape[0] - 1)
                    dbz_05 = np.nan_to_num(dbz_raw[t_idx, :, :, 0], nan=-30.0)
                    vel_05 = np.nan_to_num(vel_raw[t_idx, :, :, 0], nan=0.0)
                    vel_09 = np.nan_to_num(vel_raw[t_idx, :, :, 1], nan=0.0) if vel_raw.shape[-1] > 1 else vel_05.copy()
            except Exception as e:
                continue
            
            collectors[case_type].append({
                'idx': i, 'prob': prob, 'ef': ef_rating, 
                'dbz_05': dbz_05, 'vel_05': vel_05, 'vel_09': vel_09,
                'physics_attn': pa_full, 'spatial_pred': sp_full,
            })

    # ====== 生成 Fig.4 组合大图 (TGRS 出版级排版) ======
    print("\n🖼️ 正在渲染 TGRS 审稿人特供版 Fig.4 组合大图...")
    
    row_order = ['TP_strong', 'TP_weak', 'TN_wrn', 'FP', 'FN']
    selected = []
    for ct in row_order:
        cases = collectors[ct]
        if not cases: continue
        if ct == 'TP_strong': selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'TP_weak': selected.append((ct, min(cases, key=lambda x: abs(x['prob'] - 0.55))))
        elif ct == 'TN_wrn':  selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'FP':      selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'FN':      selected.append((ct, max(cases, key=lambda x: x['prob'])))
    
    if len(selected) < 2:
        print("  ⚠️ 收集到的样本太少，无法生成大图，请检查数据或放宽判断条件。")
        return
    
    n_rows = len(selected)
    
    # 【极致紧凑排版】
    fig = plt.figure(figsize=(26, 4.5 * n_rows + 1.5))
    gs = fig.add_gridspec(n_rows + 1, 5, 
                          height_ratios=[*([1] * n_rows), 0.06], 
                          wspace=0.03, hspace=0.05, 
                          left=0.05, right=0.98, top=0.92, bottom=0.05)
    
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(n_rows)])
    cbar_axes = [fig.add_subplot(gs[n_rows, c]) for c in range(5)]
    
    type_labels = {
        'TP_strong': 'TP (Strong Hit)', 'TP_weak': 'TP (Weak Hit)',
        'TN_wrn': 'TN (Correct Rejection)', 'FP': 'FP (False Alarm)', 'FN': 'FN (Missed)'
    }
    type_colors = {
        'TP_strong': '#1B5E20', 'TP_weak': '#689F38',
        'TN_wrn': '#0D47A1', 'FP': '#B71C1C', 'FN': '#E65100'
    }
    
    col_titles = ['DBZ (0.5°)', 'VEL (0.5°)', 'VEL (0.9°)', 'Physics Attention', 'Spatial Prediction']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, pad=12)
    
    for row_idx, (ct, case) in enumerate(selected):
        row_label = f"({chr(97 + row_idx)}) {type_labels[ct]}"
        
        dbz_cmap, dbz_norm = get_dbz_cmap()
        vel_cmap, vel_norm = get_vel_cmap()
        attn_masked = mask_attention_top(case['physics_attn'], top_percent=15)
        attn_cmap, attn_norm = get_attn_cmap_focused(case['physics_attn'], top_percent=15)
        sp_masked = mask_attention_top(case['spatial_pred'], top_percent=15)
        sp_cmap, sp_norm = get_attn_cmap_focused(case['spatial_pred'], top_percent=15)

        plot_polar_channel(axes[row_idx, 0], case['dbz_05'], '', dbz_cmap, dbz_norm)
        plot_polar_channel(axes[row_idx, 1], case['vel_05'], '', vel_cmap, vel_norm)
        plot_polar_channel(axes[row_idx, 2], case['vel_09'], '', vel_cmap, vel_norm)
        plot_polar_channel(axes[row_idx, 3], attn_masked, '', attn_cmap, attn_norm)
        plot_polar_channel(axes[row_idx, 4], sp_masked, '', sp_cmap, sp_norm)
        
        # 左侧行标签
        axes[row_idx, 0].text(
            -0.08, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=15, fontweight='bold',
            color=type_colors.get(ct, 'black'),
            va='center', ha='right', rotation=90,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
        )
        
        # 仪表盘概率徽章
        prob_color = type_colors.get(ct, 'gray')
        ef_str = f" | EF{case['ef']}" if case['ef'] >= 0 else ""
        axes[row_idx, 4].text(
            0.95, 0.05, f'P = {case["prob"]:.3f}{ef_str}',
            transform=axes[row_idx, 4].transAxes,
            fontsize=14, fontweight='bold', color='white', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=prob_color, edgecolor='black', linewidth=1.2, alpha=0.95)
        )
        
        # 注意力最大值标注
        axes[row_idx, 3].text(
            0.95, 0.05, f'Max: {case["physics_attn"].max():.2f}',
            transform=axes[row_idx, 3].transAxes,
            fontsize=12, fontweight='bold', color='white', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#424242', edgecolor='none', alpha=0.8)
        )
    
    # ====== 【底部统一极简色条】 ======
    dbz_cmap_cb, dbz_norm_cb = get_dbz_cmap()
    cb0 = plt.colorbar(plt.cm.ScalarMappable(norm=dbz_norm_cb, cmap=dbz_cmap_cb), cax=cbar_axes[0], orientation='horizontal')
    cb0.set_label('Reflectivity (dBZ)', fontsize=13, fontweight='bold', labelpad=5)
    cb0.ax.tick_params(labelsize=11, direction='in', length=5)
    
    vel_cmap_cb, vel_norm_cb = get_vel_cmap()
    for ci in [1, 2]:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=vel_norm_cb, cmap=vel_cmap_cb), cax=cbar_axes[ci], orientation='horizontal')
        cb.set_label('Radial Velocity (m/s)', fontsize=13, fontweight='bold', labelpad=5)
        cb.ax.tick_params(labelsize=11, direction='in', length=5)
    
    attn_cmap_cb = mcolors.LinearSegmentedColormap.from_list('attn_cb', ['#FFFFFF', '#FFF5CC', '#FFE066', '#FFB300', '#FF6F00', '#D50000', '#880000'], N=256)
    cb3 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=attn_cmap_cb), cax=cbar_axes[3], orientation='horizontal')
    cb3.set_label('Physics Attention Level', fontsize=13, fontweight='bold', labelpad=5)
    cb3.ax.tick_params(labelsize=11, direction='in', length=5)
    
    cb4 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=attn_cmap_cb), cax=cbar_axes[4], orientation='horizontal')
    cb4.set_label('Spatial Probability', fontsize=13, fontweight='bold', labelpad=5)
    cb4.ax.tick_params(labelsize=11, direction='in', length=5)
    
# 消除色条之间的额外粗外框，统一设置为 1.0 的细线
    for cax in cbar_axes:
        for spine in cax.spines.values():
            spine.set_linewidth(1.0)
    
    # 强制输出 PDF 以适配顶刊要求
    combo_path = os.path.join(args.save_dir, "Figure4_Microscopic_Performance_Analysis.pdf")
    plt.savefig(combo_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ 完美！出版级大图已保存至: {combo_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ⚠️ 请确保这里的路径与你的实际环境一致！
    parser.add_argument('--model_path', type=str, default='best_pi_v8_GWP.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    parser.add_argument('--max_per_type', type=int, default=5)
    args = parser.parse_args()
    main(args)
