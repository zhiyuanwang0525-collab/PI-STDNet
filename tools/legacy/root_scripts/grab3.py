# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
PI-STDNet V8.2 论文可视化图生成脚本 (修复版 V2)

修复内容:
1. EF rating 检测：兼容 TorNet catalog 中的多种列名格式
2. 注意力图对比度：使用 percentile-based 归一化，突出高激活区域
3. TP_strong 筛选条件放宽：不依赖 EF rating，用概率高+确认龙卷即可
4. 增加列标题和色条

用法:
  python extract_paper_figures_v2.py \
      --model_path best_pi_v8_server_fulldata.pth \
      --save_dir paper_figures_v2 \
      --max_per_type 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm

# ========== 导入 V8.2 模型 ==========
from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG


# ========== 色表 ==========
def get_dbz_cmap():
    """
    TorNet 原始论文风格的 DBZ 色表
    连续渐变: 青蓝 → 绿 → 黄 → 橙 → 红 → 深红/紫
    和项目文件夹中 TP_Hit / FN_Miss 样例图完全一致
    """
    # 精确匹配 TorNet 论文中的色表 (基于 matplotlib turbo)
    cmap = plt.cm.turbo
    norm = mcolors.Normalize(vmin=-30, vmax=80)
    return cmap, norm

def get_vel_cmap():
    """
    TorNet 原始论文风格的 VEL 色表
    标准 diverging: 蓝(靠近) → 白(零) → 红(远离)
    和项目文件夹中样例图一致
    """
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=-50, vmax=50)
    return cmap, norm

def get_attn_cmap_focused(data, top_percent=10):
    """
    聚焦式注意力可视化:
    - 只显示 top N% 的注意力值，其余透明
    - 用气象常用的暖色调 (白-黄-橙-红-深红)
    """
    # 创建气象风格暖色表
    colors = ['#FFFFFF', '#FFF5CC', '#FFE066', '#FFB300', '#FF6F00', '#D50000', '#880000']
    cmap = mcolors.LinearSegmentedColormap.from_list('attn_met', colors, N=256)
    
    # 计算阈值：只有 top_percent 的像素有颜色
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
    """
    将注意力图中低于 top_percent 阈值的区域设为 NaN (透明)
    只保留最高激活的区域用于可视化
    """
    valid = data[data > 0]
    if len(valid) == 0:
        return np.full_like(data, np.nan)
    threshold = np.percentile(valid, 100 - top_percent)
    masked = data.copy()
    masked[data < threshold] = np.nan
    return masked


# ========== 扇形绘图 ==========
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
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    
    # 边框
    r_max = 60
    theta_left, theta_right = np.radians(-30), np.radians(30)
    ax.plot([0, r_max * np.sin(theta_left)], [0, r_max * np.cos(theta_left)], 'k-', lw=0.5)
    ax.plot([0, r_max * np.sin(theta_right)], [0, r_max * np.cos(theta_right)], 'k-', lw=0.5)
    arc_theta = np.linspace(theta_left, theta_right, 100)
    ax.plot(r_max * np.sin(arc_theta), r_max * np.cos(arc_theta), 'k-', lw=0.5)
    ax.axis('off')
    
    if show_cbar:
        cb = plt.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    
    return mesh


# ========== 检测 EF rating 列 ==========
def get_ef_rating(catalog_df, idx):
    """兼容 TorNet 各种可能的 EF rating 列名"""
    # 尝试各种可能的列名
    ef_candidates = ['ef_number', 'ef_rating', 'EF', 'ef', 'mag', 'ef_scale', 'EF_RATING', 'tornado_rating']
    
    for col in ef_candidates:
        if col in catalog_df.columns:
            val = catalog_df.iloc[idx][col]
            if pd.notna(val):
                # 处理字符串格式如 "EF2", "2", "EF-2"
                val_str = str(val).strip().upper().replace('EF', '').replace('-', '').replace(' ', '')
                try:
                    return int(float(val_str))
                except (ValueError, TypeError):
                    pass
    return -1


# ========== 主流程 ==========
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    print("📦 加载模型...")
    model = PI_STDNet_V8().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    # 加载测试集
    print("📦 加载测试集...")
    test_ds = TorNetDatasetV8_Full(args.catalog_path, args.data_root, mode='test')
    
    # 读取 catalog
    df_full = pd.read_csv(args.catalog_path)
    test_catalog = df_full[df_full['type'] == 'test'].reset_index(drop=True)
    
    # 打印 EF 列诊断信息
    print(f"\n📋 Catalog 列名: {list(test_catalog.columns)}")
    for col in ['ef_rating', 'EF', 'ef', 'mag', 'ef_scale']:
        if col in test_catalog.columns:
            print(f"  找到 EF 列: '{col}' | 非空值: {test_catalog[col].notna().sum()} | 示例: {test_catalog[col].dropna().head(3).tolist()}")
    
    # 收集器
    collectors = {
        'TP_strong': [],
        'TP_weak': [],
        'TN_wrn': [],
        'FP': [],
        'FN': [],
    }
    
    print(f"\n🔍 扫描测试集 ({len(test_ds)} 样本)...")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Scanning"):
            all_done = all(len(v) >= args.max_per_type for v in collectors.values())
            if all_done:
                break
            
            image, label_tensor, cat_tensor = test_ds[i]
            label = label_tensor.item()
            cat_id = cat_tensor.item()
            
            img_batch = image.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=True):
                cls_logit, aux_logit, spatial_logits, physics_attn = model(img_batch)
            
            prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).item()
            
            # EF rating
            ef_rating = get_ef_rating(test_catalog, i) if i < len(test_catalog) else -1
            
            optimal_thresh = 0.60  # 🌟 必须写死你的 SOTA 阈值
            pred = 1 if prob >= optimal_thresh else 0
            
            case_type = None
            
            if pred == 1 and label == 1:
                # TP: 按概率高低分 strong/weak
                if prob > 0.85:
                    case_type = 'TP_strong'  # 极其确信的命中
                else:
                    case_type = 'TP_weak'    # 擦边命中 (0.60 ~ 0.85)，展示微弱特征提取能力
            elif pred == 0 and label == 0 and cat_id == 1:
                # TN_wrn: 困难负样本的正确拒绝
                # 找那些低于 0.60，但又不是 0.01 这种太容易的，比如 0.2 ~ 0.5 之间的诱惑样本
                if 0.15 < prob < 0.55:
                    case_type = 'TN_wrn'
            elif pred == 1 and label == 0:
                # FP: 误报 (概率高于 0.60 但实际没龙卷)
                case_type = 'FP'
            elif pred == 0 and label == 1:
                # FN: 漏报 (实际有龙卷，但概率低于 0.60)
                case_type = 'FN'
            
            if case_type is None:
                continue
            if len(collectors[case_type]) >= args.max_per_type:
                continue
            
            # ====== 提取数据 ======
            # 物理注意力 resize 到原始分辨率
            pa_full = F.interpolate(
                physics_attn, size=CONFIG['INPUT_SHAPE'],
                mode='bilinear', align_corners=True
            )[0, 0].cpu().numpy()
            
            sp_full = torch.sigmoid(
                spatial_logits[0, 0]
            ).cpu().numpy()
            
            # 读取原始 nc 数据
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
                    
                    if vel_raw.shape[-1] > 1:
                        vel_09 = np.nan_to_num(vel_raw[t_idx, :, :, 1], nan=0.0)
                    else:
                        vel_09 = vel_05.copy()
                        
            except Exception as e:
                print(f"  ⚠️ 读取失败: {nc_path} ({e})")
                continue
            
            ef_str = f"EF{ef_rating}" if ef_rating >= 0 else ""
            cat_str = ['NUL', 'WRN', 'TOR'][cat_id]
            info = f"P={prob:.3f} | {cat_str} {ef_str} | {os.path.basename(row['filename'])}"
            
            collectors[case_type].append({
                'idx': i, 'prob': prob, 'ef': ef_rating, 'info': info,
                'dbz_05': dbz_05, 'vel_05': vel_05, 'vel_09': vel_09,
                'physics_attn': pa_full, 'spatial_pred': sp_full,
            })
            print(f"  📸 [{case_type}] #{len(collectors[case_type])} | {info}")
    
    # ====== 生成单独的 case 图 ======
    print("\n🎨 生成单独 case 图...")
    for case_type, cases in collectors.items():
        if not cases:
            print(f"  ⚠️ [{case_type}] 无样本")
            continue
        for j, case in enumerate(cases):
            fig, axes = plt.subplots(1, 5, figsize=(30, 5.5))
            
            type_colors = {
                'TP_strong': '#2E7D32', 'TP_weak': '#558B2F',
                'TN_wrn': '#1565C0', 'FP': '#C62828', 'FN': '#E65100'
            }
            
            fig.suptitle(f'[{case_type}] {case["info"]}',
                         fontsize=14, fontweight='bold',
                         color=type_colors.get(case_type, 'black'))
            
            # DBZ
            dbz_cmap, dbz_norm = get_dbz_cmap()
            plot_polar_channel(axes[0], case['dbz_05'], 'DBZ (0.5°)', dbz_cmap, dbz_norm, True, 'dBZ')
            
            # VEL 0.5°
            vel_cmap, vel_norm = get_vel_cmap()
            plot_polar_channel(axes[1], case['vel_05'], 'VEL (0.5°)', vel_cmap, vel_norm, True, 'm/s')
            
            # VEL 0.9°
            plot_polar_channel(axes[2], case['vel_09'], 'VEL (0.9°)', vel_cmap, vel_norm, True, 'm/s')
            
            # Physics Attention (只显示 top 15% 的区域)
            attn_masked = mask_attention_top(case['physics_attn'], top_percent=15)
            attn_cmap, attn_norm = get_attn_cmap_focused(case['physics_attn'], top_percent=15)
            plot_polar_channel(axes[3], attn_masked,
                              f'Physics Attention (max={case["physics_attn"].max():.3f})',
                              attn_cmap, attn_norm, True, 'Attention')
            
            # Spatial Prediction (只显示 top 15% 的区域)
            sp_masked = mask_attention_top(case['spatial_pred'], top_percent=15)
            sp_cmap, sp_norm = get_attn_cmap_focused(case['spatial_pred'], top_percent=15)
            plot_polar_channel(axes[4], sp_masked,
                              f'Spatial Prediction (max={case["spatial_pred"].max():.3f})',
                              sp_cmap, sp_norm, True, 'Probability')
            
            plt.tight_layout()
            save_path = os.path.join(args.save_dir, f"{case_type}_{j:02d}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  ✅ {save_path}")
    
    # ====== 生成 Fig.4 组合大图 ======
    print("\n🖼️ 生成 Fig.4 组合图...")
    
    # 选代表样本
    row_order = ['TP_strong', 'TP_weak', 'TN_wrn', 'FP', 'FN']
    selected = []
    for ct in row_order:
        cases = collectors[ct]
        if not cases:
            continue
        if ct == 'TP_strong':
            selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'TP_weak':
            selected.append((ct, min(cases, key=lambda x: abs(x['prob'] - 0.55))))
        elif ct == 'TN_wrn':
            selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'FP':
            selected.append((ct, max(cases, key=lambda x: x['prob'])))
        elif ct == 'FN':
            # 选概率最接近阈值的（最遗憾的漏检）
            selected.append((ct, max(cases, key=lambda x: x['prob'])))
    
    if len(selected) < 2:
        print("  ⚠️ 样本不足，跳过组合图")
        return
    
    n_rows = len(selected)
    
    # 用 GridSpec 精确控制布局：每列底部留色条空间
    fig = plt.figure(figsize=(30, 5.0 * n_rows + 2))
    gs = fig.add_gridspec(n_rows + 1, 5, height_ratios=[*([1] * n_rows), 0.05],
                          wspace=0.08, hspace=0.18)
    
    # 创建主绘图 axes
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(n_rows)])
    # 创建色条 axes（底部一行）
    cbar_axes = [fig.add_subplot(gs[n_rows, c]) for c in range(5)]
    
    type_labels = {
        'TP_strong': 'TP (Strong)', 'TP_weak': 'TP (Weak)',
        'TN_wrn': 'TN (Warning)', 'FP': 'FP (False Alarm)', 'FN': 'FN (Missed)'
    }
    type_colors = {
        'TP_strong': '#2E7D32', 'TP_weak': '#558B2F',
        'TN_wrn': '#1565C0', 'FP': '#C62828', 'FN': '#E65100'
    }
    
    # 列标题（大字号，确保打印清晰）
    col_titles = ['DBZ (0.5°)', 'VEL (0.5°)', 'VEL (0.9°)', 'Physics Attention', 'Spatial Prediction']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # 记录每列的 mesh 用于画统一色条
    col_meshes = [None] * 5
    
    for row_idx, (ct, case) in enumerate(selected):
        row_label = f"({chr(97 + row_idx)}) {type_labels[ct]}"
        
        # DBZ
        dbz_cmap, dbz_norm = get_dbz_cmap()
        m = plot_polar_channel(axes[row_idx, 0], case['dbz_05'], '', dbz_cmap, dbz_norm)
        if row_idx == 0: col_meshes[0] = m
        
        # VEL 0.5°
        vel_cmap, vel_norm = get_vel_cmap()
        m = plot_polar_channel(axes[row_idx, 1], case['vel_05'], '', vel_cmap, vel_norm)
        if row_idx == 0: col_meshes[1] = m
        
        # VEL 0.9°
        m = plot_polar_channel(axes[row_idx, 2], case['vel_09'], '', vel_cmap, vel_norm)
        if row_idx == 0: col_meshes[2] = m
        
        # Physics Attention (top 15% masked)
        attn_masked = mask_attention_top(case['physics_attn'], top_percent=15)
        attn_cmap, attn_norm = get_attn_cmap_focused(case['physics_attn'], top_percent=15)
        m = plot_polar_channel(axes[row_idx, 3], attn_masked, '', attn_cmap, attn_norm)
        if row_idx == 0: col_meshes[3] = m
        
        # Spatial Prediction (top 15% masked)
        sp_masked = mask_attention_top(case['spatial_pred'], top_percent=15)
        sp_cmap, sp_norm = get_attn_cmap_focused(case['spatial_pred'], top_percent=15)
        m = plot_polar_channel(axes[row_idx, 4], sp_masked, '', sp_cmap, sp_norm)
        if row_idx == 0: col_meshes[4] = m
        
        # 行标签（左侧竖排）
        axes[row_idx, 0].text(
            -0.06, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=13, fontweight='bold',
            color=type_colors.get(ct, 'black'),
            va='center', ha='right', rotation=90
        )
        
        # 概率+EF 标注（右下角）
        prob_color = type_colors.get(ct, 'gray')
        ef_str = f" | EF{case['ef']}" if case['ef'] >= 0 else ""
        axes[row_idx, 4].text(
            0.97, 0.06, f'P={case["prob"]:.3f}{ef_str}',
            transform=axes[row_idx, 4].transAxes,
            fontsize=11, fontweight='bold', color='white', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=prob_color, alpha=0.85)
        )
        
        # 注意力最大值标注
        axes[row_idx, 3].text(
            0.97, 0.06, f'max={case["physics_attn"].max():.3f}',
            transform=axes[row_idx, 3].transAxes,
            fontsize=10, color='white', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
        )
    
    # ====== 底部统一色条 ======
    cbar_labels = ['dBZ', 'm/s', 'm/s', 'Attention', 'Probability']
    
    # DBZ 色条
    dbz_cmap_cb, dbz_norm_cb = get_dbz_cmap()
    cb0 = plt.colorbar(plt.cm.ScalarMappable(norm=dbz_norm_cb, cmap=dbz_cmap_cb),
                       cax=cbar_axes[0], orientation='horizontal')
    cb0.set_label('dBZ', fontsize=11)
    cb0.ax.tick_params(labelsize=9)
    
    # VEL 色条 (col 1 和 col 2 共用)
    vel_cmap_cb, vel_norm_cb = get_vel_cmap()
    for ci in [1, 2]:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=vel_norm_cb, cmap=vel_cmap_cb),
                          cax=cbar_axes[ci], orientation='horizontal')
        cb.set_label('m/s', fontsize=11)
        cb.ax.tick_params(labelsize=9)
    
    # Attention 色条 (用 0-1 范围)
    attn_cmap_cb = mcolors.LinearSegmentedColormap.from_list(
        'attn_cb', ['#FFFFFF', '#FFF5CC', '#FFE066', '#FFB300', '#FF6F00', '#D50000', '#880000'], N=256)
    cb3 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=attn_cmap_cb),
                       cax=cbar_axes[3], orientation='horizontal')
    cb3.set_label('Attention', fontsize=11)
    cb3.ax.tick_params(labelsize=9)
    
    # Spatial Prediction 色条
    cb4 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=attn_cmap_cb),
                       cax=cbar_axes[4], orientation='horizontal')
    cb4.set_label('Probability', fontsize=11)
    cb4.ax.tick_params(labelsize=9)
    
    combo_path = os.path.join(args.save_dir, "Fig4_attention_visualization.png")
    plt.savefig(combo_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Fig.4: {combo_path}")
    
    # 统计
    print("\n📊 收集统计:")
    for ct, cases in collectors.items():
        print(f"  {ct}: {len(cases)} samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_pi_v8_GWP.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures_4,5,6')
    parser.add_argument('--max_per_type', type=int, default=5)
    args = parser.parse_args()
    main(args)


