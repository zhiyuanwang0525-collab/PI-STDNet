# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from torch.utils.data import DataLoader

# 拦截 sys.argv 防报错
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  
import train_compare
sys.argv = original_argv  

class GalleryPlotter:
    """批量选角画廊引擎：直接可视化 Z 和 V 供人工挑选"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 12,
            'figure.dpi': 300,
            'axes.linewidth': 1.2,
        })
        self.cmap_z = plt.cm.nipy_spectral
        self.cmap_v = plt.cm.seismic

    def plot_gallery(self, dataset, cases, category_name):
        fig = plt.figure(figsize=(12, 22)) # 长图，方便从上往下看
        gs = gridspec.GridSpec(len(cases), 2, wspace=0.1, hspace=0.3)
        
        print(f"\n📸 正在冲洗 {category_name} 组选角照片...")

        for row_idx, case in enumerate(cases):
            idx = case['idx']
            image_tensor, _, _ = dataset[idx]
            images = image_tensor.numpy()

            z_data = images[0] * 110.0 - 30.0  
            v_data = images[1] * 100.0 - 50.0  

            # 1. 绘制 Z
            ax_z = fig.add_subplot(gs[row_idx, 0])
            im_z = ax_z.imshow(z_data, cmap=self.cmap_z, vmin=0, vmax=65, origin='lower')
            ax_z.set_xticks([]); ax_z.set_yticks([])
            
            # 2. 绘制 V
            ax_v = fig.add_subplot(gs[row_idx, 1])
            im_v = ax_v.imshow(v_data, cmap=self.cmap_v, vmin=-30, vmax=30, origin='lower')
            ax_v.set_xticks([]); ax_v.set_yticks([])

            # 把红蓝切变最剧烈的地方用框标出来，帮你的眼睛减负
            # 简单粗暴的切变寻找逻辑：用一个 3x3 的卷积核扫一遍 V 场，找最大差值
            v_padded = np.pad(v_data, 1, mode='edge')
            shear_map = np.zeros_like(v_data)
            for i in range(v_data.shape[0]):
                for j in range(v_data.shape[1]):
                    window = v_padded[i:i+3, j:j+3]
                    shear_map[i, j] = np.max(window) - np.min(window)
            
            cy, cx = np.unravel_index(np.argmax(shear_map), shear_map.shape)
            rect = patches.Rectangle((cx - 8, cy - 8), 16, 16, 
                                     linewidth=2.0, edgecolor='#00FF00', facecolor='none')
            ax_v.add_patch(rect)

            # 标题信息
            if row_idx == 0:
                ax_z.set_title("Reflectivity (Z)", fontweight='bold', pad=10)
                ax_v.set_title("Radial Velocity (V) + Auto Shear Box", fontweight='bold', pad=10)

            stats_str = f"Idx: [{idx}]\nBase Prob: {case['base']:.2f}\nOurs Prob: {case['ours']:.2f}"
            color = "#DC143C" if category_name == "TOR" else "#4169E1"
            ax_z.set_ylabel(stats_str, fontweight='bold', fontsize=12, color=color, labelpad=15, rotation=0, ha='right', va='center')

        save_path = os.path.join(self.save_dir, f"Gallery_Casting_{category_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ {category_name} 画廊已保存至: {save_path}")
        plt.close()

if __name__ == '__main__':
    dataset = train_compare.TorNetDatasetAblation('/path/to/TorNet/catalog.csv', '/path/to/TorNet', mode='test')
    
    # 你提供的绝杀名单直接硬编码填入
    tor_cases = [
        {"idx": 2067,  "base": 0.3657, "ours": 0.7739},
        {"idx": 9073,  "base": 0.3184, "ours": 0.7207},
        {"idx": 9074,  "base": 0.3184, "ours": 0.7207},
        {"idx": 26791, "base": 0.2346, "ours": 0.6338},
        {"idx": 2075,  "base": 0.3535, "ours": 0.7422}
    ]
    
    wrn_cases = [
        {"idx": 27306, "base": 0.6978, "ours": 0.2456},
        {"idx": 2696,  "base": 0.6162, "ours": 0.2085},
        {"idx": 18712, "base": 0.7412, "ours": 0.3535},
        {"idx": 2476,  "base": 0.6250, "ours": 0.2383},
        {"idx": 9603,  "base": 0.6274, "ours": 0.2754}
    ]
    
    plotter = GalleryPlotter()
    plotter.plot_gallery(dataset, tor_cases, "TOR")
    plotter.plot_gallery(dataset, wrn_cases, "WRN")
