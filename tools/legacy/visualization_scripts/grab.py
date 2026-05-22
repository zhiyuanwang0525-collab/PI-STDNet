# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# 拦截 sys.argv 防报错
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  
import train_compare
sys.argv = original_argv  

class PolarCasePlotter:
    """PI-STDNet 出版级雷达扇形绘图引擎 (终极防拉伸完美比例版)"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cmap_z = plt.cm.nipy_spectral  
        self.cmap_v = plt.cm.seismic
        
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 11, 
            'figure.dpi': 300,
            'axes.linewidth': 1.5
        })

    def _get_polar_grid(self, H, W):
        az = np.linspace(np.radians(-30), np.radians(30), H + 1)
        rng = np.linspace(0, 60, W + 1) 
        R, Theta = np.meshgrid(rng, az)
        X = R * np.sin(Theta)
        Y = R * np.cos(Theta)
        return X, Y

    def plot_polar_mesh(self, ax, data_2d, X, Y, cmap, vmin, vmax, title):
        masked_data = np.ma.masked_invalid(data_2d)
        
        mesh = ax.pcolormesh(X, Y, masked_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                             shading='auto', edgecolors='none', linewidth=0, rasterized=True)
                             
        ax.set_title(title, pad=12, fontweight='bold', fontsize=13)
        ax.set_facecolor('black') 
        ax.set_xticks([])
        ax.set_yticks([])

        r_max = 60
        theta_l, theta_r = np.radians(-30), np.radians(30)
        ax.plot([0, r_max*np.sin(theta_l)], [0, r_max*np.cos(theta_l)], 'white', lw=1.0, alpha=0.6)
        ax.plot([0, r_max*np.sin(theta_r)], [0, r_max*np.cos(theta_r)], 'white', lw=1.0, alpha=0.6)
        arc = np.linspace(theta_l, theta_r, 100)
        ax.plot(r_max*np.sin(arc), r_max*np.cos(arc), 'white', lw=1.0, alpha=0.6)
        
        # ==========================================================
        # 💡 核心修复：施加最强硬的 1:1 物理箱体比例锁定！
        # 强制 X轴跨度(64) 和 Y轴跨度(64) 在屏幕上的像素绝对相等
        # ==========================================================
        ax.set_xlim(-32, 32)
        ax.set_ylim(-2, 62)
        ax.set_aspect('equal', adjustable='box')
        
        return mesh

    def _add_scale_bar(self, ax, x, y, length_km, label):
        rect = patches.Rectangle((x, y), length_km, 1.5, color='white', zorder=15)
        ax.add_patch(rect)
        ax.text(x + length_km/2, y + 2.5, label, color='white', fontsize=10, fontweight='bold', ha='center')

    def plot_4_golden_cases(self, dataset, cases):
        # 微调画布长宽比，使其更契合 4个正方形并排
        fig = plt.figure(figsize=(22, 12))
        gs = gridspec.GridSpec(2, 4, wspace=0.05, hspace=0.15, bottom=0.15)

        mesh_z, mesh_v = None, None

        for i, case in enumerate(cases):
            row = i // 2
            col_start = (i % 2) * 2
            
            idx = case['idx']
            image_tensor, _, _ = dataset[idx]
            images = image_tensor.numpy()
            
            z_data = images[0].copy()
            v_data = images[1].copy()
            
            if np.max(z_data) <= 2.0: 
                z_data = z_data * 110.0 - 30.0
            if np.max(np.abs(v_data)) <= 2.0:
                v_data = v_data * 100.0 - 50.0

            H, W = z_data.shape
            
            z_data[z_data < 0] = np.nan
            v_data[np.isnan(z_data)] = np.nan 
            
            X, Y = self._get_polar_grid(H, W)

            ax_z = fig.add_subplot(gs[row, col_start])
            ax_v = fig.add_subplot(gs[row, col_start + 1])

            title_z = f"Reflectivity ($Z$)"
            title_v = f"Radial Velocity ($V_r$)"
            
            mesh_z = self.plot_polar_mesh(ax_z, z_data, X, Y, self.cmap_z, 0, 60, title_z)
            mesh_v = self.plot_polar_mesh(ax_v, v_data, X, Y, self.cmap_v, -30, 30, title_v)

            for ax in [ax_z, ax_v]:
                self._add_scale_bar(ax, x=-28, y=2, length_km=10, label='10 km')

            cx = X[H//2, W//2]
            cy = Y[H//2, W//2]
            box_half_width = 8.0 
            
            for ax in [ax_z, ax_v]:
                rect = patches.Rectangle((cx - box_half_width, cy - box_half_width), 
                                         box_half_width * 2, box_half_width * 2, 
                                         edgecolor='white', facecolor='none', lw=2.5, zorder=10)
                ax.add_patch(rect)

            border_color = '#DC143C' if case['type'] == 'TOR' else '#4169E1'
            case_title = f"Case {i+1}: {case['desc']}"
            ax_z.text(0.02, 0.96, case_title, transform=ax_z.transAxes,
                      fontsize=12, fontweight='bold', color='white', 
                      ha='left', va='top', bbox=dict(boxstyle="square,pad=0.3", fc="black", ec=border_color, lw=1.5, alpha=0.8), zorder=20)
            
            stats_text = (f"Baseline: {case['base']:.2f}\n"
                          f"PI-STDNet: {case['ours']:.2f}")
            ax_v.text(0.98, 0.96, stats_text, transform=ax_v.transAxes,
                      fontsize=11, fontweight='bold', color='white', 
                      ha='right', va='top', bbox=dict(boxstyle="square,pad=0.3", fc="black", ec=border_color, lw=1.5, alpha=0.8), zorder=20)

        cbar_ax_z = fig.add_axes([0.15, 0.08, 0.33, 0.02])
        cb_z = fig.colorbar(mesh_z, cax=cbar_ax_z, orientation='horizontal')
        cb_z.set_label('Reflectivity (dBZ)', fontsize=13, fontweight='bold')
        cb_z.ax.tick_params(labelsize=12)

        cbar_ax_v = fig.add_axes([0.55, 0.08, 0.33, 0.02])
        cb_v = fig.colorbar(mesh_v, cax=cbar_ax_v, orientation='horizontal')
        cb_v.set_label('Radial Velocity (m/s)', fontsize=13, fontweight='bold')
        cb_v.ax.tick_params(labelsize=12)

        save_path = os.path.join(self.save_dir, "Figure5_Radar_Polar_Grid.pdf")
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"\n✅ 绝杀！【完美比例防拉伸版组图】已保存至: {save_path}")
        plt.close()

if __name__ == '__main__':
    print("🌟 正在挂载 TorNet 测试集...")
    dataset = train_compare.TorNetDatasetAblation(
        '/path/to/TorNet/catalog.csv',  
        '/path/to/TorNet',                 
        mode='test'
    )
    
    golden_cases = [
        {"idx": 26791, "type": "TOR", "base": 0.23, "ours": 0.63, "desc": "Captured Missed Tornado"},
        {"idx": 2067,  "type": "TOR", "base": 0.36, "ours": 0.77, "desc": "Captured Missed Tornado"},
        {"idx": 27306, "type": "WRN", "base": 0.69, "ours": 0.24, "desc": "Suppressed False Alarm"},
        {"idx": 2696,  "type": "WRN", "base": 0.61, "ours": 0.20, "desc": "Suppressed False Alarm"}
    ]
    
    plotter = PolarCasePlotter()
    plotter.plot_4_golden_cases(dataset, golden_cases)
