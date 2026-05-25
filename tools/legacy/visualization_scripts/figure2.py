# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter

# 全局顶刊级排版配置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'axes.linewidth': 1.5,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
})

class InputSignaturesPlotter:
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.cmap_z = plt.cm.turbo
        self.norm_z = mcolors.Normalize(vmin=-10, vmax=65) 
        
        self.cmap_v = plt.cm.coolwarm
        self.norm_v = mcolors.Normalize(vmin=-40, vmax=40)
        
        self.cmap_shear = plt.cm.YlOrRd
        self.norm_shear = mcolors.Normalize(vmin=0, vmax=25) 
        
        # 🌟 优化了 TDS 的色表亮度上限，使得碎屑特征更亮、更刺眼
        self.cmap_tds = plt.cm.magma
        self.norm_tds = mcolors.Normalize(vmin=0, vmax=0.15)

    def calculate_azimuthal_shear(self, velocity, window_size=5):
        padded_vel = np.pad(velocity, ((window_size//2, window_size//2), (0, 0)), mode='edge')
        raw_shear = np.zeros_like(velocity)
        for i in range(velocity.shape[0]):
            window = padded_vel[i:i+window_size, :]
            raw_shear[i, :] = np.max(window, axis=0) - np.min(window, axis=0)
        return raw_shear

    def calculate_anomaly_shear(self, raw_shear, smooth_sigma=15):
        background = uniform_filter(raw_shear, size=smooth_sigma)
        anomaly = raw_shear - background
        anomaly[anomaly < 0] = 0
        return anomaly

    def plot_polar_mesh(self, ax, data, title, cmap, norm):
        H, W = data.shape
        az = np.linspace(np.radians(-30), np.radians(30), H + 1)
        rng = np.linspace(0, 60, W + 1)
        R, Theta = np.meshgrid(rng, az)
        X = R * np.sin(Theta)
        Y = R * np.cos(Theta)
        
        masked_data = np.ma.masked_invalid(data)
        mesh = ax.pcolormesh(X, Y, masked_data, cmap=cmap, norm=norm, shading='auto')
        
        ax.set_title(title, pad=12)
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        
        r_max = 60
        theta_left, theta_right = np.radians(-30), np.radians(30)
        ax.plot([0, r_max * np.sin(theta_left)], [0, r_max * np.cos(theta_left)], 'k-', lw=1.0)
        ax.plot([0, r_max * np.sin(theta_right)], [0, r_max * np.cos(theta_right)], 'k-', lw=1.0)
        arc_theta = np.linspace(theta_left, theta_right, 100)
        ax.plot(r_max * np.sin(arc_theta), r_max * np.cos(arc_theta), 'k-', lw=1.0)
        ax.axis('off')
        return mesh

    def draw_showcase(self, dbz_05, vel_05, rho_05):
        # 1. 物理特征计算
        raw_shear = self.calculate_azimuthal_shear(vel_05, window_size=5)
        anomaly_shear = self.calculate_anomaly_shear(raw_shear, smooth_sigma=15)
        
        # 🌟 TDS 公式: Z_norm * (1 - RhoHV)
        z_norm = np.clip((dbz_05 + 30) / 110.0, 0, 1) 
        tds = z_norm * (1.0 - rho_05)
        
        valid_mask = dbz_05 > 15.0 # 雷达回波极弱的地方没有碎屑，过滤掉噪声
        raw_shear = np.where(valid_mask, raw_shear, 0)
        anomaly_shear = np.where(valid_mask, anomaly_shear, 0)
        tds = np.where(valid_mask, tds, 0)

        # 2. 顶刊排版 (1行5列)
        fig = plt.figure(figsize=(28, 6.5))
        gs = GridSpec(2, 5, height_ratios=[1, 0.05], wspace=0.03, hspace=0.1, 
                      left=0.03, right=0.97, top=0.90, bottom=0.1)
        
        axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
        cax_list = [fig.add_subplot(gs[1, i]) for i in range(5)]

        m0 = self.plot_polar_mesh(axes[0], dbz_05, '(a) Reflectivity ($Z$)', self.cmap_z, self.norm_z)
        m1 = self.plot_polar_mesh(axes[1], vel_05, '(b) Radial Velocity ($V_r$)', self.cmap_v, self.norm_v)
        m2 = self.plot_polar_mesh(axes[2], raw_shear, '(c) Raw Azimuthal Shear', self.cmap_shear, self.norm_shear)
        m3 = self.plot_polar_mesh(axes[3], anomaly_shear, '(d) Anomaly Shear (MAAS)', self.cmap_shear, self.norm_shear)
        m4 = self.plot_polar_mesh(axes[4], tds, '(e) Debris Signature (TDS)', self.cmap_tds, self.norm_tds)

        # 色条配置
        cbs = [
            (m0, 'dBZ'), (m1, 'm/s'), (m2, 'Shear Intensity'), 
            (m3, 'Anomaly Magnitude'), (m4, 'TDS Index')
        ]
        
        for i, (m, label) in enumerate(cbs):
            cb = plt.colorbar(m, cax=cax_list[i], orientation='horizontal')
            cb.set_label(label, fontsize=13, fontweight='bold', labelpad=5)
            cax_list[i].tick_params(labelsize=11, direction='in', length=4)
            for spine in cax_list[i].spines.values():
                spine.set_linewidth(1.0)

        save_path = os.path.join(self.save_dir, "Figure2_Input_Signatures_5Panels.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\n✅ 完美！包含震撼 TDS 的全景图已保存至: {save_path}")

def execute(args):
    print(f"📖 正在加载 Catalog 数据...")
    df = pd.read_csv(args.catalog_path)
    
    tor_indices = df[df['category'] == 'TOR'].index.tolist()
    print(f"🔍 开启 TDS 黄金猎手！正在扫描 {len(tor_indices)} 个龙卷事件，寻找最完美的碎屑特征...")
    
    best_idx = None
    best_tds_score = -1
    best_data = None
    
    # 随机抽查 200 个真实的龙卷风寻找最高 TDS，避免全量扫描太慢
    np.random.seed(42)
    sample_indices = np.random.choice(tor_indices, min(200, len(tor_indices)), replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        year = pd.to_datetime(row['start_time']).year if 'start_time' in row else 2013
        nc_path = os.path.join(args.data_root, f"tornet_{year}", row['filename'])
        if not os.path.exists(nc_path):
            nc_path = os.path.join(args.data_root, row['filename'])
            
        try:
            with xr.open_dataset(nc_path, engine='netcdf4', cache=False) as ds:
                # 🌟 修复变量名读取：完美兼容 RHOHV 或 RHO
                rho_var = None
                for v in ['RHOHV', 'RHO', 'RhoHV', 'CC']:
                    if v in ds:
                        rho_var = v; break
                
                if not rho_var: continue # 如果连极化数据都没有，直接跳过
                
                dbz_raw = ds['DBZ'].values
                vel_raw = ds['VEL'].values
                rho_raw = ds[rho_var].values
                
                t_idx = min(3, dbz_raw.shape[0] - 1)
                dbz_05 = np.nan_to_num(dbz_raw[t_idx, :, :, 0], nan=-30.0)
                vel_05 = np.nan_to_num(vel_raw[t_idx, :, :, 0], nan=0.0)
                rho_05 = np.nan_to_num(rho_raw[t_idx, :, :, 0], nan=1.0)
                
                # 计算 TDS 得分 (强回波 + 低相关系数)
                z_norm = np.clip((dbz_05 + 30) / 110.0, 0, 1)
                tds = z_norm * (1.0 - rho_05)
                # 只有在高反射率区 (真正的降水/碎屑) 的 TDS 才是真实的
                valid_mask = dbz_05 > 40.0 
                
                if np.any(valid_mask):
                    score = np.max(tds[valid_mask])
                    if score > best_tds_score:
                        best_tds_score = score
                        best_idx = idx
                        best_data = (dbz_05, vel_05, rho_05)
                        
        except Exception:
            continue
            
    if best_data is None:
        print("❌ 未能在数据集中找到包含有效 RHOHV 数据的样本，请检查你的 TorNet 数据是否完整。")
        return
        
    print(f"🏆 寻宝成功！找到最佳展示样本 Index: {best_idx} (TDS 峰值强度: {best_tds_score:.3f})")
    print("📡 正在渲染终极版物理输入全景图...")
    
    plotter = InputSignaturesPlotter(save_dir=args.save_dir)
    plotter.draw_showcase(best_data[0], best_data[1], best_data[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    execute(args)


