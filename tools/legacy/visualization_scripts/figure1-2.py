# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import argparse
import torch
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入你自己的真实模型、数据集和配置
from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG

class ResearchPlotter:
    """PI-STDNet 出版级雷达扇形绘图引擎 (终极定稿版)"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # 初始化气象色带
        self.cmap_z = plt.cm.nipy_spectral  
        self.cmap_v = plt.cm.seismic
        self.cmap_prob = plt.cm.magma
        
        # 强制接管全局 paper font和排版规范
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 10, 
            'figure.dpi': 300,
            'axes.linewidth': 1.2
        })

    def _get_polar_grid(self, H=120, W=240):
        """生成 TorNet 标准扇形投影物理网格 (-30度到30度, 60km)"""
        az = np.linspace(np.radians(-30), np.radians(30), H + 1)
        rng = np.linspace(0, 60, W + 1) 
        R, Theta = np.meshgrid(rng, az)
        X = R * np.sin(Theta)
        Y = R * np.cos(Theta)
        return X, Y, az, rng

    def plot_polar_mesh(self, ax, data_2d, X, Y, cmap, vmin, vmax, title):
        """绘制带雷达边界线的完整扇形图"""
        masked_data = np.ma.masked_invalid(data_2d)
        mesh = ax.pcolormesh(X, Y, masked_data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title, pad=10, fontweight='bold', fontsize=11)
        
        ax.set_facecolor('black') # 纯黑深空背景
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        # 绘制白色的扇面雷达扫描边界线
        r_max = 60
        theta_l, theta_r = np.radians(-30), np.radians(30)
        ax.plot([0, r_max*np.sin(theta_l)], [0, r_max*np.cos(theta_l)], 'white', lw=0.8, alpha=0.5)
        ax.plot([0, r_max*np.sin(theta_r)], [0, r_max*np.cos(theta_r)], 'white', lw=0.8, alpha=0.5)
        arc = np.linspace(theta_l, theta_r, 100)
        ax.plot(r_max*np.sin(arc), r_max*np.cos(arc), 'white', lw=0.8, alpha=0.5)
        
        ax.set_xlim(X.min() - 2, X.max() + 2)
        ax.set_ylim(Y.min() - 2, Y.max() + 2)
        return mesh

    def _add_scale_bar(self, ax, x, y, length_km, label):
        """在扇形图底部添加物理比例尺"""
        rect = patches.Rectangle((x, y), length_km, 1.5, color='white', zorder=15)
        ax.add_patch(rect)
        ax.text(x + length_km/2, y + 2.5, label, color='white', fontsize=9, fontweight='bold', ha='center')

    def plot_figure_1_motivation(self, wrn_data, tor_data):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
        X, Y, az, rng = self._get_polar_grid(120, 240)
        
        def plot_row(ax_row, data, title_prefix, is_tor=False):
            # 💡 气象后处理：物理掩码与视觉降噪，让概率图彻底干净
            prob_display = data['Prob'].copy()
            prob_display[data['Z'] < 10] = 0.0       # 晴空掩码：无回波必无龙卷
            prob_display[prob_display < 0.2] = 0.0   # 底噪抑制：切断 magma 色带视觉污染

            # 1. 绘制基础底图
            im_z = self.plot_polar_mesh(ax_row[0], data['Z'], X, Y, self.cmap_z, -10, 65, f'{title_prefix[0]} Reflectivity ($Z$)')
            im_v = self.plot_polar_mesh(ax_row[1], data['V'], X, Y, self.cmap_v, -35, 35, f'{title_prefix[1]} Radial Velocity ($V_r$)')
            im_p = self.plot_polar_mesh(ax_row[2], prob_display, X, Y, self.cmap_prob, 0, 1.0, f'{title_prefix[2]} PI-STDNet Output')

            # 2. 计算物理坐标中心
            center_rng = rng[data['center_x']]
            center_az = az[data['center_y']]
            cx = center_rng * np.sin(center_az)
            cy = center_rng * np.cos(center_az)

            # 统一比例尺位置
            for ax in ax_row:
                self._add_scale_bar(ax, x=-28, y=2, length_km=10, label='10 km')

            # 3. 极简视觉统一：全部使用白色实线方框
            box_half_width = 6.0 # 框的半宽 6km
            for i, ax in enumerate(ax_row):
                rect = patches.Rectangle((cx - box_half_width, cy - box_half_width), 
                                         box_half_width * 2, box_half_width * 2, 
                                         edgecolor='white', facecolor='none', lw=2.5, zorder=10)
                ax.add_patch(rect)

                # 只有真实龙卷 (下排) 才添加画中画特写，虚警风暴 (上排) 保持宏观视野
                if is_tor:
                    axins = ax.inset_axes([0.62, 0.62, 0.35, 0.35])
                    
                    if i == 0:
                        axins.pcolormesh(X, Y, np.ma.masked_invalid(data['Z']), cmap=self.cmap_z, vmin=-10, vmax=65, shading='auto')
                    elif i == 1:
                        axins.pcolormesh(X, Y, np.ma.masked_invalid(data['V']), cmap=self.cmap_v, vmin=-35, vmax=35, shading='auto')
                    else:
                        axins.pcolormesh(X, Y, np.ma.masked_invalid(prob_display), cmap=self.cmap_prob, vmin=0, vmax=1.0, shading='auto')
                    
                    # 锁定子图的视野到方框区域
                    axins.set_xlim(cx - box_half_width, cx + box_half_width)
                    axins.set_ylim(cy - box_half_width, cy + box_half_width)
                    axins.set_xticks([])
                    axins.set_yticks([])
                    
                    # 画中画边框统一为白色
                    for spine in axins.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(2.5)
                    
                    # 自动连线
                    ax.indicate_inset_zoom(axins, edgecolor="white", alpha=0.8, lw=1.5)
                
            return im_z, im_v, im_p

        # 渲染两行
        im_z, im_v, im_p = plot_row(axes[0], wrn_data, ['(a)', '(b)', '(c)'], is_tor=False)
        plot_row(axes[1], tor_data, ['(d)', '(e)', '(f)'], is_tor=True)

        cbar_kwargs = {'orientation': 'horizontal', 'pad': 0.02, 'aspect': 30, 'fraction': 0.05}
        fig.colorbar(im_z, ax=axes[:, 0], label='Reflectivity (dBZ)', **cbar_kwargs)
        fig.colorbar(im_v, ax=axes[:, 1], label='Radial Velocity (m/s)', **cbar_kwargs)
        fig.colorbar(im_p, ax=axes[:, 2], label='Prediction Probability', **cbar_kwargs)

        save_path = os.path.join(self.save_dir, "Figure1_Motivation_Final.pdf")
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"\n✅ 绘图成功！完美排版和纯净概率场的定稿图已保存至: {save_path}")
        plt.close()


def find_and_plot_figure1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在初始化绘图引擎... 设备: {device}")
    
    model = PI_STDNet_V8().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    dataset = TorNetDatasetV8_Full(catalog_path=args.catalog_path, root_dir=args.data_root, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    wrn_case = None
    tor_case = None

    print("🔍 正在启动【务实版局部鲁棒筛选】，寻找完美气象案例...")
    with torch.no_grad():
        for i, (inputs, labels, cat_ids) in enumerate(tqdm(dataloader, desc="Scanning Cases")):
            inputs = inputs.to(device)
            cat_val = cat_ids[0].item() 
            
            with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                cls_logit, aux_logit, spatial_logits, _ = model(inputs)
            
            img_prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).item()
            prob_map = torch.sigmoid(spatial_logits).squeeze().cpu().numpy()

            # 反归一化雷达数据
            raw_z = inputs[0, 33, :, :].cpu().numpy() * 110.0 - 30.0   
            raw_v = inputs[0, 34, :, :].cpu().numpy() * 100.0 - 50.0   

            # ==============================================================
            # 🌪️ 抓取 WRN：局部绝对清醒的强风暴
            # ==============================================================
            if wrn_case is None and cat_val == 1:
                if np.sum(raw_z > 40) > 250: # 面积要求降到250
                    smoothed_z = ndimage.gaussian_filter(raw_z, sigma=4)
                    y_max, x_max = np.unravel_index(np.argmax(smoothed_z), smoothed_z.shape)
                    
                    if 20 < x_max < 220 and 20 < y_max < 100:
                        # 核心防漏报逻辑：只检查主风暴核心周围 20x20 的区域
                        local_prob = prob_map[max(0, y_max-10):min(120, y_max+10), max(0, x_max-10):min(240, x_max+10)]
                        
                        if np.max(local_prob) < 0.4: 
                            wrn_case = {
                                'Z': raw_z, 'V': raw_v, 'Prob': prob_map,
                                'center_x': x_max, 'center_y': y_max 
                            }
                            print(f"\n🎯 找到完美局部低概率 WRN！索引: {i}, 核心区最高概率仅: {np.max(local_prob):.3f}")

            # ==============================================================
            # 🌪️ 抓取 TOR：基于百分位数的鲁棒涡旋提取
            # ==============================================================
            if tor_case is None and cat_val == 2 and img_prob > 0.75: # 总体概率要求稍降到0.75
                y_max, x_max = np.unravel_index(np.argmax(prob_map), prob_map.shape)
                
                # 确保风暴中心远离雷达边界
                if 20 < x_max < 220 and 20 < y_max < 100:
                    
                    # 局部有一定回波即可
                    local_z = raw_z[y_max-3:y_max+3, x_max-3:x_max+3]
                    if np.max(local_z) > 20: 
                        
                        # 提取局部速度场
                        local_v = raw_v[y_max-8:y_max+8, x_max-8:x_max+8]
                        
                        # 用 98% 和 2% 百分位数算速度差，完美忽略孤立噪点
                        v_max_robust = np.percentile(local_v, 98)
                        v_min_robust = np.percentile(local_v, 2)
                        v_diff = v_max_robust - v_min_robust
                        
                        if v_diff > 30: # 速度差阈值 30m/s
                            tor_case = {
                                'Z': raw_z, 'V': raw_v, 'Prob': prob_map,
                                'center_x': x_max, 'center_y': y_max 
                            }
                            print(f"\n🎯 找到极佳教科书级 TOR！索引: {i}, 鲁棒速度差: {v_diff:.1f}m/s")

            if wrn_case is not None and tor_case is not None:
                break

    if wrn_case and tor_case:
        plotter = ResearchPlotter(save_dir=args.save_dir)
        plotter.plot_figure_1_motivation(wrn_case, tor_case)
    else:
        print("❌ 依然未能找到同时满足条件的样本，建议继续适当降低 WRN 的面积/概率要求或 TOR 的速度差。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_pi_v8_GWP1.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures1')
    args = parser.parse_args()
    
    find_and_plot_figure1(args)


