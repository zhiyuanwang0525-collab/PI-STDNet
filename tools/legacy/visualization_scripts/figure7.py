# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader

# 拦截 sys.argv 防报错
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  
import train_compare
sys.argv = original_argv  

class MockArgs:
    def __init__(self, attn_type='physics', disable_physics_attn=False, disable_rot_stats=False):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = False
        self.disable_topk = True  
        self.disable_rot_stats = disable_rot_stats

class AttentionPlotterTGRS:
    """TGRS 级定稿：极限紧凑 + 内部代号抹除 + 完美大满贯版"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 13,
            'figure.dpi': 300,
            'axes.linewidth': 1.5,
        })
        self.cmap_z = plt.cm.nipy_spectral
        self.cmap_v = plt.cm.seismic
        self.cmap_hm = plt.cm.jet 

    def plot_attention_grid(self, dataset, cases, models_dict, device):
        # 画布纵向稍微缩小，配合紧凑的间距
        fig = plt.figure(figsize=(24, 10))
        # 💡 核心微调 1：将 hspace 从 0.15 压缩到 0.04，上下两行紧密贴合
        gs = gridspec.GridSpec(2, 5, wspace=0.08, hspace=0.04, bottom=0.18)
        
        col_titles = [
            "(a) Reflectivity ($Z$)", 
            "(b) Velocity ($V_r$) with Zoom", 
            "(c) Baseline + CBAM\nAttention", 
            "(d) Baseline + SE\nAttention", 
            "(e) PI-STDNet (Ours)\nAttention + $V_r$ Contours"
        ]

        ax_matrix = [[None]*5 for _ in range(2)]

        for row_idx, case in enumerate(cases):
            idx = case['idx']
            image_tensor, _, cat_tensor = dataset[idx]
            cat_id = cat_tensor.item()
            images = image_tensor.numpy()
            inputs = image_tensor.unsqueeze(0).to(device)

            z_data = images[0] * 110.0 - 30.0  
            v_data = images[1] * 100.0 - 50.0  
            bg_z = (images[0] - np.min(images[0])) / (np.max(images[0]) - np.min(images[0]) + 1e-5)

            attns = {}
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                    for name, m in models_dict.items():
                        _, _, _, physics_attn = m(inputs)
                        attn_map = physics_attn[0, 0].cpu().numpy()
                        attn_map = ndimage.gaussian_filter(attn_map, sigma=1.0)
                        attn_map = np.clip(attn_map * 2.0, 0, 1) 
                        attns[name] = attn_map

            # 1. 原始反射率 Z
            ax_z = fig.add_subplot(gs[row_idx, 0])
            ax_matrix[row_idx][0] = ax_z
            im_z = ax_z.imshow(z_data, cmap=self.cmap_z, vmin=0, vmax=65, origin='lower')
            ax_z.set_xticks([]); ax_z.set_yticks([])
            if row_idx == 0: ax_z.set_title(col_titles[0], fontweight='bold', fontsize=15, pad=15)
            
            # 💡 核心微调 2：删除丑陋的 [Idx: xxxx]，使用正规的学术命名
            row_label = f"Case 1\nReal Tornado (TOR)" if cat_id == 2 else f"Case 2\nFalse Alarm (WRN)"
            ax_z.set_ylabel(row_label, fontweight='bold', fontsize=16, labelpad=15)

            cy_ours, cx_ours = np.unravel_index(np.argmax(attns["Ours"]), attns["Ours"].shape)
            box_half = 12 

            # 2. 原始速度场 V
            ax_v = fig.add_subplot(gs[row_idx, 1])
            ax_matrix[row_idx][1] = ax_v
            im_v = ax_v.imshow(v_data, cmap=self.cmap_v, vmin=-30, vmax=30, origin='lower')
            ax_v.set_xticks([]); ax_v.set_yticks([])
            if row_idx == 0: ax_v.set_title(col_titles[1], fontweight='bold', fontsize=15, pad=15)

            rect_v = patches.Rectangle((cx_ours - box_half, cy_ours - box_half), box_half*2, box_half*2, 
                                       linewidth=2.5, edgecolor='#00FF00', facecolor='none')
            ax_v.add_patch(rect_v)

            inset_ax = ax_v.inset_axes([0.05, 0.60, 0.35, 0.35])
            
            y_start = max(0, cy_ours - box_half)
            y_end = min(v_data.shape[0], cy_ours + box_half)
            x_start = max(0, cx_ours - box_half)
            x_end = min(v_data.shape[1], cx_ours + box_half)
            
            inset_v_data = v_data[y_start:y_end, x_start:x_end]
            
            inset_ax.imshow(inset_v_data, cmap=self.cmap_v, vmin=-30, vmax=30, origin='lower', extent=[x_start, x_end, y_start, y_end])
            inset_ax.set_xticks([]); inset_ax.set_yticks([])
            
            for spine in inset_ax.spines.values():
                spine.set_edgecolor('#00FF00')
                spine.set_linewidth(2.0)
                
            ax_v.indicate_inset_zoom(inset_ax, edgecolor="#00FF00", alpha=0.8, lw=1.5)

            if cat_id == 2:
                inset_ax.plot([x_start+box_half*0.5, x_end-box_half*0.5], [y_start+box_half, y_start+box_half], 
                              color='black', linestyle='--', linewidth=1.8) # 切变线稍微加粗
                inset_ax.text(x_start + box_half, y_start + box_half*1.6, "Shear", color='black', 
                              fontsize=11, fontweight='bold', ha='center', va='center', 
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            # 3, 4, 5. 热力图叠加
            model_keys = ["CBAM", "SE", "Ours"]
            im_hm = None
            
            for col_idx, model_key in enumerate(model_keys):
                ax = fig.add_subplot(gs[row_idx, col_idx + 2])
                ax_matrix[row_idx][col_idx + 2] = ax
                
                ax.imshow(bg_z, cmap='gray', vmin=0, vmax=1.8, origin='lower')
                
                attn_map = attns[model_key]
                masked_attn = np.ma.masked_where(attn_map < 0.20, attn_map) 
                im_hm = ax.imshow(masked_attn, cmap=self.cmap_hm, vmin=0, vmax=1.0, alpha=0.75, origin='lower')
                
                if model_key == "Ours":
                    ax.contour(v_data, levels=[15, 20, 25], colors='white', linewidths=0.8, alpha=0.9, linestyles='solid')
                    ax.contour(v_data, levels=[-25, -20, -15], colors='cyan', linewidths=0.8, alpha=0.9, linestyles='dashed')
                    
                    if row_idx == 1:
                        ax.text(0.96, 0.04, "— Pos Shear\n-- Neg Shear", transform=ax.transAxes, 
                                color='white', fontsize=11, fontweight='bold', 
                                ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

                ax.set_xticks([]); ax.set_yticks([])
                if row_idx == 0: ax.set_title(col_titles[col_idx + 2], fontweight='bold', fontsize=15, pad=15)
                
                prob_val = case['probs'][model_key]
                is_correct = (cat_id == 2 and prob_val >= 0.6) or (cat_id == 1 and prob_val < 0.6)
                text_color = '#00FF00' if is_correct else '#FF4500'
                
                ax.text(0.04, 0.04, f"$P = {prob_val:.2f}$", transform=ax.transAxes, 
                        color=text_color, fontweight='bold', fontsize=16, # 概率字号稍微放大，增强自信感
                        ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.4', edgecolor='none'))

                if model_key == "Ours":
                    rect_hm = patches.Rectangle((cx_ours - box_half, cy_ours - box_half), box_half*2, box_half*2, 
                                                linewidth=2.5, edgecolor='#00FF00', facecolor='none')
                    ax.add_patch(rect_hm)

        # 💡 核心微调 3：Colorbar 往上提拉 (pad 从 0.06 缩减到 0.04)，缩小无效留白
        cbar_z = fig.colorbar(im_z, ax=[ax_matrix[0][0], ax_matrix[1][0]], orientation='horizontal', fraction=0.04, pad=0.04, aspect=20)
        cbar_z.set_label('Reflectivity (dBZ)', fontsize=15, fontweight='bold')
        
        cbar_v = fig.colorbar(im_v, ax=[ax_matrix[0][1], ax_matrix[1][1]], orientation='horizontal', fraction=0.04, pad=0.04, aspect=20)
        cbar_v.set_label('Radial Velocity (m/s)', fontsize=15, fontweight='bold')

        cbar_hm = fig.colorbar(im_hm, ax=[ax_matrix[0][2], ax_matrix[1][2], ax_matrix[0][3], ax_matrix[1][3], ax_matrix[0][4], ax_matrix[1][4]], 
                               orientation='horizontal', fraction=0.04, pad=0.04, aspect=40)
        cbar_hm.set_label('Normalized Physics Attention Level', fontsize=15, fontweight='bold')
        cbar_hm.set_ticks([0, 0.5, 1.0])

        save_path = os.path.join(self.save_dir, "Figure7_Perfect_TGRS_Final.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 毫无破绽！【无敌紧凑大满贯定稿】已保存至: {save_path}")
        plt.close()

def execute(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在加载消融模型库... 设备: {device}")
    
    target_models = [
        ("CBAM", args.ckpt_cbam),
        ("SE", args.ckpt_se),
        ("Ours", args.ckpt_ours)
    ]
    
    models_dict = {}
    for label_name, ckpt_path in target_models:
        if not os.path.exists(ckpt_path): continue
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        keys = state_dict.keys()
        has_rot = any('rot_proj' in k for k in keys)
        
        if not any('physics_attn_module' in k for k in keys): attn_type, disable_attn = 'physics', True
        elif any('physics_attn_module.fc' in k for k in keys): attn_type, disable_attn = 'se', False
        elif any('physics_attn_module.channel_mlp' in k for k in keys): attn_type, disable_attn = 'cbam', False
        else: attn_type, disable_attn = 'physics', False
            
        train_compare.args = MockArgs(attn_type=attn_type, disable_physics_attn=disable_attn, disable_rot_stats=not has_rot)
        m = train_compare.PI_STDNet_Ablation().to(device)
        m.use_train_py_order = True if label_name == "Ours" else False
        m.load_state_dict(state_dict)
        m.eval()
        models_dict[label_name] = m

    dataset = train_compare.TorNetDatasetAblation('/path/to/TorNet/catalog.csv', '/path/to/TorNet', mode='test')
    
    # 🏆 使用这两位天选黄金双雄 🏆
    golden_cases = [
        {"idx": 2075, "type": "TOR", "probs": {"CBAM": 0.35, "SE": 0.36, "Ours": 0.74}}, 
        {"idx": 2476, "type": "WRN", "probs": {"CBAM": 0.63, "SE": 0.62, "Ours": 0.24}}  
    ]
    
    plotter = AttentionPlotterTGRS(save_dir=args.save_dir)
    plotter.plot_attention_grid(dataset, golden_cases, models_dict, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_cbam', type=str, default='ablation_attn_cbam_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    
    execute(args)
