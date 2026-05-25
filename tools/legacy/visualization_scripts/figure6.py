# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from tqdm import tqdm

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

class MatrixPlotterResearch:
    """Research 级：2x3 终极性能矩阵图 (主图画布全局扩容版)"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 14, # 字号也稍微跟进一点
            'figure.dpi': 300,
            'axes.linewidth': 1.5,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.top': True, 'ytick.right': True,
            'xtick.major.size': 6, 'ytick.major.size': 6,
            'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        })
        
        self.styles = {
            "Baseline (w/o)":    {"color": "#999999", "ls": ":",  "lw": 1.8},  
            "Baseline + CBAM":   {"color": "#2E8B57", "ls": "-.", "lw": 1.8},  
            "Baseline + SE":     {"color": "#4169E1", "ls": "--", "lw": 1.8},  
            "PI-STDNet (Ours)":  {"color": "#DC143C", "ls": "-",  "lw": 2.8}
        }

    def calc_metrics(self, probs, targets, thresh, mask):
        p = probs[mask]
        t = targets[mask]
        if len(t) == 0: return 0, 0, 0
        
        preds = (p >= thresh).astype(int)
        trues = (t == 2).astype(int) 
        
        TP = np.sum((preds == 1) & (trues == 1))
        FP = np.sum((preds == 1) & (trues == 0))
        FN = np.sum((preds == 0) & (trues == 1))
        
        CSI = TP / (TP + FP + FN + 1e-6)
        POD = TP / (TP + FN + 1e-6)
        FAR = FP / (TP + FP + 1e-6)
        return CSI, POD, FAR

    def plot_matrix(self, probs_dict, targets):
        thresholds = np.arange(0.1, 0.95, 0.05)
        
        masks = {
            "NC": np.ones_like(targets, dtype=bool),
            "WC": (targets == 1) | (targets == 2)
        }

        # ==========================================================
        # 🚀 核心修复：把整张大画布直接撑开到 24x14，彻底释放空间！
        # ==========================================================
        fig, axes = plt.subplots(2, 3, figsize=(24, 14), gridspec_kw={'wspace': 0.22, 'hspace': 0.30})
        
        metrics = ['CSI', 'POD', 'FAR']
        metric_titles = ['Critical Success Index (CSI)', 'Probability of Detection (POD)', 'False Alarm Ratio (FAR)']
        optimal_thresh = 0.60
        
        for row_idx, view in enumerate(["NC", "WC"]):
            mask = masks[view]
            
            results = {name: {'CSI': [], 'POD': [], 'FAR': []} for name in probs_dict.keys()}
            for t in thresholds:
                for name, p in probs_dict.items():
                    csi, pod, far = self.calc_metrics(p, targets, t, mask)
                    results[name]['CSI'].append(csi)
                    results[name]['POD'].append(pod)
                    results[name]['FAR'].append(far)

            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.3, zorder=0)
                
                all_y = [v for name in results for v in results[name][metric]]
                y_min, y_max = min(all_y), max(all_y)
                y_range = y_max - y_min
                
                ax.set_ylim(max(0, y_min - y_range * 0.05), y_max + y_range * 0.15)
                ax.set_xlim(0.05, 0.95)

                ax.axvline(x=optimal_thresh, color='#666666', linestyle='-.', lw=1.2, alpha=0.6, zorder=1)
                
                if metric == 'CSI':
                    vert_x, vert_ha = optimal_thresh - 0.02, 'right'
                else:
                    vert_x, vert_ha = optimal_thresh + 0.02, 'left'
                    
                ax.text(vert_x, ax.get_ylim()[0] + y_range * 0.03, "Optimal Threshold (0.6)", 
                        rotation=90, va='bottom', ha=vert_ha, color='#666666', 
                        fontsize=12, fontweight='bold', zorder=5)

                for name, style in self.styles.items():
                    if name not in results: continue
                    y_vals = results[name][metric]
                    ax.plot(thresholds, y_vals, label=name, zorder=3 if "Ours" not in name else 4,
                            color=style['color'], linestyle=style['ls'], linewidth=style['lw'])
                    
                    idx_opt = np.where(np.isclose(thresholds, optimal_thresh))[0][0]
                    target_val = y_vals[idx_opt]

                    if name == "PI-STDNet (Ours)":
                        ax.plot(optimal_thresh, target_val, marker='*', color=style['color'], 
                                markersize=15, markeredgecolor='white', markeredgewidth=0.8, zorder=10)
                        
                        if metric == 'CSI':
                            xy_text = (optimal_thresh + 0.08, target_val - y_range * 0.05)
                            conn_style = "arc3,rad=-0.1"
                        else:
                            xy_text = (optimal_thresh - 0.15, target_val - y_range * 0.10)
                            conn_style = "arc3,rad=-0.2"
                            
                        ax.annotate(f"{target_val:.4f}", xy=(optimal_thresh, target_val), xytext=xy_text,
                                    arrowprops=dict(arrowstyle="->", color='#DC143C', lw=1.2, connectionstyle=conn_style),
                                    fontsize=14, fontweight='bold', color='#DC143C', zorder=15)
                    else:
                        ax.plot(optimal_thresh, target_val, marker='o', color=style['color'], markersize=6, zorder=5)

                # ==========================================================
                # 🎯 画中画恢复正常比例：因为外围画布大了，它自然就宽敞了，绝不重叠
                # ==========================================================
                if metric == 'CSI':
                    # 退回经典的 30% 占比，紧贴左上角
                    inset_bounds = [0.06, 0.58, 0.30, 0.38] 
                else:
                    # 紧贴右上角
                    inset_bounds = [0.64, 0.58, 0.30, 0.38]

                axins = ax.inset_axes(inset_bounds)
                axins.grid(True, linestyle='--', alpha=0.3)
                
                peak_idx = np.where(np.isclose(thresholds, optimal_thresh))[0][0]
                vals_at_06 = [results[n][metric][peak_idx] for n in self.styles.keys() if n in results]
                min_peak, max_peak = min(vals_at_06), max(vals_at_06)
                margin = max((max_peak - min_peak) * 0.4, 0.02)
                
                axins.set_xlim(0.57, 0.63) 
                
                if metric == 'CSI' or metric == 'POD': 
                    axins.set_ylim(min_peak - margin, max_peak + margin * 1.5)
                else: 
                    axins.set_ylim(min_peak - margin * 1.5, max_peak + margin)
                
                axins.set_xticks([0.58, 0.60, 0.62])
                axins.yaxis.set_major_locator(ticker.MaxNLocator(4))
                axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) 
                
                if metric == 'CSI':
                    axins.yaxis.tick_right()
                
                axins.tick_params(labelsize=11, pad=3) 
                
                for name, style in self.styles.items():
                    if name not in results: continue
                    y_vals = results[name][metric]
                    axins.plot(thresholds, y_vals, color=style['color'], linestyle=style['ls'], linewidth=style['lw'])
                    val = y_vals[peak_idx]
                    if name == "PI-STDNet (Ours)":
                        axins.plot(optimal_thresh, val, marker='*', color=style['color'], markersize=14, markeredgecolor='white', markeredgewidth=0.8, zorder=10)
                    else:
                        axins.plot(optimal_thresh, val, marker='o', color=style['color'], markersize=5, zorder=5)
                
                ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5, lw=1.5)

                letter = chr(97 + row_idx * 3 + col_idx)
                view_desc = "NC (All Samples)" if view == "NC" else "WC (Hard Samples: TOR+WRN)"
                ax.set_title(f"({letter}) {view_desc}\n{metric_titles[col_idx]}", fontweight='bold', fontsize=16, pad=12)
                ax.set_xlabel("Decision Threshold", fontweight='bold', fontsize=14)
                ax.set_ylabel(metric, fontweight='bold', fontsize=14)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), 
                   frameon=False, fontsize=16, title_fontsize=16)

        save_path = os.path.join(self.save_dir, "Figure6_Matrix_NC_WC_TrueScale.pdf")
        plt.subplots_adjust(bottom=0.1) 
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 恍然大悟！【主图全局扩容版】已保存至: {save_path}")
        plt.close()

def extract_and_plot(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在加载 4 个模型的权重... 设备: {device}")
    
    target_models = [
        ("Baseline (w/o)", args.ckpt_base),
        ("Baseline + CBAM", args.ckpt_cbam),
        ("Baseline + SE", args.ckpt_se),
        ("PI-STDNet (Ours)", args.ckpt_ours)
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
        m.use_train_py_order = True if "Ours" in label_name else False
        m.load_state_dict(state_dict)
        m.eval()
        models_dict[label_name] = m

    dataset = train_compare.TorNetDatasetAblation('/path/to/TorNet/catalog.csv', '/path/to/TorNet', mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

    probs_collected = {name: [] for name in models_dict.keys()}
    labels_list = []

    print("\n🔍 正在全速遍历测试集计算概率 (捕获分类掩码)...")
    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning Testset"):
            images = images.to(device)
            cats = cats.numpy() 
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                for name, m in models_dict.items():
                    cls_logit, aux_logit, _, _ = m(images)
                    prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).cpu().numpy()
                    probs_collected[name].extend(prob)
            labels_list.extend(cats)

    targets = np.array(labels_list)
    
    for name in probs_collected:
        probs_collected[name] = np.array(probs_collected[name])
        
    plotter = MatrixPlotterResearch(save_dir=args.save_dir)
    plotter.plot_matrix(probs_collected, targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_base', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_cbam', type=str, default='ablation_attn_cbam_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    
    extract_and_plot(args)


