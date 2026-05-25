# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 拦截 sys.argv 防止 import 报错
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  
import train_compare
sys.argv = original_argv  

class MockArgs:
    def __init__(self, attn_type='physics', disable_physics_attn=False, disable_rot_stats=False):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = False
        self.disable_topk = True  # 契合你们最新的最大池化
        self.disable_rot_stats = disable_rot_stats

def calc_metrics(probs, labels, mask, thresh):
    if mask.sum() == 0: return 0, 0, 0
    p, l = probs[mask], labels[mask]
    preds = (p >= thresh).astype(int)
    
    tp = ((preds == 1) & (l == 1)).sum()
    fp = ((preds == 1) & (l == 0)).sum()
    fn = ((preds == 0) & (l == 1)).sum()
    
    csi = tp / (tp + fn + fp + 1e-7)
    pod = tp / (tp + fn + 1e-7)
    far = fp / (tp + fp + 1e-7)
    return csi, pod, far

class MetricsCurvePlotter:
    """Research 级：1x3 阈值演化性能评估曲线"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 14,
            'figure.dpi': 300,
            'axes.linewidth': 1.5
        })

    def plot_1x3_curves(self, results_dict, thresholds):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'wspace': 0.25})
        
        styles = {
            "Baseline (w/o Attention)": {"color": "#999999", "ls": "--", "marker": "o", "lw": 2, "zorder": 1},
            "Baseline + SE Module":     {"color": "#4169E1", "ls": "-.", "marker": "s", "lw": 2, "zorder": 2},
            "Baseline + CBAM Module":   {"color": "#3CB371", "ls": ":",  "marker": "^", "lw": 2, "zorder": 3},
            "PI-STDNet (Ours)":         {"color": "#DC143C", "ls": "-",  "marker": "*", "lw": 3.5, "zorder": 10, "markersize": 10}
        }

        titles = ["(a) Critical Success Index (CSI)", "(b) Probability of Detection (POD)", "(c) False Alarm Ratio (FAR)"]
        metrics_keys = ["csi", "pod", "far"]
        y_labels = ["CSI (Higher is better)", "POD (Higher is better)", "FAR (Lower is better)"]

        for i, ax in enumerate(axes):
            metric = metrics_keys[i]
            for model_name, data in results_dict.items():
                sty = styles[model_name]
                ax.plot(thresholds, data[metric], label=model_name, 
                        color=sty["color"], linestyle=sty["ls"], linewidth=sty["lw"], 
                        marker=sty["marker"], markersize=sty.get("markersize", 6), zorder=sty["zorder"])
            
            # 植入最优阈值 0.6 的参考线
            ax.axvline(x=0.6, color='#DC143C', linestyle='-', linewidth=1.5, alpha=0.5, zorder=0)
            if i == 0:
                ax.text(0.62, ax.get_ylim()[1]*0.85, "Optimal\nThreshold\n(0.6)", color='#DC143C', fontweight='bold', fontsize=13)

            ax.set_title(titles[i], pad=15, fontweight='bold', fontsize=16)
            ax.set_xlabel("Decision Threshold", fontweight='bold', fontsize=14)
            ax.set_ylabel(y_labels[i], fontweight='bold', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0.08, 0.92)
            
            for spine in ax.spines.values():
                spine.set_color('#555555')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08), 
                   frameon=True, fontsize=14, edgecolor='black', framealpha=1.0)

        save_path = os.path.join(self.save_dir, "Figure6_Metrics_Threshold_Curves.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 绝杀！极具说服力的 1x3 评估曲线已保存至: {save_path}")
        plt.close()

def extract_and_plot(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在初始化概率提取流水线... 设备: {device}")
    
    target_models = [
        ("Baseline (w/o Attention)", args.ckpt_base),
        ("Baseline + SE Module", args.ckpt_se),
        ("Baseline + CBAM Module", args.ckpt_cbam),
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
        m.load_state_dict(state_dict)
        m.eval()
        models_dict[label_name] = m

    # 💡 核心修复：num_workers 强制设为 0，防止 Windows 多进程导致的数据错乱！
    dataset = train_compare.TorNetDatasetAblation(train_compare.CONFIG['CATALOG_PATH'], train_compare.CONFIG['DATA_ROOT'], mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    probs_collected = {name: [] for name in models_dict.keys()}
    labels_list, cats_list = [], []

    print("\n🔍 正在单进程严谨遍历测试集，采集真实输出概率...")
    with torch.no_grad():
        for images, labels, cats in tqdm(dataloader, desc="Scanning"):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                for name, m in models_dict.items():
                    cls_logit, aux_logit, _, _ = m(images)
                    prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).cpu().numpy()
                    probs_collected[name].extend(prob)
            
            labels_list.extend(labels.numpy())
            cats_list.extend(cats.numpy())

    targets = np.array(labels_list)
    cats = np.array(cats_list)
    mask_wc = (cats == 1) | (cats == 2)

    print("\n⚔️ 正在计算全面性能指标...")
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = {name: {"csi": [], "pod": [], "far": []} for name in models_dict.keys()}

    for name in models_dict.keys():
        probs = np.array(probs_collected[name])
        for t in thresholds:
            csi, pod, far = calc_metrics(probs, targets, mask_wc, t)
            results[name]["csi"].append(csi)
            results[name]["pod"].append(pod)
            results[name]["far"].append(far)

    plotter = MetricsCurvePlotter(save_dir=args.save_dir)
    plotter.plot_1x3_curves(results, thresholds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_base', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_cbam', type=str, default='ablation_attn_cbam_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures2')
    args = parser.parse_args()
    
    extract_and_plot(args)


