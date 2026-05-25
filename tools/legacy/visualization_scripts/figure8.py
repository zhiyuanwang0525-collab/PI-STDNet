# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 拦截 sys.argv 防报错
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  
import train_compare
sys.argv = original_argv  

class MockArgs:
    def __init__(self, attn_type='physics', disable_physics_attn=False, disable_rot_stats=False, disable_topk=True):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = False
        self.disable_topk = disable_topk
        self.disable_rot_stats = disable_rot_stats 
        
class CurvePlotterResearch:
    """Research 级：ROC(带局部放大) & PR 黄金三角版"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 14,
            'figure.dpi': 300,
            'axes.linewidth': 1.5,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.top': True, 'ytick.right': True,
            'xtick.major.size': 6, 'ytick.major.size': 6,
            'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        })
        
        self.styles = {
            "Baseline (w/o Physics Inputs)": {"color": "#999999", "ls": ":",  "lw": 2.2},  
            "w/o Physics Attn":              {"color": "#4682B4", "ls": "--", "lw": 2.2},  
            "PI-STDNet (Ours)":              {"color": "#DC143C", "ls": "-",  "lw": 3.0} 
        }

    def plot_curves(self, y_true, y_scores_dict):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'wspace': 0.22})
        ax_roc, ax_pr = axes[0], axes[1]
        
        ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.6)
        
        # ==========================================================
        # 🎯 核心升级：在 ROC 图中央偏下位置创建一个画中画 (Inset)
        # ==========================================================
        axins = ax_roc.inset_axes([0.42, 0.28, 0.50, 0.40])
        axins.grid(True, linestyle='--', alpha=0.3)
        axins.tick_params(labelsize=11)
        
        for name, style in self.styles.items():
            if name not in y_scores_dict:
                continue
                
            y_scores = y_scores_dict[name]
            
            # 1. 主 ROC 曲线与 AUC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            label_roc = f'{name} (AUC = {roc_auc:.4f})'
            ax_roc.plot(fpr, tpr, color=style['color'], linestyle=style['ls'], 
                        linewidth=style['lw'], label=label_roc, zorder=10 if "Ours" in name else 5)
            
            # 🎯 同时把曲线画到放大镜 (Inset) 里
            axins.plot(fpr, tpr, color=style['color'], linestyle=style['ls'], 
                       linewidth=style['lw'], zorder=10 if "Ours" in name else 5)
            
            # 2. PR 曲线与 AP
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_ap = average_precision_score(y_true, y_scores)
            label_pr = f'{name} (AP = {pr_ap:.4f})'
            ax_pr.plot(recall, precision, color=style['color'], linestyle=style['ls'], 
                       linewidth=style['lw'], label=label_pr, zorder=10 if "Ours" in name else 5)

        # 🎯 设置放大镜的观察范围：聚焦在 FPR 极低的核心区域
        axins.set_xlim(0.01, 0.25)
        axins.set_ylim(0.60, 0.88)
        # 自动画出从主图到放大镜的连接线
        ax_roc.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6, lw=1.2)

        # ROC 格式化
        ax_roc.set_xlim([-0.02, 1.02]); ax_roc.set_ylim([-0.02, 1.02])
        ax_roc.set_xlabel('False Positive Rate (FPR)', fontweight='bold', fontsize=15)
        ax_roc.set_ylabel('True Positive Rate (TPR)', fontweight='bold', fontsize=15)
        ax_roc.set_title('(a) Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=16, pad=15)
        ax_roc.grid(True, linestyle='--', alpha=0.3)
        ax_roc.legend(loc="lower right", fontsize=12, framealpha=0.9, edgecolor='black')

        # PR 格式化
        ax_pr.set_xlim([-0.02, 1.02]); ax_pr.set_ylim([-0.02, 1.02])
        ax_pr.set_xlabel('Recall (True Positive Rate)', fontweight='bold', fontsize=15)
        ax_pr.set_ylabel('Precision', fontweight='bold', fontsize=15)
        ax_pr.set_title('(b) Precision-Recall (PR) Curve', fontweight='bold', fontsize=16, pad=15)
        ax_pr.grid(True, linestyle='--', alpha=0.3)
        ax_pr.legend(loc="lower left", fontsize=12, framealpha=0.9, edgecolor='black')

        save_path = os.path.join(self.save_dir, "Figure8_Ablation_ROC_PR_Zoomed.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 显微镜已部署！【带局部放大的 ROC/PR 图】已保存至: {save_path}")
        plt.close()

def execute(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在加载消融模型库... 设备: {device}")
    
    target_models = [
        ("Baseline (w/o Physics Inputs)", args.ckpt_no_inputs),
        ("w/o Physics Attn", args.ckpt_no_attn), 
        ("PI-STDNet (Ours)", args.ckpt_ours)
    ]
    
    models_dict = {}
    for label_name, ckpt_path in target_models:
        if not os.path.exists(ckpt_path): 
            print(f"⚠️ 未找到权重文件 {ckpt_path}，跳过模型 [{label_name}]")
            continue
        
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        keys = state_dict.keys()
        
        has_rot = any('rot_proj' in k for k in keys)
        has_topk = any('topk_pool' in k for k in keys)
        
        disable_inputs = True if "w/o Physics Inputs" in label_name else False
        disable_attn = True if ("w/o Physics Attn" in label_name or disable_inputs) else False
            
        train_compare.args = MockArgs(
            attn_type='physics', 
            disable_physics_attn=disable_attn,
            disable_rot_stats=not has_rot,
            disable_topk=not has_topk
        )
        
        train_compare.CONFIG['CHANNELS_PER_FRAME'] = 6 if disable_inputs else 11
        
        m = train_compare.PI_STDNet_Ablation().to(device)
        m.use_train_py_order = True if "Ours" in label_name else False
        m.load_state_dict(state_dict, strict=False)
        m.eval()
        models_dict[label_name] = m

    if len(models_dict) == 0:
        print("❌ 没有找到任何模型权重，请检查路径。")
        return

    train_compare.CONFIG['CHANNELS_PER_FRAME'] = 11 
    dataset = train_compare.TorNetDatasetAblation('/path/to/TorNet/catalog.csv', '/path/to/TorNet', mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

    probs_collected = {name: [] for name in models_dict.keys()}
    targets_list = []

    print("\n🔍 正在全速遍历测试集计算全域概率...")
    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning Testset"):
            images = images.to(device)
            true_labels = (cats.numpy() == 2).astype(int) 
            targets_list.extend(true_labels)
            
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                for name, m in models_dict.items():
                    if "w/o Physics Inputs" in name:
                        selected_channels = []
                        for s in range(train_compare.CONFIG['N_SWEEPS']):
                            for t in range(train_compare.CONFIG['N_FRAMES']):
                                base = s * (train_compare.CONFIG['N_FRAMES'] * 11) + t * 11
                                selected_channels.append(images[:, base:base+6, :, :])
                        input_img = torch.cat(selected_channels, dim=1)
                    else:
                        input_img = images

                    cls_logit, aux_logit, _, _ = m(input_img)
                    prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).cpu().numpy()
                    probs_collected[name].extend(prob)

    y_true = np.array(targets_list)
    y_scores_dict = {name: np.array(probs) for name, probs in probs_collected.items()}
    
    plotter = CurvePlotterResearch(save_dir=args.save_dir)
    plotter.plot_curves(y_true, y_scores_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_no_inputs', type=str, default='ablation_no_physics_input_best.pth')
    parser.add_argument('--ckpt_no_attn', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP.pth') 
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    
    execute(args)


