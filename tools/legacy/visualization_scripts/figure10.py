# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

class ReliabilityPlotterTGRS:
    """TGRS 级：概率可靠性图 (Reliability Diagram / Calibration Curve)"""
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
            'xtick.major.size': 6, 'ytick.major.width': 1.2,
            'ytick.major.size': 6, 'ytick.major.width': 1.2,
        })
        
        self.styles = {
            "Baseline (w/o Physics Inputs)": {"color": "#999999", "ls": "--", "lw": 2.5, "marker": "o", "markersize": 7},  
            "w/o Physics Attn":              {"color": "#4682B4", "ls": "-.", "lw": 2.5, "marker": "s", "markersize": 7},  
            "PI-STDNet (Ours)":              {"color": "#DC143C", "ls": "-",  "lw": 3.5, "marker": "D", "markersize": 8} 
        }

    def plot_reliability(self, y_true, y_scores_dict):
        # 气象界标准的画法：主图是 Reliability Curve，有时附带一个预测概率分布图
        fig, ax = plt.subplots(figsize=(9, 9)) # 故意用正方形，凸显 y=x 的对角线
        
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        
        # 1. 绘制完美校准线 (y = x)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", zorder=1, alpha=0.7)
        
        # 2. 遍历并绘制模型校准曲线
        for name, style in self.styles.items():
            if name not in y_scores_dict:
                continue
            
            y_prob = y_scores_dict[name]
            
            # 计算 Brier Score (BS)，分数越低越好
            bs_score = brier_score_loss(y_true, y_prob)
            
            # 使用 sklearn 的 calibration_curve 按照预测概率将样本分箱计算观测频率
            # strategy='quantile' (等样本量分箱) 通常比 'uniform' (等距分箱) 在极度不平衡数据上更稳定
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10, strategy='quantile'
            )
            
            label_text = f"{name} (BS = {bs_score:.4f})"
            
            ax.plot(mean_predicted_value, fraction_of_positives, 
                    color=style['color'], linestyle=style['ls'], linewidth=style['lw'], 
                    marker=style['marker'], markersize=style['markersize'], 
                    label=label_text, zorder=10 if "Ours" in name else 5,
                    markeredgecolor='white', markeredgewidth=1.0)

        # 轴向限制与标签格式化
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=16, labelpad=12)
        ax.set_ylabel('Observed Fraction of Positives', fontweight='bold', fontsize=16, labelpad=12)
        
        ax.set_title('Figure 10: Reliability Diagram (Calibration)', fontweight='bold', fontsize=18, pad=15)
        
        # 图例放置在左上角 (通常对角线图的左上角是空出来的)
        ax.legend(loc="upper left", fontsize=13, framealpha=0.9, edgecolor='black', borderpad=0.8)

        save_path = os.path.join(self.save_dir, "Figure10_Reliability_Diagram.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 业务级信誉背书！【概率可靠性校准图】已保存至: {save_path}")
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
        if not os.path.exists(ckpt_path): continue
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        keys = state_dict.keys()
        
        has_rot = any('rot_proj' in k for k in keys)
        has_topk = any('topk_pool' in k for k in keys)
        disable_inputs = True if "w/o Physics Inputs" in label_name else False
        disable_attn = True if ("w/o Physics Attn" in label_name or disable_inputs) else False
            
        train_compare.args = MockArgs(attn_type='physics', disable_physics_attn=disable_attn, 
                                      disable_rot_stats=not has_rot, disable_topk=not has_topk)
        train_compare.CONFIG['CHANNELS_PER_FRAME'] = 6 if disable_inputs else 11
        
        m = train_compare.PI_STDNet_Ablation().to(device)
        m.use_train_py_order = True if "Ours" in label_name else False
        m.load_state_dict(state_dict, strict=False)
        m.eval()
        models_dict[label_name] = m

    if not models_dict: return print("❌ 缺少模型权重！")

    # =========================================================
    # 🎯 全量加载测试集 (无需分类物理距离)
    # =========================================================
    catalog_path = '/path/to/TorNet/catalog.csv'
    train_compare.CONFIG['CHANNELS_PER_FRAME'] = 11 
    dataset = train_compare.TorNetDatasetAblation(catalog_path, '/path/to/TorNet', mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

    probs_collected = {name: [] for name in models_dict.keys()}
    targets_list = []

    print("\n🔍 正在全速遍历测试集收集全部预测概率...")
    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning Testset"):
            images = images.to(device)
            # 同样：龙卷 TOR(2) 为 1，其他(WRN/NUL) 为 0
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

    print("\n📊 正在执行概率校准计算与绘图...")
    plotter = ReliabilityPlotterTGRS(save_dir=args.save_dir)
    plotter.plot_reliability(y_true, y_scores_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_no_inputs', type=str, default='ablation_no_physics_input_best.pth')
    parser.add_argument('--ckpt_no_attn', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth') 
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    
    execute(args)
