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

class RangeRobustnessPlotterResearch:
    """Research 级：真实雷达探测距离衰减曲线"""
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
            'xtick.top': False, 'ytick.right': False,
            'xtick.major.size': 6, 'ytick.major.width': 1.2,
            'ytick.major.size': 6, 'ytick.major.width': 1.2,
        })
        
        self.styles = {
            "Baseline (w/o Physics Inputs)": {"color": "#999999", "ls": "--", "lw": 2.5, "marker": "o", "markersize": 8},  
            "w/o Physics Attn":              {"color": "#4682B4", "ls": "-.", "lw": 2.5, "marker": "s", "markersize": 8},  
            "PI-STDNet (Ours)":              {"color": "#DC143C", "ls": "-",  "lw": 3.5, "marker": "D", "markersize": 9} 
        }

    def plot_robustness(self, range_bins, data_dict):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        
        x_indices = np.arange(len(range_bins))
        
        for name, style in self.styles.items():
            if name not in data_dict:
                continue
                
            scores = data_dict[name]
            ax.plot(x_indices, scores, color=style['color'], linestyle=style['ls'], 
                    linewidth=style['lw'], marker=style['marker'], markersize=style['markersize'], 
                    label=name, zorder=10 if "Ours" in name else 5, 
                    markeredgecolor='white', markeredgewidth=1.2)
            
            # 在最后一个点标注具体数值
            last_score = scores[-1]
            text_color = style['color']
            weight = 'bold' if "Ours" in name else 'normal'
            y_offset = 0.015 if "Ours" in name else -0.02
            ax.text(x_indices[-1], last_score + y_offset, f"{last_score:.3f}", 
                    color=text_color, fontweight=weight, fontsize=13, ha='center', zorder=15)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(range_bins, fontweight='bold', fontsize=14)
        ax.set_xlabel('Distance from Radar (km)', fontweight='bold', fontsize=15, labelpad=10)
        
        # 动态设置 Y 轴
        all_scores = [val for lst in data_dict.values() for val in lst]
        y_min, y_max = min(all_scores) - 0.05, max(all_scores) + 0.05
        ax.set_ylim([max(0, y_min), min(1.0, y_max)]) 
        
        ax.set_ylabel('Critical Success Index (CSI)', fontweight='bold', fontsize=15, labelpad=10)
        ax.set_title('Figure 9: Performance Robustness across Radar Ranges', fontweight='bold', fontsize=16, pad=15)
        ax.legend(loc="lower left", fontsize=13, framealpha=0.9, edgecolor='black', borderpad=0.8)

        save_path = os.path.join(self.save_dir, "Figure9_Range_Robustness.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 数据驱动图表生成！【探测距离鲁棒性真实折线图】已保存至: {save_path}")
        plt.close()

def evaluate_csi_for_bin(probs, labels, thresh=0.6):
    """计算单个区间段内的 CSI (阈值固定为业务常用的 0.6)"""
    if len(probs) == 0: return 0.0
    preds = (probs > thresh).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    if (tp + fp + fn) == 0: return 0.0
    return tp / (tp + fn + fp)

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
    # 🎯 提取真实数据的距离信息 (已替换为绝对路径)
    # =========================================================
    catalog_path = '/path/to/TorNet/catalog.csv'
    print(f"📖 正在从 {catalog_path} 中提取距离元数据...")
    df = pd.read_csv(catalog_path)
    test_df = df[df['type'] == 'test'].copy().reset_index(drop=True)
    
    # 计算中心距离 (公里)
    test_df['center_range_km'] = (test_df['rng_min'] + test_df['rng_max']) / 2.0 / 1000.0

    train_compare.CONFIG['CHANNELS_PER_FRAME'] = 11 
    # 🎯 数据集加载器也替换为了绝对路径
    dataset = train_compare.TorNetDatasetAblation(catalog_path, '/path/to/TorNet', mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

    probs_collected = {name: [] for name in models_dict.keys()}
    targets_list = []

    print("\n🔍 正在全速遍历测试集计算预测概率...")
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

    # =========================================================
    # 🎯 核心逻辑：按雷达距离分箱 (Binning) 并计算 CSI
    # =========================================================
    print("\n📊 正在执行雷达波束展宽(范围)鲁棒性分析...")
    bins = [0, 40, 80, 120, float('inf')]
    bin_labels = ['0-40', '40-80', '80-120', '> 120']
    
    test_df['range_bin'] = pd.cut(test_df['center_range_km'], bins=bins, labels=bin_labels, right=False)
    
    robustness_results = {name: [] for name in models_dict.keys()}
    
    for label in bin_labels:
        idx_in_bin = test_df.index[test_df['range_bin'] == label].tolist()
        y_true_bin = np.array([targets_list[i] for i in idx_in_bin])
        
        for name in models_dict.keys():
            y_prob_bin = np.array([probs_collected[name][i] for i in idx_in_bin])
            csi = evaluate_csi_for_bin(y_prob_bin, y_true_bin, thresh=0.6)
            robustness_results[name].append(csi)
            
        print(f"  📍 区间 [{label} km]: 样本数 = {len(idx_in_bin)} (其中龙卷: {sum(y_true_bin)})")

    plotter = RangeRobustnessPlotterResearch(save_dir=args.save_dir)
    plotter.plot_robustness(bin_labels, robustness_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_no_inputs', type=str, default='ablation_no_physics_input_best.pth')
    parser.add_argument('--ckpt_no_attn', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP.pth') 
    parser.add_argument('--save_dir', type=str, default='paper_figures_final')
    args = parser.parse_args()
    
    execute(args)


