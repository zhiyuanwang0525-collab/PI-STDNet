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

class DecisionSpacePlotterTGRS:
    """TGRS 级：极简克制、学术质感拉满的 0.6 决胜散点图"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 严格的 IEEE 顶刊绘图参数
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 14,
            'figure.dpi': 300,
            'axes.linewidth': 1.5,
            'xtick.direction': 'in',       # 刻度朝内
            'ytick.direction': 'in',       # 刻度朝内
            'xtick.top': True,             # 顶部显示刻度
            'ytick.right': True,           # 右侧显示刻度
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })
        self.color_wrn = '#4169E1' # 皇家蓝 
        self.color_tor = '#DC143C' # 猩红 
        self.thresh = 0.60         

    def plot_decision_shift(self, baseline_probs, ours_probs, targets):
        fig, ax = plt.subplots(figsize=(9, 9))
        
        mask_wrn = (targets == 1)
        mask_tor = (targets == 2)

        # 核心样本筛选
        saved_tor_mask = mask_tor & (baseline_probs < self.thresh) & (ours_probs >= self.thresh)
        suppressed_wrn_mask = mask_wrn & (baseline_probs >= self.thresh) & (ours_probs < self.thresh)
        bg_tor_mask = mask_tor & ~saved_tor_mask
        bg_wrn_mask = mask_wrn & ~suppressed_wrn_mask

        # ==========================================================
        # 🎨 1. 区域晕染 (Victory Zone Shading) - 极具学术高级感
        # ==========================================================
        # 虚警镇压区 (右下角) - 极浅的蓝色
        ax.fill_between([self.thresh, 1.05], -0.05, self.thresh, color=self.color_wrn, alpha=0.06, zorder=0)
        # 漏报挽救区 (左上角) - 极浅的红色
        ax.fill_between([-0.05, self.thresh], self.thresh, 1.05, color=self.color_tor, alpha=0.06, zorder=0)

        # ==========================================================
        # 🎨 2. 轴线与基准线
        # ==========================================================
        ax.plot([-0.1, 1.1], [-0.1, 1.1], color='#888888', linestyle=':', linewidth=1.5, zorder=1)
        ax.axvline(x=self.thresh, color='#333333', linestyle='--', linewidth=1.8, zorder=2)
        ax.axhline(y=self.thresh, color='#333333', linestyle='--', linewidth=1.8, zorder=2)

        # ==========================================================
        # 🎨 3. 数据映射：底噪极限压制，高光极致凸显
        # ==========================================================
        # 背景点：进一步降低透明度，让图面更干净
        ax.scatter(baseline_probs[bg_wrn_mask], ours_probs[bg_wrn_mask], 
                   c=self.color_wrn, marker='o', s=12, alpha=0.08, edgecolors='none', zorder=3)
        ax.scatter(baseline_probs[bg_tor_mask], ours_probs[bg_tor_mask], 
                   c=self.color_tor, marker='*', s=20, alpha=0.08, edgecolors='none', zorder=3)

        # 决胜点：保留冲击力
        ax.scatter(baseline_probs[suppressed_wrn_mask], ours_probs[suppressed_wrn_mask], 
                   c=self.color_wrn, marker='o', s=70, alpha=0.9, 
                   edgecolors='white', linewidths=0.7, zorder=4,
                   label='WRN (Suppressed False Alarms)')

        ax.scatter(baseline_probs[saved_tor_mask], ours_probs[saved_tor_mask], 
                   c=self.color_tor, marker='*', s=180, alpha=0.95, 
                   edgecolors='white', linewidths=0.8, zorder=5,
                   label='TOR (Captured Missed Tornadoes)')

        # ==========================================================
        # 🎨 4. 极简学术文本标注 (摒弃大彩框)
        # ==========================================================
        bbox_style = dict(boxstyle="square,pad=0.4", fc="white", ec="#AAAAAA", lw=1.0, alpha=0.95)

        ax.text(0.85, 0.25, "Region of False Alarm\nSuppression", 
                ha='center', va='center', fontsize=13, fontweight='bold', color='#222222',
                bbox=bbox_style, zorder=6)
        
        ax.text(0.25, 0.85, "Region of Improved\nDetection", 
                ha='center', va='center', fontsize=13, fontweight='bold', color='#222222',
                bbox=bbox_style, zorder=6)

        ax.text(self.thresh + 0.015, 0.03, f"Decision Threshold ({self.thresh})", 
                color='#333333', fontweight='bold', fontsize=12)

        # ==========================================================
        # 🎨 5. 坐标轴与排版清理 (移除图纸大标题)
        # ==========================================================
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Baseline (SE Module) Predicted Probability", fontweight='bold', fontsize=15)
        ax.set_ylabel("PI-STDNet (Ours) Predicted Probability", fontweight='bold', fontsize=15)
        
        # 强制设置主刻度，显得极其严谨
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # 图例放置在极净区，去除圆角
        legend = ax.legend(loc='lower left', framealpha=1.0, fontsize=12, edgecolor='#AAAAAA')
        legend.get_frame().set_boxstyle("square,pad=0.2")

        save_path = os.path.join(self.save_dir, "Figure5_Decision_Scatter_TGRS.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 绝杀！【TGRS 级极简定稿版】四象限散点图已保存至: {save_path}")
        plt.close()

def extract_and_plot(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在初始化概率提取流水线... 设备: {device}")
    
    target_models = [("Best Baseline", args.ckpt_se), ("PI-STDNet (Ours)", args.ckpt_ours)]
    models_dict = {}
    
    for label_name, ckpt_path in target_models:
        if not os.path.exists(ckpt_path): continue
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        keys = state_dict.keys()
        has_rot = any('rot_proj' in k for k in keys)
        
        if any('physics_attn_module.fc' in k for k in keys): attn_type, disable_attn = 'se', False
        else: attn_type, disable_attn = 'physics', False
            
        train_compare.args = MockArgs(attn_type=attn_type, disable_physics_attn=disable_attn, disable_rot_stats=not has_rot)
        m = train_compare.PI_STDNet_Ablation().to(device)
        
        if "Ours" in label_name: m.use_train_py_order = True
        else: m.use_train_py_order = False

        m.load_state_dict(state_dict)
        m.eval()
        models_dict[label_name] = m

    # 全速提取
    dataset = train_compare.TorNetDatasetAblation(train_compare.CONFIG['CATALOG_PATH'], train_compare.CONFIG['DATA_ROOT'], mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=12, pin_memory=False)

    probs_collected = {name: [] for name in models_dict.keys()}
    labels_list = []

    print("\n🔍 正在光速遍历测试集...")
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

    baseline_probs = np.array(probs_collected["Best Baseline"])
    ours_probs = np.array(probs_collected["PI-STDNet (Ours)"])
    targets = np.array(labels_list)

    plotter = DecisionSpacePlotterTGRS(save_dir=args.save_dir)
    plotter.plot_decision_shift(baseline_probs, ours_probs, targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures')
    args = parser.parse_args()
    
    extract_and_plot(args)
