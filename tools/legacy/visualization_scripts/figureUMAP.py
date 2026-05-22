# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
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

class UMAPPlotter:
    """TGRS 级：深层语义空间 UMAP 流形对比图 (完美分离版)"""
    def __init__(self, save_dir="paper_figures_final"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif', 
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 12,
            'figure.dpi': 300,
        })
        
        self.colors = {1: '#4169E1', 2: '#DC143C'} # WRN: 皇家蓝, TOR: 猩红
        self.labels = {1: 'WRN (False Alarm Storms)', 2: 'TOR (True Tornadoes)'}

    def plot_2x2_comparison(self, features_dict, targets):
        ordered_names = [
            "Baseline (w/o Attention)", 
            "Baseline + SE Module", 
            "Baseline + CBAM Module", 
            "PI-STDNet (Ours)"
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 13), gridspec_kw={'wspace': 0.1, 'hspace': 0.15})
        axes = axes.flatten()
        
        print("\n🚀 开始执行深层语义 UMAP 降维渲染...")
        
        for i, model_name in enumerate(ordered_names):
            if model_name not in features_dict: continue
            
            # 💡 核心修正：绝对禁止 L2 归一化！保留 GELU 激活的原始强度！
            feats = features_dict[model_name]
            
            # 💡 核心修正：使用 euclidean 距离，更能体现全连接层的线性决策边界
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            embed = reducer.fit_transform(feats)
            
            ax = axes[i]
            
            # 1. 绘制底层密度等高线 (让集群边界清晰可见)
            for cat_id in [1, 2]:
                idx = (targets == cat_id)
                sns.kdeplot(
                    x=embed[idx, 0], y=embed[idx, 1], 
                    ax=ax, color=self.colors[cat_id], 
                    fill=True, alpha=0.15, levels=6, linewidths=0, zorder=1
                )
            
            # 2. 绘制散点
            for cat_id in [1, 2]:
                idx = (targets == cat_id)
                ax.scatter(embed[idx, 0], embed[idx, 1], 
                           c=self.colors[cat_id], marker='.' if cat_id == 1 else '*', 
                           s=25 if cat_id == 1 else 60, # 龙卷星号稍微显眼一点
                           alpha=0.7, 
                           label=self.labels[cat_id] if i == 0 else "",
                           edgecolors='white', linewidths=0.3, zorder=2)
            
            title_prefix = ['(a)', '(b)', '(c)', '(d)'][i]
            title_color = '#DC143C' if "Ours" in model_name else 'black'
            ax.set_title(f"{title_prefix} {model_name}", pad=12, fontweight='bold', fontsize=16, color=title_color)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#BBBBBB')
                spine.set_linewidth(1.5)

        # 底部统一图例
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05), frameon=True, fontsize=15, edgecolor='black')

        fig.suptitle("Deep Semantic Space Decoupling Before Final Classifier", 
                     fontsize=20, fontweight='bold', y=0.94)

        save_path = os.path.join(self.save_dir, "Figure5_UMAP_Deep_Semantic.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✅ 绝杀！【深层语义 UMAP 图】已保存至: {save_path}")
        plt.close()

def extract_features(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 正在初始化特征提取流水线... 设备: {device}")
    
    target_models = [
        ("Baseline (w/o Attention)", args.ckpt_base),
        ("Baseline + SE Module", args.ckpt_se),
        ("Baseline + CBAM Module", args.ckpt_cbam),
        ("PI-STDNet (Ours)", args.ckpt_ours)
    ]
    
    models_dict = {}
    captured = {} 
    
    # 💡 核心修正：获取层输出 (Output)，而不是输入 (Input)
    def get_hook(name):
        def hook(model, input, output):
            captured[name] = output.detach().cpu().numpy()
        return hook

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
        
        if "Ours" in label_name: m.use_train_py_order = True
        else: m.use_train_py_order = False

        m.load_state_dict(state_dict)
        m.eval()
        
        # 💡 核心修正：将钩子挂在 cls_head 的第 2 层 (GELU 层之后，也是 256 维深层特征)
        # cls_head = [LayerNorm(0), Linear(1), GELU(2), Dropout(3), Linear(4)]
        m.cls_head[2].register_forward_hook(get_hook(label_name))
        
        models_dict[label_name] = m

    # 满速读取数据
    dataset = train_compare.TorNetDatasetAblation(train_compare.CONFIG['CATALOG_PATH'], train_compare.CONFIG['DATA_ROOT'], mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=12, pin_memory=False)

    features_collected = {name: [] for name in models_dict.keys()}
    labels_list = []

    print("\n🔍 正在 12核全速 遍历测试集，采集 256维 深层语义特征...")
    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning Testset"):
            images = images.to(device)
            cats = cats.numpy()
            
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                for name, m in models_dict.items():
                    _ = m(images) # 触发前向传播，让钩子拦截特征
                    features_collected[name].append(captured[name])
            
            labels_list.extend(cats)

    Y = np.array(labels_list)
    final_features = {name: np.concatenate(feat_list, axis=0) for name, feat_list in features_collected.items()}
    
    # =========================================================================
    # 💡 纯净随机抽样：不作弊，靠实力硬碰硬！(各 1500 个样本)
    # =========================================================================
    print("\n⚔️ 正在执行公平等量抽样...")
    
    idx_tor = np.where(Y == 2)[0]
    idx_wrn = np.where(Y == 1)[0]
    
    np.random.shuffle(idx_tor)
    np.random.shuffle(idx_wrn)
    
    # 提取 1500 个 TOR 和 1500 个 WRN
    selected_indices = np.concatenate([idx_tor[:1500], idx_wrn[:1500]])
    np.random.shuffle(selected_indices)

    filtered_features = {name: feats[selected_indices] for name, feats in final_features.items()}
    filtered_Y = Y[selected_indices]

    plotter = UMAPPlotter(save_dir=args.save_dir)
    plotter.plot_2x2_comparison(filtered_features, filtered_Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_base', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_cbam', type=str, default='ablation_attn_cbam_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    parser.add_argument('--save_dir', type=str, default='paper_figures')
    args = parser.parse_args()
    
    extract_features(args)
