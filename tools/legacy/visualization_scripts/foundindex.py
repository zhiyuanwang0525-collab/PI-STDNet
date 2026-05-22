# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def find_golden_cases(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🌟 正在启动【高光物理案例海选雷达】... 设备: {device}")
    
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

    dataset = train_compare.TorNetDatasetAblation(train_compare.CONFIG['CATALOG_PATH'], train_compare.CONFIG['DATA_ROOT'], mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

    probs_base, probs_ours = [], []
    labels_list = []

    print("\n🔍 正在全速扫描测试集，评估每一个风暴样本的'优势分'...")
    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning Testset"):
            images = images.to(device)
            cats = cats.numpy()
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                # 基线模型
                cls_b, aux_b, _, _ = models_dict["Best Baseline"](images)
                p_b = (0.6 * torch.sigmoid(cls_b) + 0.4 * torch.sigmoid(aux_b)).cpu().numpy()
                probs_base.extend(p_b)
                # 我们的模型
                cls_o, aux_o, _, _ = models_dict["PI-STDNet (Ours)"](images)
                p_o = (0.6 * torch.sigmoid(cls_o) + 0.4 * torch.sigmoid(aux_o)).cpu().numpy()
                probs_ours.extend(p_o)
            labels_list.extend(cats)

    p_base = np.array(probs_base)
    p_ours = np.array(probs_ours)
    targets = np.array(labels_list)
    
    thresh = 0.60
    
    print("\n" + "="*70)
    print(" 🏆 PI-STDNet 高光物理案例排行榜 (Top 5)")
    print("="*70)

    # 1. 寻找最佳“漏报挽救 (Saved TOR)”
    # 条件：真龙卷，基线 < 0.6 (漏报)，我们 >= 0.6 (抓获)
    mask_saved_tor = (targets == 2) & (p_base < thresh) & (p_ours >= thresh)
    saved_tor_idx = np.where(mask_saved_tor)[0]
    
    # 按照 (我们的概率 - 基线的概率) 排序，找差距最大的！
    adv_tor = p_ours[saved_tor_idx] - p_base[saved_tor_idx]
    best_tor_order = np.argsort(adv_tor)[::-1]
    
    print("\n🌪️ 【神级挽救】微弱龙卷 (基线漏报，我们成功抓获):")
    for i in range(min(5, len(best_tor_order))):
        real_idx = saved_tor_idx[best_tor_order[i]]
        print(f"  👉 Dataset Index: [{real_idx:5d}] | 基线概率: {p_base[real_idx]:.4f} ❌ | PI-STDNet概率: {p_ours[real_idx]:.4f} ✅")

    # 2. 寻找最佳“虚警镇压 (Suppressed WRN)”
    # 条件：虚警风暴，基线 >= 0.6 (误报)，我们 < 0.6 (看穿)
    mask_suppressed_wrn = (targets == 1) & (p_base >= thresh) & (p_ours < thresh)
    suppressed_wrn_idx = np.where(mask_suppressed_wrn)[0]
    
    # 按照 (基线的概率 - 我们的概率) 排序，找基线错得最离谱的！
    adv_wrn = p_base[suppressed_wrn_idx] - p_ours[suppressed_wrn_idx]
    best_wrn_order = np.argsort(adv_wrn)[::-1]
    
    print("\n🛡️ 【完美大坝】强对流虚警 (基线误报，我们成功镇压):")
    for i in range(min(5, len(best_wrn_order))):
        real_idx = suppressed_wrn_idx[best_wrn_order[i]]
        print(f"  👉 Dataset Index: [{real_idx:5d}] | 基线概率: {p_base[real_idx]:.4f} ❌ | PI-STDNet概率: {p_ours[real_idx]:.4f} ✅")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    args = parser.parse_args()
    find_golden_cases(args)
