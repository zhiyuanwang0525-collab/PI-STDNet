# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
import torch
import numpy as np
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
        self.disable_topk = True  # 契合最新的 Max Pooling 架构
        self.disable_rot_stats = disable_rot_stats

def calc_metrics(probs, labels, mask, thresh=0.6):
    """标准的 CSI, POD, FAR 气象评估矩阵计算"""
    if mask.sum() == 0: return 0, 0, 0, 0, 0, 0
    p, l = probs[mask], labels[mask]
    preds = (p >= thresh).astype(int)
    
    tp = ((preds == 1) & (l == 1)).sum()
    fp = ((preds == 1) & (l == 0)).sum()
    fn = ((preds == 0) & (l == 1)).sum()
    
    csi = tp / (tp + fn + fp + 1e-7)
    pod = tp / (tp + fn + 1e-7)
    far = fp / (tp + fp + 1e-7)
    return csi, pod, far, tp, fp, fn

def inspect_weights(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔍 正在启动【模型权重体检中心 (双轨兼容版)】... 设备: {device}\n")
    
    target_models = [
        ("Baseline (w/o Attention)", args.ckpt_base),
        ("Baseline + SE Module", args.ckpt_se),
        ("Baseline + CBAM Module", args.ckpt_cbam),
        ("PI-STDNet (Ours)", args.ckpt_ours)
    ]
    
    models_dict = {}
    
    print("="*65)
    print(" 🛠️ 阶段一：读取权重内置的历史最高记录 (Saved Best Checkpoint)")
    print("="*65)
    
    for label_name, ckpt_path in target_models:
        if not os.path.exists(ckpt_path):
            print(f"❌ 找不到文件跳过: {ckpt_path}")
            continue
            
        ckpt = torch.load(ckpt_path, map_location=device)
        
        saved_nc = ckpt.get('nc_csi', '未记录 (可能是旧版权重)')
        saved_wc = ckpt.get('wc_csi', '未记录 (可能是旧版权重)')
        if isinstance(saved_nc, float): saved_nc = f"{saved_nc:.4f}"
        if isinstance(saved_wc, float): saved_wc = f"{saved_wc:.4f}"
        
        print(f"📦 [{label_name}] -> {ckpt_path}")
        print(f"   内置保存记录 -> NC CSI: {saved_nc} | WC CSI: {saved_wc}")
        
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        keys = state_dict.keys()
        has_rot = any('rot_proj' in k for k in keys)
        
        if not any('physics_attn_module' in k for k in keys):
            attn_type, disable_attn = 'physics', True
        elif any('physics_attn_module.fc' in k for k in keys):
            attn_type, disable_attn = 'se', False
        elif any('physics_attn_module.channel_mlp' in k for k in keys):
            attn_type, disable_attn = 'cbam', False
        else:
            attn_type, disable_attn = 'physics', False
            
        train_compare.args = MockArgs(attn_type=attn_type, disable_physics_attn=disable_attn, disable_rot_stats=not has_rot)
        m = train_compare.PI_STDNet_Ablation().to(device)
        
        # ==========================================================
        # 🌟 核心开关：给不同模型分配对应的“特征拼接顺序”
        # ==========================================================
        if "Ours" in label_name:
            m.use_train_py_order = True   # PI-STDNet 用 train.py 的正确顺序
            print("   [路由分配] ✅ 启用 train.py 拼接顺序 (Ours专用)")
        else:
            m.use_train_py_order = False  # 基线模型用 train_compare.py 的历史顺序
            print("   [路由分配] ⚠️ 启用 train_compare.py 历史拼接顺序 (基线专用)")
            
        m.load_state_dict(state_dict)
        m.eval()
        models_dict[label_name] = m

    print("\n" + "="*65)
    print(" ⚔️  阶段二：在测试集上进行单进程全量推理考核 (以 0.6 为阈值)")
    print("="*65)
    
    # 使用单进程以防 Windows 幽灵内存 Bug
    dataset = train_compare.TorNetDatasetAblation(train_compare.CONFIG['CATALOG_PATH'], train_compare.CONFIG['DATA_ROOT'], mode='test')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=12)

    probs_collected = {name: [] for name in models_dict.keys()}
    labels_list, cats_list = [], []

    with torch.no_grad():
        for images, labels, cats in tqdm(dataloader, desc="Evaluating Testset"):
            images = images.to(device)
            cats = cats.numpy()
            labels = labels.numpy()
            
            with torch.amp.autocast('cuda', enabled=train_compare.CONFIG['USE_AMP']):
                for name, m in models_dict.items():
                    cls_logit, aux_logit, _, _ = m(images)
                    prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).cpu().numpy()
                    probs_collected[name].extend(prob)
            
            labels_list.extend(labels)
            cats_list.extend(cats)

    labels = np.array(labels_list)
    cats = np.array(cats_list)
    mask_wc = (cats == 1) | (cats == 2)
    
    print("\n" + "="*65)
    print(" 📊 最终测试集量化报告 (Decision Threshold = 0.60)")
    print("="*65)
    
    for name in models_dict.keys():
        if name not in probs_collected or len(probs_collected[name]) == 0: continue
        probs = np.array(probs_collected[name])
        csi, pod, far, tp, fp, fn = calc_metrics(probs, labels, mask_wc, thresh=0.6)
        
        print(f"🚀 [{name}]")
        print(f"   CSI (临界成功指数): {csi:.4f}")
        print(f"   POD (击中率)      : {pod:.4f}  [TP={tp}, FN={fn}]")
        print(f"   FAR (虚警率)      : {far:.4f}  [FP={fp}]")
        print("-" * 50)
        
    print("\n✅ 双轨兼容体检完成！请核对 CSI 数据。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_base', type=str, default='ablation_no_physics_attn_best.pth')
    parser.add_argument('--ckpt_se', type=str, default='ablation_attn_se_best.pth')
    parser.add_argument('--ckpt_cbam', type=str, default='ablation_attn_cbam_best.pth')
    parser.add_argument('--ckpt_ours', type=str, default='best_pi_v8_GWP1.pth')
    
    # 路径请根据你实际的运行环境确认
    parser.add_argument('--catalog_path', type=str, default='/path/to/TorNet/catalog.csv')
    parser.add_argument('--data_root', type=str, default='/path/to/TorNet/')
    args = parser.parse_args()
    
    inspect_weights(args)
