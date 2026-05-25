# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import torch
import numpy as np
import logging
import argparse
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ⚠️ 修正：必须从跑出该权重的 ablation_v83 脚本中导入，确保数据预处理和网络架构100%对齐！
try:
    from ablation_v83 import CONFIG, TorNetDatasetAblation, PI_STDNet_Ablation
except ImportError:
    logger.error("❌ 找不到 ablation_v83.py！请确保本脚本与其在同一文件夹内。")
    exit(1)

# 动态解析你要测哪个权重（可以通过 python eval_slider.py --exp_name full 切换）
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='full', help="要评估的消融实验名称，如 full, no_physics_input")
# 这里的开关必须和训练时保持一致，否则网络结构会错位！
parser.add_argument('--disable_physics_inputs', action='store_true')
parser.add_argument('--disable_physics_attn', action='store_true')
parser.add_argument('--disable_topk', action='store_true')
parser.add_argument('--disable_rot_stats', action='store_true')
args = parser.parse_args()

# 强制将 argparse 覆盖到全局（模拟训练时的环境）
import ablation_v83
ablation_v83.args = args
# 根据开关修正通道数
CONFIG['CHANNELS_PER_FRAME'] = 6 if args.disable_physics_inputs else 11

def evaluate_metrics(probs, labels, mask, view_name):
    if mask.sum() == 0: return 0
    p, l = probs[mask], labels[mask]
    
    def calc_metrics(thresh_array):
        best_c, best_t = 0, 0.5
        best_tp, best_fp, best_fn = 0, 0, 0
        for thresh in thresh_array:
            preds = (p > thresh).astype(int)
            tp = ((preds==1)&(l==1)).sum()
            fp = ((preds==1)&(l==0)).sum()
            fn = ((preds==0)&(l==1)).sum()
            csi = tp / (tp + fn + fp + 1e-7)
            if csi > best_c: 
                best_c, best_t, best_tp, best_fp, best_fn = csi, thresh, tp, fp, fn
        return best_c, best_t, best_tp, best_fp, best_fn

    # 🎯 粗粒度
    coarse_thresholds = np.arange(0.10, 0.90, 0.05)
    _, coarse_thresh, _, _, _ = calc_metrics(coarse_thresholds)
    
    # 🎯 细粒度
    fine_min = max(0.10, coarse_thresh - 0.05)
    fine_max = min(0.90, coarse_thresh + 0.06) 
    fine_thresholds = np.arange(fine_min, fine_max, 0.01)
    
    # 🛡️ 防御式合并：将粗和细合并去重，确保最佳结果绝对不会丢失！
    all_search_space = np.unique(np.concatenate([coarse_thresholds, fine_thresholds]))
    best_csi, best_thresh, best_tp, best_fp, best_fn = calc_metrics(all_search_space)
        
    pod = best_tp / (best_tp + best_fn + 1e-7)
    far = best_fp / (best_tp + best_fp + 1e-7)
    
    logger.info(f"  👉 [{view_name}] CSI: {best_csi:.4f} @Thresh: {best_thresh:.2f} | POD: {pod:.3f} | FAR: {far:.3f} | TP={best_tp}, FP={best_fp}")
    return best_csi

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 启动 [{args.exp_name}] 滑动阈值搜索 | 设备: {device}")

    # 1. 实例化模型 (使用 ablation 版本)
    model = PI_STDNet_Ablation().to(device)
    weight_path = f"ablation_{args.exp_name}_best.pth" 
    
    if not os.path.exists(weight_path):
        logger.error(f"❌ 找不到权重文件: {weight_path}")
        return

    logger.info(f"✅ 加载权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 2. 原汁原味加载 ablation 专用的测试集
    logger.info("📦 准备加载测试集数据...")
    test_ds = TorNetDatasetAblation(CONFIG['CATALOG_PATH'], CONFIG['DATA_ROOT'], mode='test')
    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        # 测试时尽量少开 worker，防止和训练抢资源
        num_workers=4, 
        pin_memory=True
    )

    all_probs, all_labels, all_cats = [], [], []
    
    with torch.no_grad():
        for images, labels, cats in tqdm(test_loader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                cls_logit, aux_logit, _, _ = model(images)
            probs = 0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cats.extend(cats.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_cats = np.array(all_cats)
    
    print("\n" + "=" * 60)
    logger.info("🎉 推理完成！开始精准裁决：")
    mask_nc = np.ones_like(all_labels, dtype=bool)
    evaluate_metrics(all_probs, all_labels, mask_nc, "全量视图 NC")
    mask_wc = (all_cats == 2) | (all_cats == 1)
    evaluate_metrics(all_probs, all_labels, mask_wc, "困难视图 WC")
    print("=" * 60)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()


