# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
PI-STDNet 公平对比评估脚本
========================================
功能:
  1. 加载最佳权值，在测试集上推理得到所有样本的概率
  2. 全阈值扫描 (0.01~0.99)，输出完整的 POD/FAR/CSI/SR/BIAS 曲线
  3. 在 CNN baseline 的 FAR≈0.241 操作点上报告 PI-STDNet 的 POD
  4. 计算 AUC (ROC) 和 AUC-PD (Performance Diagram 下面积)
  5. 分 NC / WC 两个视图输出结果
  6. 保存所有概率用于后续绘制 Performance Diagram

用法:
  python evaluate_fair_comparison.py \
      --checkpoint best_pi_v8_server_fulldata.pth \
      --catalog_path /path/to/catalog.csv \
      --data_root /path/to/tornet
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# ============= 从训练代码导入模型和数据集 =============
# 请确保 trainV72.py 在同目录或 PYTHONPATH 中
from train import (
    PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG
)

def parse_args():
    parser = argparse.ArgumentParser(description='PI-STDNet Fair Comparison Evaluation')
    parser.add_argument('--checkpoint', type=str, default='best_pi_v8_GWP.pth',
                        help='模型权值路径')
    parser.add_argument('--catalog_path', type=str, default=CONFIG['CATALOG_PATH'])
    parser.add_argument('--data_root', type=str, default=CONFIG['DATA_ROOT'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='保存结果的目录')
    return parser.parse_args()


def evaluate_at_threshold(probs, labels, thresh):
    """在给定阈值下计算所有指标"""
    preds = (probs > thresh).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    
    pod = tp / (tp + fn + 1e-7)       # = TPR = Recall
    far = fp / (tp + fp + 1e-7)       # False Alarm Ratio
    sr = tp / (tp + fp + 1e-7)        # Success Rate = 1 - FAR = Precision
    csi = tp / (tp + fn + fp + 1e-7)  # Critical Success Index
    bias = (tp + fp) / (tp + fn + 1e-7)
    fpr = fp / (fp + tn + 1e-7)       # False Positive Rate (for ROC)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    return {
        'thresh': thresh,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'pod': pod, 'far': far, 'sr': sr, 'csi': csi, 'bias': bias,
        'fpr': fpr, 'acc': acc
    }


def compute_auc_pd(probs, labels, n_points=1000):
    """
    计算 Performance Diagram 下面积 (AUC-PD)
    PD: x轴 = SR (Success Rate = 1-FAR), y轴 = POD (TPR)
    使用梯形法则积分
    """
    sr_list, pod_list = [], []
    
    for thresh in np.linspace(0.001, 0.999, n_points):
        m = evaluate_at_threshold(probs, labels, thresh)
        if m['tp'] + m['fp'] > 0:  # 避免无预测的情况
            sr_list.append(m['sr'])
            pod_list.append(m['pod'])
    
    # 按 SR 排序用于积分
    sr_arr = np.array(sr_list)
    pod_arr = np.array(pod_list)
    sort_idx = np.argsort(sr_arr)
    sr_sorted = sr_arr[sort_idx]
    pod_sorted = pod_arr[sort_idx]
    
    # 梯形法则
    auc_pd = np.trapz(pod_sorted, sr_sorted)
    return auc_pd


def find_threshold_at_far(probs, labels, target_far, tolerance=0.015):
    """找到最接近目标 FAR 的阈值"""
    best_thresh, best_diff = 0.5, 1.0
    best_metrics = None
    
    for thresh in np.arange(0.01, 0.99, 0.005):
        m = evaluate_at_threshold(probs, labels, thresh)
        diff = abs(m['far'] - target_far)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            best_metrics = m
    
    return best_thresh, best_metrics


def full_evaluation(probs, labels, view_name, save_dir=None):
    """对一个视图进行完整评估"""
    print(f"\n{'='*70}")
    print(f"📊 {view_name}")
    print(f"{'='*70}")
    print(f"  样本总数: {len(labels)} | 正样本: {labels.sum()} | 负样本: {(1-labels).sum()}")
    
    # ====== 1. 全阈值扫描，找最优 CSI ======
    all_results = []
    best_csi, best_thresh = 0, 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        m = evaluate_at_threshold(probs, labels, thresh)
        all_results.append(m)
        if m['csi'] > best_csi:
            best_csi = m['csi']
            best_thresh = thresh
    
    best_m = evaluate_at_threshold(probs, labels, best_thresh)
    print(f"\n  🏆 最优 CSI 操作点:")
    print(f"     Thresh={best_m['thresh']:.2f} | CSI={best_m['csi']:.4f} | "
          f"POD={best_m['pod']:.4f} | FAR={best_m['far']:.4f} | "
          f"SR={best_m['sr']:.4f} | BIAS={best_m['bias']:.3f}")
    print(f"     TP={best_m['tp']} | FP={best_m['fp']} | FN={best_m['fn']} | TN={best_m['tn']}")
    
    # ====== 2. AUC (ROC) ======
    roc_auc = roc_auc_score(labels, probs)
    print(f"\n  📈 AUC (ROC): {roc_auc:.4f}")
    
    # ====== 3. AUC-PD (Performance Diagram) ======
    auc_pd = compute_auc_pd(probs, labels)
    print(f"  📈 AUC-PD:   {auc_pd:.4f}")
    
    # ====== 4. 在 CNN Baseline 的 FAR≈0.241 操作点比较 ======
    print(f"\n  🎯 在 CNN Baseline FAR≈0.241 操作点的公平对比:")
    thresh_fair, m_fair = find_threshold_at_far(probs, labels, target_far=0.241)
    print(f"     Thresh={thresh_fair:.3f} | FAR={m_fair['far']:.4f} | "
          f"POD={m_fair['pod']:.4f} | CSI={m_fair['csi']:.4f} | "
          f"SR={m_fair['sr']:.4f} | BIAS={m_fair['bias']:.3f}")
    print(f"     TP={m_fair['tp']} | FP={m_fair['fp']} | FN={m_fair['fn']} | TN={m_fair['tn']}")
    
    cnn_pod_baseline = 0.396  # CNN baseline 在 FAR≈0.241 时的 POD
    if m_fair['pod'] > cnn_pod_baseline:
        print(f"     ✅ PI-STDNet POD={m_fair['pod']:.4f} > CNN Baseline POD=0.396")
        print(f"     → 在相同误报率下，PI-STDNet 多检出了 "
              f"{m_fair['tp'] - 789} 个龙卷 (789→{m_fair['tp']})")
    else:
        print(f"     ⚠️ PI-STDNet POD={m_fair['pod']:.4f} ≤ CNN Baseline POD=0.396")
    
    # ====== 5. 额外操作点：FAR=0.3, 0.35, 0.4, 0.5 ======
    print(f"\n  📋 不同 FAR 操作点的 POD:")
    print(f"     {'FAR目标':>10} | {'实际FAR':>8} | {'POD':>8} | {'CSI':>8} | {'Thresh':>8}")
    print(f"     {'-'*55}")
    for target_far in [0.15, 0.20, 0.241, 0.30, 0.35, 0.40, 0.50]:
        t, m = find_threshold_at_far(probs, labels, target_far)
        marker = " ← CNN baseline" if abs(target_far - 0.241) < 0.01 else ""
        print(f"     {target_far:>10.3f} | {m['far']:>8.4f} | {m['pod']:>8.4f} | "
              f"{m['csi']:>8.4f} | {t:>8.3f}{marker}")
    
    # ====== 6. 保存详细结果 ======
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存全阈值扫描结果
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(save_dir, f"threshold_sweep_{view_name.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  💾 阈值扫描结果已保存: {csv_path}")
        
        # 保存 ROC 曲线数据
        fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
        roc_df = pd.DataFrame({'fpr': fpr_arr, 'tpr': tpr_arr})
        roc_path = os.path.join(save_dir, f"roc_curve_{view_name.replace(' ', '_')}.csv")
        roc_df.to_csv(roc_path, index=False)
        
    return {
        'view': view_name,
        'best_csi': best_csi,
        'best_thresh': best_thresh,
        'auc': roc_auc,
        'auc_pd': auc_pd,
        'pod_at_baseline_far': m_fair['pod'],
        'csi_at_baseline_far': m_fair['csi'],
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("🔬 PI-STDNet 公平对比评估")
    print("=" * 70)
    print(f"  设备: {device}")
    print(f"  权值: {args.checkpoint}")
    
    # ====== 加载模型 ======
    model = PI_STDNet_V8().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  模型加载成功 | 保存时 NC CSI: {ckpt.get('nc_csi', 'N/A')} | WC CSI: {ckpt.get('wc_csi', 'N/A')}")
    
    # ====== 加载测试集 ======
    test_ds = TorNetDatasetV8_Full(args.catalog_path, args.data_root, mode='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True)
    
    # ====== 推理 ======
    print("\n🚀 开始推理...")
    all_probs, all_labels, all_cats = [], [], []
    
    with torch.no_grad():
        for images, labels, cats in tqdm(test_loader, desc="推理中"):
            with torch.amp.autocast('cuda', enabled=True):
                cls_logit, aux_logit, _, _ = model(images.to(device, non_blocking=True))
            
            probs = 0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cats.extend(cats.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_cats = np.array(all_cats)
    
    print(f"  推理完成 | 总样本: {len(all_probs)} | 正样本: {all_labels.sum():.0f}")
    
    # ====== 保存概率 (用于后续绘图) ======
    os.makedirs(args.save_dir, exist_ok=True)
    np.savez(os.path.join(args.save_dir, 'predictions.npz'),
             probs=all_probs, labels=all_labels, cats=all_cats)
    print(f"  💾 概率已保存: {args.save_dir}/predictions.npz")
    
    # ====== NC 视图: All nulls vs TOR ======
    mask_nc = np.ones_like(all_labels, dtype=bool)
    result_nc = full_evaluation(all_probs[mask_nc], all_labels[mask_nc],
                                "NC (All nulls vs TOR)", args.save_dir)
    
    # ====== WC 视图: Warnings vs TOR ======
    mask_wc = (all_cats == 2) | (all_cats == 1)
    result_wc = full_evaluation(all_probs[mask_wc], all_labels[mask_wc],
                                "WC (Warnings vs TOR)", args.save_dir)
    
    # ====== 汇总对比表 ======
    print(f"\n{'='*70}")
    print(f"📋 最终汇总: PI-STDNet vs CNN Baseline")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'指标':<20} | {'CNN Baseline':>14} | {'PI-STDNet':>14} | {'提升':>10}")
    print(f"  {'-'*65}")
    
    # NC view
    print(f"  {'[NC] CSI':.<20} | {'0.3487':>14} | {result_nc['best_csi']:>14.4f} | "
          f"{result_nc['best_csi']-0.3487:>+10.4f}")
    print(f"  {'[NC] AUC':.<20} | {'0.8760':>14} | {result_nc['auc']:>14.4f} | "
          f"{result_nc['auc']-0.8760:>+10.4f}")
    print(f"  {'[NC] AUC-PD':.<20} | {'0.5294':>14} | {result_nc['auc_pd']:>14.4f} | "
          f"{result_nc['auc_pd']-0.5294:>+10.4f}")
    print(f"  {'[NC] POD@FAR=0.24':.<20} | {'0.396':>14} | {result_nc['pod_at_baseline_far']:>14.4f} | "
          f"{result_nc['pod_at_baseline_far']-0.396:>+10.4f}")
    
    print(f"  {'-'*65}")
    
    # WC view  
    print(f"  {'[WC] CSI':.<20} | {'0.3617':>14} | {result_wc['best_csi']:>14.4f} | "
          f"{result_wc['best_csi']-0.3617:>+10.4f}")
    print(f"  {'[WC] AUC':.<20} | {'0.7910':>14} | {result_wc['auc']:>14.4f} | "
          f"{result_wc['auc']-0.7910:>+10.4f}")
    print(f"  {'[WC] AUC-PD':.<20} | {'0.5698':>14} | {result_wc['auc_pd']:>14.4f} | "
          f"{result_wc['auc_pd']-0.5698:>+10.4f}")
    
    print(f"\n  🎯 关键结论:")
    print(f"  如果 [NC] POD@FAR=0.24 > 0.396，说明在相同误报率下")
    print(f"  PI-STDNet 的检测能力仍然更强，提升来自架构而非策略。")
    print(f"\n  如果 AUC 和 AUC-PD 均优于 baseline，说明模型在所有")
    print(f"  操作点上都更好，这是 threshold-independent 的证据。")


if __name__ == '__main__':
    main()


