# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 从你上传的 V8 满血版训练代码中导入模型、数据集和配置
from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG

def run_local_inference(model_path, data_root, catalog_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 正在使用 {device} 启动本地推理 (i9-14900K + 4070S)...")
    
    # 1. 加载模型与权重
    model = PI_STDNet_V8().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 准备本地测试集
    test_ds = TorNetDatasetV8_Full(catalog_path, data_root, mode='test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)

    all_probs, all_labels, all_cats = [], [], []
    
    # ================= 强壮的 EF 等级提取逻辑 =================
    raw_efs = test_ds.catalog['ef_number'].fillna(-1).values
    ef_scales_list = []
    for val in raw_efs:
        if pd.isna(val) or val == -1 or val == 'Unknown':
            ef_scales_list.append('Unknown')
        else:
            val_str = str(val).strip()
            if val_str.startswith('EF') or val_str.startswith('F'):
                ef_scales_list.append(val_str.replace('F', 'EF'))
            else:
                # 处理纯数字情况，如 "0.0", "1", "2.0" -> "EF0", "EF1", "EF2"
                try:
                    ef_num = int(float(val_str))
                    ef_scales_list.append(f'EF{ef_num}')
                except:
                    ef_scales_list.append('Unknown')
    ef_scales = np.array(ef_scales_list)
    # ==========================================================

    with torch.no_grad():
        for images, labels, cats in tqdm(test_loader, desc="推理中"):
            images = images.to(device)
            cls_logit, aux_logit, _, _ = model(images)
            
            # 使用与训练一致的 0.6/0.4 融合逻辑
            probs = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).cpu().numpy()
            
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_cats.extend(cats.numpy().tolist())

    return np.array(all_probs), np.array(all_labels), np.array(all_cats), ef_scales

def plot_performance_diagram(probs, labels, mask, view_name):
    print(f"\n🎨 正在绘制 Figure 5: Performance Diagram ({view_name})...")
    p, l = probs[mask], labels[mask]
    
    thresholds = np.linspace(0.01, 0.99, 100)
    pods, srs = [], []
    
    for t in thresholds:
        preds = (p > t).astype(int)
        tp = ((preds == 1) & (l == 1)).sum()
        fp = ((preds == 1) & (l == 0)).sum()
        fn = ((preds == 0) & (l == 1)).sum()
        
        pod = tp / (tp + fn + 1e-7)
        sr = tp / (tp + fp + 1e-7) # Success Ratio = 1 - FAR
        pods.append(pod)
        srs.append(sr)

    plt.figure(figsize=(8, 8))
    
    # 绘制 CSI 等值线背景
    x, y = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    csi_grid = 1 / (1/x + 1/y - 1)
    csi_contours = plt.contour(x, y, csi_grid, levels=np.arange(0.1, 1.0, 0.1), colors='gray', linestyles='dashed', alpha=0.5)
    plt.clabel(csi_contours, inline=True, fontsize=10, fmt='%1.1f')

    # 绘制 PI-STDNet 曲线
    plt.plot(srs, pods, color='crimson', linewidth=3, label='PI-STDNet (Ours)')

    # 标注 TorNet 官方基线点
    if "NC" in view_name:
        plt.scatter(0.58, 0.48, color='blue', marker='*', s=250, zorder=5, label='CNN Baseline (CSI: 0.3487)')
        plt.scatter(0.52, 0.46, color='orange', marker='s', s=100, zorder=5, label='Random Forest')
        plt.scatter(0.38, 0.47, color='green', marker='^', s=100, zorder=5, label='Logistic Regression')
        plt.scatter(0.08, 0.31, color='black', marker='v', s=100, zorder=5, label='TVS')
    elif "WC" in view_name:
        plt.scatter(0.61, 0.48, color='blue', marker='*', s=250, zorder=5, label='CNN Baseline (CSI: 0.3617)')
        plt.scatter(0.51, 0.50, color='orange', marker='s', s=100, zorder=5, label='Random Forest')
        plt.scatter(0.39, 0.53, color='green', marker='^', s=100, zorder=5, label='Logistic Regression')
        plt.scatter(0.39, 0.31, color='black', marker='v', s=100, zorder=5, label='TVS')

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Success Ratio (1 - FAR)', fontsize=14, fontweight='bold')
    plt.ylabel('Probability of Detection (POD)', fontsize=14, fontweight='bold')
    plt.title(f'Performance Diagram - {view_name}', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'Fig5_Performance_{view_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def normalize_ef_label(val):
    """
    Normalize EF labels into EF0, EF1, ..., EF5, or Unknown.
    """
    if pd.isna(val):
        return "Unknown"

    val_str = str(val).strip().upper()

    if val_str in ["", "UNKNOWN", "NAN", "-1"]:
        return "Unknown"

    if val_str.startswith("EF"):
        return val_str

    if val_str.startswith("F"):
        return "EF" + val_str[1:]

    try:
        return f"EF{int(float(val_str))}"
    except Exception:
        return "Unknown"


def wilson_ci(k, n, z=1.96):
    """
    Wilson confidence interval for binomial proportion.
    """
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom

    return max(0.0, center - half), min(1.0, center + half)


def normalize_ef_label(val):
    """
    Normalize EF labels into EF0, EF1, ..., EF5, or Unknown.
    """
    if pd.isna(val):
        return "Unknown"

    val_str = str(val).strip().upper()

    if val_str in ["", "UNKNOWN", "NAN", "-1"]:
        return "Unknown"

    if val_str.startswith("EF"):
        return val_str

    if val_str.startswith("F"):
        return "EF" + val_str[1:]

    try:
        return f"EF{int(float(val_str))}"
    except Exception:
        return "Unknown"


def wilson_ci(k, n, z=1.96):
    """
    Wilson 95% confidence interval for a binomial proportion.
    """
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom

    return max(0.0, center - half), min(1.0, center + half)


def plot_ef_scale_pod(
    probs,
    labels,
    efs,
    threshold=0.55,
    save_dir="paper_figures_ef_scale_clean",
    include_empty=False,
):
    """
    Clean EF-scale POD figure.

    Compared with the old simple bar chart, this version adds:
    - 95% Wilson confidence intervals
    - sample size under each EF label
    - detected / total count above each bar
    - mean predicted probability as a small marker line

    The default threshold is kept as 0.55 to reproduce the original
    Fig. 10-style values. If the manuscript requires the unified
    threshold tau=0.6, change threshold to 0.60 in the function call.
    """
    os.makedirs(save_dir, exist_ok=True)

    tor_mask = labels == 1
    p_tor = np.asarray(probs)[tor_mask]
    ef_tor = np.asarray([normalize_ef_label(v) for v in efs[tor_mask]])

    categories_all = ["EF0", "EF1", "EF2", "EF3", "EF4", "EF5"]

    rows = []
    for ef in categories_all:
        idx = ef_tor == ef
        n = int(idx.sum())

        if n == 0 and not include_empty:
            continue

        if n > 0:
            hits = int((p_tor[idx] > threshold).sum())
            pod = hits / n
            mean_prob = float(np.mean(p_tor[idx]))
            ci_low, ci_high = wilson_ci(hits, n)
        else:
            hits = 0
            pod = np.nan
            mean_prob = np.nan
            ci_low, ci_high = np.nan, np.nan

        rows.append({
            "EF": ef,
            "n": n,
            "hits": hits,
            "POD": pod,
            "mean_probability": mean_prob,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No EF-rated tornado samples found. Please check ef_number in the catalog.")

    # ----------------------------
    # Nature-like clean style
    # ----------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.4,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 0.65,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })

    x = np.arange(len(df))
    pod = df["POD"].values.astype(float)
    mean_prob = df["mean_probability"].values.astype(float)

    yerr_lower = pod - df["ci_low"].values.astype(float)
    yerr_upper = df["ci_high"].values.astype(float) - pod
    yerr = np.vstack([yerr_lower, yerr_upper])

    fig, ax = plt.subplots(figsize=(5.15, 3.30))

    bars = ax.bar(
        x,
        pod,
        width=0.62,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.55,
        alpha=0.88,
        label="POD",
        zorder=3,
    )

    ax.errorbar(
        x,
        pod,
        yerr=yerr,
        fmt="none",
        ecolor="0.20",
        elinewidth=0.75,
        capsize=2.3,
        capthick=0.75,
        zorder=4,
        label="95% CI",
    )

    ax.plot(
        x,
        mean_prob,
        marker="o",
        markersize=3.8,
        linewidth=1.15,
        color="#C44E52",
        label="Mean probability",
        zorder=5,
    )

    # Put detected/total above bars, not inside bars.
    for i, row in df.iterrows():
        if not np.isfinite(row["POD"]):
            continue

        label_y = min(max(row["ci_high"] + 0.035, row["POD"] + 0.045), 1.045)

        ax.text(
            i,
            label_y,
            f"{int(row['hits'])}/{int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="0.15",
            fontweight="bold",
        )

    # Optional visual separator for significant tornadoes.
    if "EF2" in df["EF"].values:
        ef2_pos = int(df.index[df["EF"] == "EF2"][0])
        if ef2_pos > 0:
            ax.axvline(
                ef2_pos - 0.5,
                color="0.35",
                linestyle="--",
                linewidth=0.75,
                zorder=1,
            )
            ax.text(
                ef2_pos - 0.43,
                1.065,
                "EF2+",
                ha="left",
                va="center",
                fontsize=6.6,
                color="0.25",
            )

    xtick_labels = [
        f"{ef}\n$n$={n}"
        for ef, n in zip(df["EF"].tolist(), df["n"].tolist())
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)

    ax.set_ylim(0.0, 1.10)
    ax.set_xlim(-0.55, len(df) - 0.45)

    ax.set_ylabel("Probability of Detection (POD)")
    ax.set_xlabel("Enhanced Fujita (EF) scale")

    ax.text(
        0.01,
        0.985,
        f"Decision threshold = {threshold:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="0.25",
    )

    ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.35, zorder=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.80",
        handlelength=1.5,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.20)

    png_path = os.path.join(save_dir, "Figure10_EF_Scale_POD_Clean.png")
    pdf_path = os.path.join(save_dir, "Figure10_EF_Scale_POD_Clean.pdf")
    csv_path = os.path.join(save_dir, "Figure10_EF_Scale_POD_Clean.csv")

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    df.to_csv(csv_path, index=False)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")

    return df

if __name__ == '__main__':
    # 🚀 注意：请再次确认以下路径为你本地真实的路径
    LOCAL_DATA_ROOT = r'/path/to/TorNet/'
    LOCAL_CATALOG = r'/path/to/TorNet/catalog.csv'
    MODEL_WEIGHTS = r'best_pi_v8_GWP.pth'

    # 1. 运行本地推理
    probs, labels, cats, efs = run_local_inference(MODEL_WEIGHTS, LOCAL_DATA_ROOT, LOCAL_CATALOG)
    
    # 2. 绘制 NC 视图性能图 (Normal Category)
    plot_performance_diagram(probs, labels, np.ones_like(labels, dtype=bool), view_name="NC View")
    
    # 3. 绘制 WC 视图性能图 (Warning Category)
    wc_mask = (cats == 2) | (cats == 1)
    plot_performance_diagram(probs, labels, wc_mask, view_name="WC View")
    
    # 4. 绘制 EF 等级检测率
    # 使用在 WC 视图上达到最佳 CSI 的阈值（假设为 0.55，可根据你的最佳结果微调）
    plot_ef_scale_pod(
        probs,
        labels,
        efs,
        threshold=0.55,
        save_dir="paper_figures_ef_scale_clean"
    )
    print("\n✅ 任务完成！高清图表已保存在本地目录。")


