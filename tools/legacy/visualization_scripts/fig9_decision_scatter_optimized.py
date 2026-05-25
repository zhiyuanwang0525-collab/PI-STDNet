# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
Optimized split Fig. 9: WC probability-shift scatter plot.

Purpose
-------
This script replaces the left panel of the old composite Fig. 9 with an
independent, cleaner scatter plot. It compares the predicted probabilities
of the SE-Net replacement baseline and PI-STDNet in the WC subset and
highlights two decision-change regions:

1. TOR samples missed by the baseline but captured by PI-STDNet.
2. WRN samples falsely flagged by the baseline but suppressed by PI-STDNet.

Outputs are written to a new folder by default:
    paper_figures_split_optimized/

Expected files in your working directory
----------------------------------------
- train_compare.py
- ablation_attn_se_best.pth
- best_pi_v8_GWP1.pth
- TorNet data and catalog paths matching your local setup

Example
-------
python fig9_decision_scatter_optimized.py \
    --ckpt_se ablation_attn_se_best.pth \
    --ckpt_ours best_pi_v8_GWP1.pth \
    --catalog_path /path/to/TorNet/catalog.csv \
    --data_root /path/to/TorNet \
    --save_dir paper_figures_split_optimized
"""

import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import train_compare safely. Some training scripts parse sys.argv at import time.
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import train_compare  # noqa: E402
sys.argv = _original_argv


class MockArgs:
    def __init__(self, attn_type="physics", disable_physics_attn=False, disable_rot_stats=False):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = False
        self.disable_topk = True
        self.disable_rot_stats = disable_rot_stats


def set_publication_style():
    """Use paper-friendly typography and line weights."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "axes.labelsize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "legend.fontsize": 10.5,
    })


class DecisionScatterPlotter:
    def __init__(self, save_dir="paper_figures_split_optimized", threshold=0.60):
        self.save_dir = save_dir
        self.threshold = threshold
        os.makedirs(self.save_dir, exist_ok=True)
        set_publication_style()
        self.color_wrn = "#2F65D9"
        self.color_tor = "#C82432"
        self.color_gray = "#8C8C8C"

    def plot(self, baseline_probs, ours_probs, targets):
        baseline_probs = np.asarray(baseline_probs).reshape(-1)
        ours_probs = np.asarray(ours_probs).reshape(-1)
        targets = np.asarray(targets).reshape(-1)

        if len(baseline_probs) != len(ours_probs) or len(ours_probs) != len(targets):
            raise ValueError("baseline_probs, ours_probs, and targets must have the same length.")

        mask_wrn = targets == 1
        mask_tor = targets == 2
        saved_tor = mask_tor & (baseline_probs < self.threshold) & (ours_probs >= self.threshold)
        suppressed_wrn = mask_wrn & (baseline_probs >= self.threshold) & (ours_probs < self.threshold)
        bg_tor = mask_tor & ~saved_tor
        bg_wrn = mask_wrn & ~suppressed_wrn

        fig, ax = plt.subplots(figsize=(7.2, 7.0))

        # Decision-change regions.
        ax.fill_between(
            [-0.03, self.threshold], self.threshold, 1.03,
            color=self.color_tor, alpha=0.07, zorder=0,
        )
        ax.fill_between(
            [self.threshold, 1.03], -0.03, self.threshold,
            color=self.color_wrn, alpha=0.07, zorder=0,
        )

        # Reference lines.
        ax.plot([-0.05, 1.05], [-0.05, 1.05], color="#777777", linestyle=":", linewidth=1.2, zorder=1)
        ax.axvline(self.threshold, color="#333333", linestyle="--", linewidth=1.4, zorder=2)
        ax.axhline(self.threshold, color="#333333", linestyle="--", linewidth=1.4, zorder=2)

        # Background points are intentionally de-emphasized.
        ax.scatter(
            baseline_probs[bg_wrn], ours_probs[bg_wrn],
            c=self.color_wrn, marker="o", s=9, alpha=0.06, edgecolors="none", zorder=3,
        )
        ax.scatter(
            baseline_probs[bg_tor], ours_probs[bg_tor],
            c=self.color_tor, marker="*", s=18, alpha=0.07, edgecolors="none", zorder=3,
        )

        # Highlight only decision-changing samples.
        ax.scatter(
            baseline_probs[suppressed_wrn], ours_probs[suppressed_wrn],
            c=self.color_wrn, marker="o", s=48, alpha=0.88,
            edgecolors="white", linewidths=0.5, zorder=5,
            label=f"WRN suppressed by PI-STDNet (n={suppressed_wrn.sum()})",
        )
        ax.scatter(
            baseline_probs[saved_tor], ours_probs[saved_tor],
            c=self.color_tor, marker="*", s=120, alpha=0.95,
            edgecolors="white", linewidths=0.6, zorder=6,
            label=f"TOR captured by PI-STDNet (n={saved_tor.sum()})",
        )

        bbox = dict(boxstyle="square,pad=0.30", facecolor="white", edgecolor="#A0A0A0", linewidth=0.8, alpha=0.95)
        ax.text(
            0.23, 0.84, "Improved\ndetection",
            ha="center", va="center", fontsize=11.5, fontweight="bold", bbox=bbox, zorder=7,
        )
        ax.text(
            0.80, 0.24, "False-alarm\nsuppression",
            ha="center", va="center", fontsize=11.5, fontweight="bold", bbox=bbox, zorder=7,
        )
        ax.text(
            self.threshold + 0.015, 0.035,
            f"threshold = {self.threshold:.1f}",
            color="#333333", fontsize=10.5, ha="left", va="bottom",
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("SE-Net replacement predicted probability")
        ax.set_ylabel("PI-STDNet predicted probability")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)

        legend = ax.legend(loc="lower left", framealpha=0.96, edgecolor="#A0A0A0", handletextpad=0.5)
        legend.get_frame().set_linewidth(0.8)

        base = os.path.join(self.save_dir, "Figure9_Decision_Scatter_Optimized")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=600)
        plt.close(fig)

        # Save a compact CSV for reproducibility / later case selection.
        csv_path = base + "_decision_changes.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("type,index,baseline_prob,pi_stdnet_prob\n")
            for idx in np.where(saved_tor)[0]:
                f.write(f"saved_TOR,{idx},{baseline_probs[idx]:.6f},{ours_probs[idx]:.6f}\n")
            for idx in np.where(suppressed_wrn)[0]:
                f.write(f"suppressed_WRN,{idx},{baseline_probs[idx]:.6f},{ours_probs[idx]:.6f}\n")

        print(f"Saved: {base}.pdf")
        print(f"Saved: {base}.png")
        print(f"Saved: {csv_path}")
        print(f"Decision changes: saved TOR={saved_tor.sum()}, suppressed WRN={suppressed_wrn.sum()}")


def load_model(label_name, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {label_name}: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    keys = state_dict.keys()
    has_rot = any("rot_proj" in k for k in keys)

    if any("physics_attn_module.fc" in k for k in keys):
        attn_type, disable_attn = "se", False
    else:
        attn_type, disable_attn = "physics", False

    train_compare.args = MockArgs(
        attn_type=attn_type,
        disable_physics_attn=disable_attn,
        disable_rot_stats=not has_rot,
    )
    model = train_compare.PI_STDNet_Ablation().to(device)
    model.use_train_py_order = "Ours" in label_name or "PI-STDNet" in label_name
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_probabilities(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Keep CLI paths authoritative when train_compare uses CONFIG internally.
    if hasattr(train_compare, "CONFIG"):
        if args.catalog_path:
            train_compare.CONFIG["CATALOG_PATH"] = args.catalog_path
        if args.data_root:
            train_compare.CONFIG["DATA_ROOT"] = args.data_root

    models = {
        "SE-Net replacement": load_model("Best Baseline", args.ckpt_se, device),
        "PI-STDNet (Ours)": load_model("PI-STDNet (Ours)", args.ckpt_ours, device),
    }

    dataset = train_compare.TorNetDatasetAblation(args.catalog_path, args.data_root, mode="test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    probs = {name: [] for name in models}
    targets = []

    amp_enabled = device.type == "cuda" and train_compare.CONFIG.get("USE_AMP", False)
    amp_context = lambda: torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()

    with torch.no_grad():
        for images, _, cats in tqdm(dataloader, desc="Scanning WC test samples"):
            images = images.to(device)
            cats_np = cats.numpy()
            targets.extend(cats_np)
            with amp_context():
                for name, model in models.items():
                    cls_logit, aux_logit, _, _ = model(images)
                    prob = (0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)).detach().cpu().numpy()
                    probs[name].extend(prob)

    return (
        np.asarray(probs["SE-Net replacement"]).reshape(-1),
        np.asarray(probs["PI-STDNet (Ours)"]).reshape(-1),
        np.asarray(targets).reshape(-1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_se", type=str, default="ablation_attn_se_best.pth")
    parser.add_argument("--ckpt_ours", type=str, default="best_pi_v8_GWP1.pth")
    parser.add_argument("--catalog_path", type=str, default="/path/to/TorNet/catalog.csv")
    parser.add_argument("--data_root", type=str, default="/path/to/TorNet")
    parser.add_argument("--save_dir", type=str, default="paper_figures_split_optimized")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    baseline_probs, ours_probs, targets = extract_probabilities(args)
    plotter = DecisionScatterPlotter(save_dir=args.save_dir, threshold=args.threshold)
    plotter.plot(baseline_probs, ours_probs, targets)


if __name__ == "__main__":
    main()



