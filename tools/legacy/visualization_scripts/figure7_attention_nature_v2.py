# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import scipy.ndimage as ndimage

# ============================================================
# Import train_compare safely
# ============================================================

_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import train_compare
sys.argv = _original_argv


# ============================================================
# Mock args for train_compare.PI_STDNet_Ablation
# ============================================================

class MockArgs:
    def __init__(
        self,
        attn_type="physics",
        disable_physics_attn=False,
        disable_rot_stats=False,
    ):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = False
        self.disable_topk = True
        self.disable_rot_stats = disable_rot_stats


# ============================================================
# Nature-like style
# ============================================================

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.3,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 0.60,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


# ============================================================
# Colormaps
# ============================================================

def get_dbz_cmap():
    cmap = plt.cm.turbo.copy()
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=-10, vmax=65)
    return cmap, norm


def get_vel_cmap():
    cmap = plt.cm.seismic.copy()
    cmap.set_bad("white")
    norm = mcolors.TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)
    return cmap, norm


def get_attention_cmap():
    colors = [
        "#FFFFFF",
        "#FFF7BC",
        "#FEE391",
        "#FEC44F",
        "#FE9929",
        "#EC7014",
        "#CC4C02",
        "#8C2D04",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "attention_nature",
        colors,
        N=256,
    )
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    return cmap, norm


def prepare_attention(attn, scale=2.0, smooth_sigma=1.0, top_percent=0.0):
    """
    Prepare attention map for display.

    top_percent:
        0 means keep full attention map.
        20 means only keep top 20% response.
    """
    attn = np.asarray(attn, dtype=np.float32)
    attn = np.nan_to_num(attn, nan=0.0)

    if smooth_sigma > 0:
        attn = ndimage.gaussian_filter(attn, sigma=smooth_sigma)

    attn = np.clip(attn * scale, 0.0, 1.0)

    if top_percent and top_percent > 0:
        valid = attn[np.isfinite(attn)]
        if valid.size > 0:
            threshold = np.percentile(valid, 100.0 - top_percent)
            attn = np.where(attn >= threshold, attn, np.nan)

    return attn


# ============================================================
# Radar-sector geometry and plotting
# ============================================================

def get_polar_grid(H=120, W=240, r_max=60.0, az_min=-30.0, az_max=30.0):
    az_edges = np.linspace(np.radians(az_min), np.radians(az_max), H + 1)
    r_edges = np.linspace(0.0, r_max, W + 1)

    R, Theta = np.meshgrid(r_edges, az_edges)
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)

    return X, Y


def draw_sector_boundary(ax, r_max=60.0, az_min=-30.0, az_max=30.0, color="0.25"):
    theta_l = np.radians(az_min)
    theta_r = np.radians(az_max)

    ax.plot(
        [0, r_max * np.sin(theta_l)],
        [0, r_max * np.cos(theta_l)],
        color=color,
        lw=0.55,
    )
    ax.plot(
        [0, r_max * np.sin(theta_r)],
        [0, r_max * np.cos(theta_r)],
        color=color,
        lw=0.55,
    )

    arc = np.linspace(theta_l, theta_r, 180)
    ax.plot(r_max * np.sin(arc), r_max * np.cos(arc), color=color, lw=0.55)


def add_scale_bar(ax, length_km=10.0):
    x0, y0 = -28.0, 2.7

    backing = patches.Rectangle(
        (x0 - 1.0, y0 - 1.0),
        length_km + 2.0,
        3.4,
        facecolor="black",
        edgecolor="none",
        alpha=0.30,
        zorder=18,
    )
    ax.add_patch(backing)

    ax.plot(
        [x0, x0 + length_km],
        [y0, y0],
        color="white",
        lw=1.8,
        solid_capstyle="butt",
        zorder=20,
    )

    ax.text(
        x0 + length_km / 2,
        y0 + 1.0,
        f"{int(length_km)} km",
        color="white",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        zorder=21,
    )


def plot_polar_field(ax, data_2d, cmap, norm, column_title=None, add_scale=False):
    H, W = data_2d.shape
    X, Y = get_polar_grid(H=H, W=W)

    mesh = ax.pcolormesh(
        X,
        Y,
        np.ma.masked_invalid(data_2d),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    if column_title is not None:
        ax.set_title(column_title, fontsize=8.1, fontweight="normal", pad=3.0)

    ax.set_aspect("equal")
    ax.set_xlim(X.min() - 1.0, X.max() + 1.0)
    ax.set_ylim(Y.min() - 1.0, Y.max() + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    draw_sector_boundary(ax)

    if add_scale:
        add_scale_bar(ax)

    return mesh


def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        zorder=30,
    )


def plot_case_card(ax, case):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if case["case_type"] == "TOR":
        title = "TOR case"
        subtitle = "localized rotation"
        face = "#F8E8E8"
        color = "#B2182B"
    else:
        title = "WRN case"
        subtitle = "false-alarm prone"
        face = "#E8F0FA"
        color = "#2166AC"

    card = patches.FancyBboxPatch(
        (0.05, 0.10),
        0.90,
        0.80,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=face,
        edgecolor="none",
        alpha=1.0,
    )
    ax.add_patch(card)

    ax.text(
        0.50,
        0.66,
        title,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=color,
    )

    ax.text(
        0.50,
        0.45,
        subtitle,
        ha="center",
        va="center",
        fontsize=6.3,
        color="0.25",
    )

    prob_text = (
        f"CBAM {case['prob_cbam']:.2f}\n"
        f"SE {case['prob_se']:.2f}\n"
        f"PI {case['prob_pi']:.2f}"
    )

    ax.text(
        0.50,
        0.22,
        prob_text,
        ha="center",
        va="center",
        fontsize=6.1,
        color="0.18",
        fontweight="bold",
        linespacing=1.15,
    )


# ============================================================
# Model loading and inference
# ============================================================

def infer_attn_type_from_checkpoint(model_name, state_dict):
    model_name_lower = model_name.lower()
    keys_lower = [k.lower() for k in state_dict.keys()]

    if "cbam" in model_name_lower or any("cbam" in k for k in keys_lower):
        return "cbam"

    if "se" in model_name_lower:
        return "se"

    if any("physics_attn_module.fc" in k for k in state_dict.keys()):
        return "se"

    return "physics"


def load_ablation_model(ckpt_path, model_name, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    has_rot = any("rot_proj" in k for k in state_dict.keys())
    attn_type = infer_attn_type_from_checkpoint(model_name, state_dict)

    train_compare.args = MockArgs(
        attn_type=attn_type,
        disable_physics_attn=False,
        disable_rot_stats=not has_rot,
    )

    model = train_compare.PI_STDNet_Ablation().to(device)

    # Same convention used in your previous scripts.
    model.use_train_py_order = "PI" in model_name or "Ours" in model_name

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"\nLoaded {model_name}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  inferred attn_type: {attn_type}")
    print(f"  has_rot: {has_rot}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    return model


def predict_and_attention(model, image_tensor, device, target_shape):
    x = image_tensor.unsqueeze(0).to(device)

    amp_enabled = device.type == "cuda" and train_compare.CONFIG.get("USE_AMP", False)

    if device.type == "cuda":
        ctx = torch.amp.autocast("cuda", enabled=amp_enabled)
    else:
        ctx = nullcontext()

    with torch.no_grad():
        with ctx:
            cls_logit, aux_logit, _, attention = model(x)

            prob = (
                0.6 * torch.sigmoid(cls_logit)
                + 0.4 * torch.sigmoid(aux_logit)
            ).item()

    attn = attention[0, 0].detach().cpu().float()

    if tuple(attn.shape) != tuple(target_shape):
        attn = F.interpolate(
            attn.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return prob, attn.numpy()


# ============================================================
# Data extraction
# ============================================================

def extract_z_v_from_ablation_tensor(image_tensor):
    """
    Matches your original figure7.py convention:

        images[0] -> DBZ normalized to [0, 1]
        images[1] -> VEL normalized to [0, 1]

    Converted back to:
        Z  = images[0] * 110 - 30
        Vr = images[1] * 100 - 50
    """
    arr = image_tensor.numpy()

    z_data = arr[0] * 110.0 - 30.0
    v_data = arr[1] * 100.0 - 50.0

    return z_data, v_data


def build_case(
    dataset,
    idx,
    case_type,
    models,
    device,
    attn_scale,
    attn_smooth,
    attn_top_percent,
):
    image_tensor, _, cat_tensor = dataset[idx]

    z_data, v_data = extract_z_v_from_ablation_tensor(image_tensor)
    target_shape = z_data.shape

    prob_cbam, attn_cbam = predict_and_attention(
        models["CBAM"],
        image_tensor,
        device,
        target_shape,
    )
    prob_se, attn_se = predict_and_attention(
        models["SE-Net"],
        image_tensor,
        device,
        target_shape,
    )
    prob_pi, attn_pi = predict_and_attention(
        models["PI-STDNet"],
        image_tensor,
        device,
        target_shape,
    )

    case = {
        "idx": idx,
        "case_type": case_type,
        "cat_id": int(cat_tensor.item()),
        "z": z_data,
        "vr": v_data,
        "attn_cbam": prepare_attention(
            attn_cbam,
            scale=attn_scale,
            smooth_sigma=attn_smooth,
            top_percent=attn_top_percent,
        ),
        "attn_se": prepare_attention(
            attn_se,
            scale=attn_scale,
            smooth_sigma=attn_smooth,
            top_percent=attn_top_percent,
        ),
        "attn_pi": prepare_attention(
            attn_pi,
            scale=attn_scale,
            smooth_sigma=attn_smooth,
            top_percent=attn_top_percent,
        ),
        "prob_cbam": prob_cbam,
        "prob_se": prob_se,
        "prob_pi": prob_pi,
    }

    return case


# ============================================================
# Figure rendering
# ============================================================

class AttentionPlotterNatureV3:
    def __init__(self, save_dir="paper_figures_fig7_nature_v3"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_nature_style()

        self.dbz_cmap, self.dbz_norm = get_dbz_cmap()
        self.vel_cmap, self.vel_norm = get_vel_cmap()
        self.attn_cmap, self.attn_norm = get_attention_cmap()

    def plot(self, cases):
        if len(cases) != 2:
            raise ValueError("Fig. 7 expects exactly two cases: one TOR and one WRN.")

        fig = plt.figure(figsize=(8.25, 3.55))

        gs = GridSpec(
            3,
            6,
            height_ratios=[1.0, 1.0, 0.075],
            width_ratios=[0.78, 1.06, 1.06, 1.06, 1.06, 1.06],
            hspace=0.13,
            wspace=0.08,
            left=0.030,
            right=0.992,
            top=0.925,
            bottom=0.130,
        )

        column_titles = [
            None,
            "Reflectivity $Z$",
            "Radial velocity $V_r$",
            "CBAM attention",
            "SE attention",
            "PAM",
        ]

        panel_letters = list("abcdefghij")

        im_z = None
        im_v = None
        im_a = None

        metadata_rows = []

        for row_idx, case in enumerate(cases):
            ax_card = fig.add_subplot(gs[row_idx, 0])
            plot_case_card(ax_card, case)

            fields = [
                ("z", case["z"], self.dbz_cmap, self.dbz_norm, True),
                ("vr", case["vr"], self.vel_cmap, self.vel_norm, True),
                ("attn_cbam", case["attn_cbam"], self.attn_cmap, self.attn_norm, False),
                ("attn_se", case["attn_se"], self.attn_cmap, self.attn_norm, False),
                ("attn_pi", case["attn_pi"], self.attn_cmap, self.attn_norm, False),
            ]

            for col_idx, (name, data, cmap, norm, show_scale) in enumerate(fields, start=1):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                title = column_titles[col_idx] if row_idx == 0 else None

                mesh = plot_polar_field(
                    ax,
                    data,
                    cmap,
                    norm,
                    column_title=title,
                    add_scale=show_scale,
                )

                label_id = row_idx * 5 + (col_idx - 1)
                add_panel_label(ax, f"({panel_letters[label_id]})")

                if name == "z":
                    im_z = mesh
                elif name == "vr":
                    im_v = mesh
                else:
                    im_a = mesh

            metadata_rows.append({
                "row": row_idx,
                "dataset_index": case["idx"],
                "case_type": case["case_type"],
                "cat_id": case["cat_id"],
                "prob_cbam": case["prob_cbam"],
                "prob_se": case["prob_se"],
                "prob_pi": case["prob_pi"],
            })

        # Shared colorbars.
        cax_blank = fig.add_subplot(gs[2, 0])
        cax_blank.axis("off")

        cax_z = fig.add_subplot(gs[2, 1])
        cax_v = fig.add_subplot(gs[2, 2])
        cax_a = fig.add_subplot(gs[2, 3:6])

        cb_z = fig.colorbar(im_z, cax=cax_z, orientation="horizontal")
        cb_z.set_label("Reflectivity $Z$ (dBZ)", fontsize=6.9, labelpad=1.8)
        cb_z.set_ticks([0, 20, 40, 60])
        cb_z.ax.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.5)
        cb_z.outline.set_linewidth(0.5)

        cb_v = fig.colorbar(im_v, cax=cax_v, orientation="horizontal")
        cb_v.set_label("Radial velocity $V_r$ (m s$^{-1}$)", fontsize=6.9, labelpad=1.8)
        cb_v.set_ticks([-30, 0, 30])
        cb_v.ax.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.5)
        cb_v.outline.set_linewidth(0.5)

        cb_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal")
        cb_a.set_label("Attention response", fontsize=6.9, labelpad=1.8)
        cb_a.set_ticks([0.0, 0.5, 1.0])
        cb_a.ax.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.5)
        cb_a.outline.set_linewidth(0.5)

        pdf_path = self.save_dir / "Figure7_Attention_Nature_v3.pdf"
        png_path = self.save_dir / "Figure7_Attention_Nature_v3.png"
        csv_path = self.save_dir / "Figure7_Attention_Nature_v3_metadata.csv"

        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        pd.DataFrame(metadata_rows).to_csv(csv_path, index=False)

        print(f"Saved: {pdf_path}")
        print(f"Saved: {png_path}")
        print(f"Saved: {csv_path}")


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    catalog_path = args.catalog_path or train_compare.CONFIG["CATALOG_PATH"]
    data_root = args.data_root or train_compare.CONFIG["DATA_ROOT"]

    dataset = train_compare.TorNetDatasetAblation(
        catalog_path,
        data_root,
        mode="test",
    )

    models = {
        "CBAM": load_ablation_model(args.ckpt_cbam, "CBAM", device),
        "SE-Net": load_ablation_model(args.ckpt_se, "SE-Net", device),
        "PI-STDNet": load_ablation_model(args.ckpt_ours, "PI-STDNet (Ours)", device),
    }

    tor_case = build_case(
        dataset=dataset,
        idx=args.tor_idx,
        case_type="TOR",
        models=models,
        device=device,
        attn_scale=args.attn_scale,
        attn_smooth=args.attn_smooth,
        attn_top_percent=args.attn_top_percent,
    )

    wrn_case = build_case(
        dataset=dataset,
        idx=args.wrn_idx,
        case_type="WRN",
        models=models,
        device=device,
        attn_scale=args.attn_scale,
        attn_smooth=args.attn_smooth,
        attn_top_percent=args.attn_top_percent,
    )

    plotter = AttentionPlotterNatureV3(save_dir=args.save_dir)
    plotter.plot([tor_case, wrn_case])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--catalog_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="paper_figures_fig7_nature_v3")

    parser.add_argument("--ckpt_cbam", type=str, default="ablation_attn_cbam_best.pth")
    parser.add_argument("--ckpt_se", type=str, default="ablation_attn_se_best.pth")
    parser.add_argument("--ckpt_ours", type=str, default="best_pi_v8_GWP1.pth")

    parser.add_argument("--tor_idx", type=int, default=2075)
    parser.add_argument("--wrn_idx", type=int, default=27306)

    parser.add_argument("--attn_scale", type=float, default=2.0)
    parser.add_argument("--attn_smooth", type=float, default=1.0)
    parser.add_argument(
        "--attn_top_percent",
        type=float,
        default=0.0,
        help="0 keeps full attention map; 20 keeps only top 20 percent response.",
    )

    args = parser.parse_args()
    main(args)


