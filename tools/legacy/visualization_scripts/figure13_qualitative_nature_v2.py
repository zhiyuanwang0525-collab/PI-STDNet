# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
"""
Fig. 13 qualitative visualization and error analysis.

Revised version:
- 4 case types: hit, correct rejection, false alarm, false negative
- 4 displayed fields: reflectivity Z, low-level radial velocity Vr,
  PAM attention, and final spatial prediction
- Nature-style compact layout
- Output files are written to a new folder by default

Usage:
  python figure13_qualitative_nature_v2.py ^
      --model_path best_pi_v8_server_fulldata.pth ^
      --catalog_path /path/to/TorNet/catalog.csv ^
      --data_root /path/to/TorNet/ ^
      --save_dir paper_figures_fig13_nature_v2
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


# ============================================================
# Import model and dataset
# ============================================================

try:
    from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG
except ImportError:
    print("ERROR: Cannot import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG from train.py.")
    print("Please make sure train.py is in the same directory.")
    raise


# ============================================================
# Figure style
# ============================================================

def set_nature_style():
    """
    Clean Nature-like figure style.
    The goal is a light, compact, and consistent multi-panel layout.
    """
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
    cmap = mcolors.LinearSegmentedColormap.from_list("attention_nature", colors, N=256)
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    return cmap, norm


def mask_top_fraction(data, top_percent=15.0):
    """
    Keep only the strongest top_percent response for cleaner qualitative display.
    The colorbar remains 0-1, but low-response regions are transparent/white.
    """
    data = np.asarray(data, dtype=np.float32)
    valid = data[np.isfinite(data)]

    if valid.size == 0:
        return np.full_like(data, np.nan)

    threshold = np.percentile(valid, 100.0 - top_percent)
    masked = data.copy()
    masked[masked < threshold] = np.nan
    return masked


# ============================================================
# Radar-sector plotting
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


def plot_polar_channel(ax, data_2d, cmap, norm, column_title=None):
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
        ax.set_title(column_title, fontsize=8.2, fontweight="normal", pad=3.0)

    ax.set_aspect("equal")
    ax.set_xlim(X.min() - 1.0, X.max() + 1.0)
    ax.set_ylim(Y.min() - 1.0, Y.max() + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    draw_sector_boundary(ax)
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
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.1),
        zorder=30,
    )


def plot_case_card(ax, case_type, prob, ef):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    styles = {
        "HIT": {
            "title": "Hit",
            "subtitle": "tornado detected",
            "color": "#1B7837",
            "face": "#E8F3E8",
            "meta_type": "TOR",
        },
        "CR": {
            "title": "Correct\nrejection",
            "subtitle": "WRN suppressed",
            "color": "#2166AC",
            "face": "#E8F0FA",
            "meta_type": "WRN",
        },
        "FA": {
            "title": "False\nalarm",
            "subtitle": "non-tornadic shear",
            "color": "#B2182B",
            "face": "#F8E8E8",
            "meta_type": "WRN",
        },
        "FN": {
            "title": "False\nnegative",
            "subtitle": "missed tornado",
            "color": "#E66101",
            "face": "#FFF0DF",
            "meta_type": "TOR",
        },
    }

    st = styles[case_type]

    card = patches.FancyBboxPatch(
        (0.055, 0.105),
        0.89,
        0.79,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=st["face"],
        edgecolor="none",
        alpha=1.0,
    )
    ax.add_patch(card)

    ax.text(
        0.50,
        0.64,
        st["title"],
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=st["color"],
        linespacing=1.08,
    )

    ax.text(
        0.50,
        0.39,
        st["subtitle"],
        ha="center",
        va="center",
        fontsize=6.25,
        color="0.25",
    )

    if st["meta_type"] == "TOR":
        ef_text = f"EF{ef}" if ef >= 0 else "EF n/a"
    else:
        ef_text = "WRN"

    ax.text(
        0.50,
        0.22,
        f"P={prob:.2f} | {ef_text}",
        ha="center",
        va="center",
        fontsize=6.35,
        fontweight="bold",
        color="0.18",
    )


# ============================================================
# Metadata
# ============================================================

def get_ef_rating(catalog_df, idx):
    ef_candidates = [
        "ef_number",
        "ef_rating",
        "EF",
        "ef",
        "mag",
        "ef_scale",
        "EF_RATING",
        "tornado_rating",
    ]

    for col in ef_candidates:
        if col in catalog_df.columns:
            val = catalog_df.iloc[idx][col]
            if pd.notna(val):
                val_str = str(val).strip().upper().replace("EF", "").replace("-", "").replace(" ", "")
                try:
                    return int(float(val_str))
                except (ValueError, TypeError):
                    pass

    return -1


# ============================================================
# Model and data helpers
# ============================================================

def load_model(model_path, device):
    print("Loading PI-STDNet model...")

    model = PI_STDNet_V8().to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Cannot find model checkpoint: {model_path}")

    model.eval()
    return model


def get_nc_path(data_root, row):
    year = pd.to_datetime(row["start_time"]).year if "start_time" in row else 2013

    path1 = os.path.join(data_root, f"tornet_{year}", row["filename"])
    path2 = os.path.join(data_root, row["filename"])

    if os.path.exists(path1):
        return path1
    if os.path.exists(path2):
        return path2

    raise FileNotFoundError(f"Cannot find file: {path1} or {path2}")


def load_raw_radar_fields(data_root, row):
    nc_path = get_nc_path(data_root, row)

    with xr.open_dataset(nc_path, engine="netcdf4", cache=False) as ds:
        dbz_raw = ds["DBZ"].values
        vel_raw = ds["VEL"].values

        t_idx = min(3, dbz_raw.shape[0] - 1)

        dbz_05 = np.nan_to_num(dbz_raw[t_idx, :, :, 0], nan=-30.0)
        vel_05 = np.nan_to_num(vel_raw[t_idx, :, :, 0], nan=0.0)

    return dbz_05, vel_05


# ============================================================
# Case collection
# ============================================================

def collect_cases(args, model, dataset, test_catalog, device):
    collectors = {
        "HIT": [],
        "CR": [],
        "FA": [],
        "FN": [],
    }

    optimal_thresh = args.threshold

    scan_limit = len(dataset) if args.scan_limit <= 0 else min(len(dataset), args.scan_limit)

    amp_enabled = device.type == "cuda" and CONFIG.get("USE_AMP", False)

    print(f"Scanning test samples: {scan_limit}")

    with torch.no_grad():
        for i in tqdm(range(scan_limit), desc="Scanning Fig. 13 cases"):
            all_done = all(len(v) >= args.max_per_type for v in collectors.values())
            if all_done:
                break

            try:
                image, label_tensor, cat_tensor = dataset[i]
            except Exception:
                continue

            label = int(label_tensor.item())
            cat_id = int(cat_tensor.item())

            img_batch = image.unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                cls_logit, aux_logit, spatial_logits, physics_attn = model(img_batch)

            prob = (
                0.6 * torch.sigmoid(cls_logit)
                + 0.4 * torch.sigmoid(aux_logit)
            ).item()

            pred = 1 if prob >= optimal_thresh else 0

            case_type = None

            if pred == 1 and label == 1:
                case_type = "HIT"
            elif pred == 0 and label == 0 and cat_id == 1:
                case_type = "CR"
            elif pred == 1 and label == 0:
                case_type = "FA"
            elif pred == 0 and label == 1:
                case_type = "FN"

            if case_type is None:
                continue

            if len(collectors[case_type]) >= args.max_per_type:
                continue

            try:
                row = dataset.catalog.iloc[i]
                dbz_05, vel_05 = load_raw_radar_fields(args.data_root, row)
            except Exception:
                continue

            target_shape = dbz_05.shape

            pam = F.interpolate(
                physics_attn,
                size=target_shape,
                mode="bilinear",
                align_corners=True,
            )[0, 0].detach().cpu().numpy()

            spatial_pred = torch.sigmoid(spatial_logits[0, 0]).detach().cpu().numpy()

            if spatial_pred.shape != target_shape:
                spatial_pred = F.interpolate(
                    torch.from_numpy(spatial_pred).float().unsqueeze(0).unsqueeze(0),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=True,
                )[0, 0].numpy()

            ef_rating = get_ef_rating(test_catalog, i) if i < len(test_catalog) else -1

            collectors[case_type].append({
                "idx": i,
                "prob": prob,
                "ef": ef_rating,
                "dbz_05": dbz_05,
                "vel_05": vel_05,
                "physics_attn": np.clip(pam, 0.0, 1.0),
                "spatial_pred": np.clip(spatial_pred, 0.0, 1.0),
            })

    return collectors


def select_representative_cases(collectors):
    """
    Pick one representative case for each row.

    HIT: highest probability hit
    CR: hardest correct rejection, i.e., highest probability below threshold
    FA: highest-probability false alarm
    FN: hardest missed tornado, i.e., highest probability below threshold
    """
    selected = []

    if collectors["HIT"]:
        selected.append(("HIT", max(collectors["HIT"], key=lambda x: x["prob"])))

    if collectors["CR"]:
        selected.append(("CR", max(collectors["CR"], key=lambda x: x["prob"])))

    if collectors["FA"]:
        selected.append(("FA", max(collectors["FA"], key=lambda x: x["prob"])))

    if collectors["FN"]:
        selected.append(("FN", max(collectors["FN"], key=lambda x: x["prob"])))

    return selected


# ============================================================
# Figure rendering
# ============================================================

def render_figure13(selected, args):
    if len(selected) < 2:
        raise RuntimeError("Too few selected cases. Increase scan_limit or max_per_type.")

    set_nature_style()

    dbz_cmap, dbz_norm = get_dbz_cmap()
    vel_cmap, vel_norm = get_vel_cmap()
    attn_cmap, attn_norm = get_attention_cmap()

    n_rows = len(selected)

    fig = plt.figure(figsize=(7.35, 1.48 * n_rows + 0.50))

    gs = GridSpec(
        n_rows + 1,
        5,
        height_ratios=[*([1.0] * n_rows), 0.070],
        width_ratios=[0.74, 1.12, 1.12, 1.12, 1.12],
        hspace=0.105,
        wspace=0.075,
        left=0.030,
        right=0.992,
        top=0.925,
        bottom=0.108,
    )

    column_titles = [
        None,
        "Reflectivity $Z$",
        "Radial velocity $V_r$",
        "PAM",
        "Prediction",
    ]

    panel_letters = list("abcdefghijklmnopqrstuvwxyz")

    axes = []

    im_z = im_v = im_pam = im_pred = None

    metadata_rows = []

    for row_idx, (case_type, case) in enumerate(selected):
        row_axes = []

        ax_label = fig.add_subplot(gs[row_idx, 0])
        plot_case_card(ax_label, case_type, case["prob"], case["ef"])
        row_axes.append(ax_label)

        fields = [
            ("Z", case["dbz_05"], dbz_cmap, dbz_norm),
            ("Vr", case["vel_05"], vel_cmap, vel_norm),
            ("PAM", mask_top_fraction(case["physics_attn"], args.top_percent), attn_cmap, attn_norm),
            ("Prediction", mask_top_fraction(case["spatial_pred"], args.top_percent), attn_cmap, attn_norm),
        ]

        for col_idx, (_, field, cmap, norm) in enumerate(fields, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            title = column_titles[col_idx] if row_idx == 0 else None
            mesh = plot_polar_channel(ax, field, cmap, norm, column_title=title)

            panel_id = row_idx * 4 + (col_idx - 1)
            add_panel_label(ax, f"({panel_letters[panel_id]})")

            if col_idx == 1:
                im_z = mesh
            elif col_idx == 2:
                im_v = mesh
            elif col_idx == 3:
                im_pam = mesh
            elif col_idx == 4:
                im_pred = mesh

            row_axes.append(ax)

        axes.append(row_axes)

        metadata_rows.append({
            "row": row_idx,
            "case_type": case_type,
            "dataset_index": case["idx"],
            "probability": case["prob"],
            "ef": case["ef"],
        })

    # Shared colorbars
    cax_z = fig.add_subplot(gs[n_rows, 1])
    cax_v = fig.add_subplot(gs[n_rows, 2])
    cax_a = fig.add_subplot(gs[n_rows, 3:5])

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

    cb_a = fig.colorbar(im_pam, cax=cax_a, orientation="horizontal")
    cb_a.set_label("PAM / prediction response", fontsize=6.9, labelpad=1.8)
    cb_a.set_ticks([0.0, 0.5, 1.0])
    cb_a.ax.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.5)
    cb_a.outline.set_linewidth(0.5)

    blank = fig.add_subplot(gs[n_rows, 0])
    blank.axis("off")

    os.makedirs(args.save_dir, exist_ok=True)

    pdf_path = os.path.join(args.save_dir, "Figure13_Qualitative_Nature_v2.pdf")
    png_path = os.path.join(args.save_dir, "Figure13_Qualitative_Nature_v2.png")
    csv_path = os.path.join(args.save_dir, "Figure13_Qualitative_Nature_v2_metadata.csv")

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

    os.makedirs(args.save_dir, exist_ok=True)

    model = load_model(args.model_path, device)

    if not os.path.exists(args.catalog_path):
        raise FileNotFoundError(f"Cannot find catalog file: {args.catalog_path}")

    dataset = TorNetDatasetV8_Full(
        catalog_path=args.catalog_path,
        root_dir=args.data_root,
        mode="test",
    )

    df_full = pd.read_csv(args.catalog_path)
    test_catalog = df_full[df_full["type"] == "test"].reset_index(drop=True)

    collectors = collect_cases(args, model, dataset, test_catalog, device)
    selected = select_representative_cases(collectors)

    print("\nSelected cases:")
    for case_type, case in selected:
        print(
            f"  {case_type}: idx={case['idx']}, "
            f"P={case['prob']:.3f}, EF={case['ef']}"
        )

    render_figure13(selected, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="best_pi_v8_GWP.pth")
    parser.add_argument("--catalog_path", type=str, default="/path/to/TorNet/catalog.csv")
    parser.add_argument("--data_root", type=str, default="/path/to/TorNet/")
    parser.add_argument("--save_dir", type=str, default="paper_figures_fig13_nature_v2")

    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max_per_type", type=int, default=10)
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=5000,
        help="Number of test samples to scan. Use 0 to scan the full test set.",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=15.0,
        help="Top response percentage retained for PAM and prediction maps.",
    )

    args = parser.parse_args()
    main(args)
