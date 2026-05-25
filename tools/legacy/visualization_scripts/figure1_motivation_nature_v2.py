# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import PI_STDNet_V8, TorNetDatasetV8_Full, CONFIG


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


def get_prob_cmap():
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
    cmap = mcolors.LinearSegmentedColormap.from_list("prob_response", colors, N=256)
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    return cmap, norm


# ============================================================
# Radar geometry
# ============================================================

def get_polar_grid(H=120, W=240, r_max=60.0, az_min=-30.0, az_max=30.0):
    az_edges = np.linspace(np.radians(az_min), np.radians(az_max), H + 1)
    r_edges = np.linspace(0.0, r_max, W + 1)

    R, Theta = np.meshgrid(r_edges, az_edges)
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)

    az_centers = np.linspace(np.radians(az_min), np.radians(az_max), H)
    r_centers = np.linspace(0.0, r_max, W)

    return X, Y, az_centers, r_centers


def cell_to_xy(row, col, az_centers, r_centers):
    row = int(np.clip(row, 0, len(az_centers) - 1))
    col = int(np.clip(col, 0, len(r_centers) - 1))

    theta = az_centers[row]
    radius = r_centers[col]

    x = radius * np.sin(theta)
    y = radius * np.cos(theta)

    return x, y


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


def plot_polar_field(ax, data_2d, cmap, norm, column_title=None):
    H, W = data_2d.shape
    X, Y, _, _ = get_polar_grid(H=H, W=W)

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


def draw_focus_box(
    ax,
    center_row,
    center_col,
    shape,
    half_y=10,
    half_x=16,
):
    """
    Draw a high-contrast focus box without a center cross.

    The box uses a black outer stroke and a white inner stroke, so it remains
    visible on both reflectivity, velocity, and prediction-response panels.
    The center cross is intentionally removed to avoid blocking local radar
    structures.
    """
    H, W = shape
    _, _, az_centers, r_centers = get_polar_grid(H=H, W=W)

    corners = [
        (center_row - half_y, center_col - half_x),
        (center_row - half_y, center_col + half_x),
        (center_row + half_y, center_col + half_x),
        (center_row + half_y, center_col - half_x),
    ]

    xy = [cell_to_xy(r, c, az_centers, r_centers) for r, c in corners]

    # Black outer outline.
    outer = patches.Polygon(
        xy,
        closed=True,
        facecolor="none",
        edgecolor="black",
        lw=1.8,
        zorder=15,
    )

    # White inner outline.
    inner = patches.Polygon(
        xy,
        closed=True,
        facecolor="none",
        edgecolor="white",
        lw=0.9,
        zorder=16,
    )

    ax.add_patch(outer)
    ax.add_patch(inner)

    # x, y = cell_to_xy(center_row, center_col, az_centers, r_centers)
    # ax.plot(x, y, marker="+", color="black", ms=7.5, mew=1.9, zorder=17)
    # ax.plot(x, y, marker="+", color="white", ms=6.0, mew=1.2, zorder=18)


# ============================================================
# Case cards
# ============================================================

def plot_case_card(ax, case_type, prob):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if case_type == "WRN":
        title = "WRN storm"
        subtitle = "low model response"
        color = "#2166AC"
        face = "#E8F0FA"
    else:
        title = "TOR case"
        subtitle = "localized response"
        color = "#B2182B"
        face = "#F8E8E8"

    card = patches.FancyBboxPatch(
        (0.055, 0.105),
        0.89,
        0.79,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=face,
        edgecolor="none",
        alpha=1.0,
    )
    ax.add_patch(card)

    ax.text(
        0.50,
        0.62,
        title,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=color,
    )

    ax.text(
        0.50,
        0.40,
        subtitle,
        ha="center",
        va="center",
        fontsize=6.3,
        color="0.25",
    )

    ax.text(
        0.50,
        0.23,
        f"P={prob:.2f}",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="0.18",
    )


# ============================================================
# Model and data
# ============================================================

def load_model(model_path, device):
    model = PI_STDNet_V8().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_amp_context(device):
    amp_enabled = device.type == "cuda" and CONFIG.get("USE_AMP", False)

    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=amp_enabled)

    return nullcontext()


def run_model(model, inputs, device):
    inputs = inputs.to(device)

    with torch.no_grad():
        with get_amp_context(device):
            cls_logit, aux_logit, spatial_logits, _ = model(inputs)

    image_prob = (
        0.6 * torch.sigmoid(cls_logit)
        + 0.4 * torch.sigmoid(aux_logit)
    ).item()

    prob_map = torch.sigmoid(spatial_logits).squeeze().detach().cpu().numpy()

    return image_prob, prob_map


def extract_display_fields(inputs):
    """
    Keep the original Fig. 1 channel convention from figure1-2.py:
        inputs[0, 33] -> reflectivity Z
        inputs[0, 34] -> radial velocity Vr
    """
    raw_z = inputs[0, 33, :, :].detach().cpu().numpy() * 110.0 - 30.0
    raw_v = inputs[0, 34, :, :].detach().cpu().numpy() * 100.0 - 50.0

    return raw_z, raw_v


def clean_probability_map(prob_map, z_data):
    prob = prob_map.copy()
    prob[z_data < 10.0] = 0.0
    prob[prob < 0.20] = np.nan
    prob = np.clip(prob, 0.0, 1.0)
    return prob


def build_case_from_dataset(dataset, idx, model, device):
    image, label, cat_id = dataset[idx]
    inputs = image.unsqueeze(0)

    image_prob, prob_map = run_model(model, inputs, device)
    z_data, v_data = extract_display_fields(inputs)

    cat_val = int(cat_id.item())
    case_type = "TOR" if cat_val == 2 else "WRN"

    if case_type == "TOR":
        center_y, center_x = np.unravel_index(np.nanargmax(prob_map), prob_map.shape)
    else:
        smoothed_z = ndimage.gaussian_filter(z_data, sigma=4)
        center_y, center_x = np.unravel_index(np.nanargmax(smoothed_z), smoothed_z.shape)

    return {
        "idx": idx,
        "case_type": case_type,
        "prob": image_prob,
        "Z": z_data,
        "V": v_data,
        "Prob": clean_probability_map(prob_map, z_data),
        "center_y": int(center_y),
        "center_x": int(center_x),
    }


# ============================================================
# Automatic case selection
# ============================================================

def find_cases(args, dataset, model, device):
    wrn_case = None
    tor_case = None

    if args.wrn_idx >= 0:
        wrn_case = build_case_from_dataset(dataset, args.wrn_idx, model, device)
        wrn_case["case_type"] = "WRN"

    if args.tor_idx >= 0:
        tor_case = build_case_from_dataset(dataset, args.tor_idx, model, device)
        tor_case["case_type"] = "TOR"

    if wrn_case is not None and tor_case is not None:
        return wrn_case, tor_case

    scan_limit = len(dataset) if args.scan_limit <= 0 else min(len(dataset), args.scan_limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Scanning {scan_limit} test samples for Fig. 1 cases...")

    for i, (inputs, labels, cat_ids) in enumerate(tqdm(loader, desc="Scanning Fig. 1 cases")):
        if i >= scan_limit:
            break

        cat_val = int(cat_ids[0].item())

        image_prob, prob_map = run_model(model, inputs, device)
        z_data, v_data = extract_display_fields(inputs)

        # WRN: strong reflectivity but low local PI-STDNet response.
        if wrn_case is None and cat_val == 1:
            if np.sum(z_data > 40) > 250:
                smoothed_z = ndimage.gaussian_filter(z_data, sigma=4)
                center_y, center_x = np.unravel_index(np.nanargmax(smoothed_z), smoothed_z.shape)

                if 20 < center_x < 220 and 20 < center_y < 100:
                    local_prob = prob_map[
                        max(0, center_y - 10):min(120, center_y + 10),
                        max(0, center_x - 10):min(240, center_x + 10),
                    ]

                    if np.nanmax(local_prob) < args.wrn_local_prob_max:
                        wrn_case = {
                            "idx": i,
                            "case_type": "WRN",
                            "prob": image_prob,
                            "Z": z_data,
                            "V": v_data,
                            "Prob": clean_probability_map(prob_map, z_data),
                            "center_y": int(center_y),
                            "center_x": int(center_x),
                        }
                        print(
                            f"Selected WRN case: idx={i}, "
                            f"P={image_prob:.3f}, local max={np.nanmax(local_prob):.3f}"
                        )

        # TOR: high image probability and robust local velocity contrast.
        if tor_case is None and cat_val == 2 and image_prob > args.tor_prob_min:
            center_y, center_x = np.unravel_index(np.nanargmax(prob_map), prob_map.shape)

            if 20 < center_x < 220 and 20 < center_y < 100:
                local_z = z_data[
                    max(0, center_y - 3):min(120, center_y + 3),
                    max(0, center_x - 3):min(240, center_x + 3),
                ]

                local_v = v_data[
                    max(0, center_y - 8):min(120, center_y + 8),
                    max(0, center_x - 8):min(240, center_x + 8),
                ]

                if local_z.size > 0 and local_v.size > 0:
                    v_diff = np.nanpercentile(local_v, 98) - np.nanpercentile(local_v, 2)

                    if np.nanmax(local_z) > 20 and v_diff > args.tor_vdiff_min:
                        tor_case = {
                            "idx": i,
                            "case_type": "TOR",
                            "prob": image_prob,
                            "Z": z_data,
                            "V": v_data,
                            "Prob": clean_probability_map(prob_map, z_data),
                            "center_y": int(center_y),
                            "center_x": int(center_x),
                        }
                        print(
                            f"Selected TOR case: idx={i}, "
                            f"P={image_prob:.3f}, robust ΔV={v_diff:.1f} m/s"
                        )

        if wrn_case is not None and tor_case is not None:
            break

    if wrn_case is None or tor_case is None:
        raise RuntimeError(
            "Could not find both WRN and TOR cases. "
            "Try increasing --scan_limit or relaxing thresholds."
        )

    return wrn_case, tor_case


# ============================================================
# Figure rendering
# ============================================================

class Figure1MotivationNature:
    def __init__(self, save_dir="paper_figures_fig1_nature_v2"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_nature_style()

        self.dbz_cmap, self.dbz_norm = get_dbz_cmap()
        self.vel_cmap, self.vel_norm = get_vel_cmap()
        self.prob_cmap, self.prob_norm = get_prob_cmap()

    def plot(self, wrn_case, tor_case):
        fig = plt.figure(figsize=(7.35, 3.65))

        gs = GridSpec(
            3,
            4,
            height_ratios=[1.0, 1.0, 0.075],
            width_ratios=[0.72, 1.18, 1.18, 1.18],
            hspace=0.14,
            wspace=0.08,
            left=0.030,
            right=0.992,
            top=0.925,
            bottom=0.135,
        )

        cases = [wrn_case, tor_case]
        column_titles = [
            None,
            "Reflectivity $Z$",
            "Radial velocity $V_r$",
            "PI-STDNet response",
        ]

        panel_letters = list("abcdef")

        im_z = None
        im_v = None
        im_p = None

        metadata_rows = []

        for row_idx, case in enumerate(cases):
            ax_card = fig.add_subplot(gs[row_idx, 0])
            plot_case_card(ax_card, case["case_type"], case["prob"])

            fields = [
                ("Z", case["Z"], self.dbz_cmap, self.dbz_norm),
                ("V", case["V"], self.vel_cmap, self.vel_norm),
                ("Prob", case["Prob"], self.prob_cmap, self.prob_norm),
            ]

            for col_idx, (name, data, cmap, norm) in enumerate(fields, start=1):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                title = column_titles[col_idx] if row_idx == 0 else None

                mesh = plot_polar_field(
                    ax,
                    data,
                    cmap,
                    norm,
                    column_title=title,
                )

                label_id = row_idx * 3 + (col_idx - 1)
                add_panel_label(ax, f"({panel_letters[label_id]})")

                draw_focus_box(
                    ax,
                    case["center_y"],
                    case["center_x"],
                    shape=case["Z"].shape,
                    half_y=10,
                    half_x=16,
                )

                if name == "Z":
                    im_z = mesh
                elif name == "V":
                    im_v = mesh
                else:
                    im_p = mesh

            metadata_rows.append({
                "row": row_idx,
                "case_type": case["case_type"],
                "dataset_index": case["idx"],
                "probability": case["prob"],
                "center_y": case["center_y"],
                "center_x": case["center_x"],
            })

        # Shared colorbars.
        cax_blank = fig.add_subplot(gs[2, 0])
        cax_blank.axis("off")

        cax_z = fig.add_subplot(gs[2, 1])
        cax_v = fig.add_subplot(gs[2, 2])
        cax_p = fig.add_subplot(gs[2, 3])

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

        cb_p = fig.colorbar(im_p, cax=cax_p, orientation="horizontal")
        cb_p.set_label("Prediction response", fontsize=6.9, labelpad=1.8)
        cb_p.set_ticks([0.0, 0.5, 1.0])
        cb_p.ax.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.5)
        cb_p.outline.set_linewidth(0.5)

        pdf_path = self.save_dir / "Figure1_Motivation_Nature_v2.pdf"
        png_path = self.save_dir / "Figure1_Motivation_Nature_v2.png"
        csv_path = self.save_dir / "Figure1_Motivation_Nature_v2_metadata.csv"

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

    model = load_model(args.model_path, device)

    dataset = TorNetDatasetV8_Full(
        catalog_path=args.catalog_path,
        root_dir=args.data_root,
        mode="test",
    )

    wrn_case, tor_case = find_cases(args, dataset, model, device)

    plotter = Figure1MotivationNature(save_dir=args.save_dir)
    plotter.plot(wrn_case, tor_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="best_pi_v8_GWP1.pth")
    parser.add_argument("--catalog_path", type=str, default="/path/to/TorNet/catalog.csv")
    parser.add_argument("--data_root", type=str, default="/path/to/TorNet/")
    parser.add_argument("--save_dir", type=str, default="paper_figures_fig1_nature_v2")

    # Optional fixed cases. Use -1 to enable automatic search.
    parser.add_argument("--wrn_idx", type=int, default=-1)
    parser.add_argument("--tor_idx", type=int, default=-1)

    # Automatic search controls.
    parser.add_argument("--scan_limit", type=int, default=5000)
    parser.add_argument("--wrn_local_prob_max", type=float, default=0.40)
    parser.add_argument("--tor_prob_min", type=float, default=0.75)
    parser.add_argument("--tor_vdiff_min", type=float, default=30.0)

    args = parser.parse_args()
    main(args)


