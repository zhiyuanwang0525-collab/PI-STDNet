# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import maximum_filter, minimum_filter

# Prevent train_compare from consuming command-line arguments.
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import train_compare
sys.argv = _original_argv


# ============================================================
# Style
# ============================================================

def set_nature_style():
    """
    Clean Nature-like figure style:
    - sans-serif font
    - light axes
    - compact labels
    - embedded fonts in PDF
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 0.7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Input helpers
# ============================================================

def parse_case(case_string, category, case_title):
    """
    Format:
        "2075:0.3535:0.7422"

    Meaning:
        dataset_index:se_probability:pi_stdnet_probability
    """
    parts = case_string.strip().split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid case specification: {case_string}. "
            "Expected format: idx:se_prob:pi_prob"
        )

    return {
        "idx": int(parts[0]),
        "category": category,
        "title": case_title,
        "se": float(parts[1]),
        "pi": float(parts[2]),
    }


def parse_center(center_string):
    """
    Optional manual center format:
        "row,col"

    Example:
        --tor_center "62,145"

    If empty, "auto", or "none", the strongest local velocity
    contrast is used automatically.
    """
    if center_string is None:
        return None

    s = center_string.strip().lower()
    if s in ["", "auto", "none"]:
        return None

    parts = s.replace(";", ",").split(",")
    if len(parts) != 2:
        raise ValueError("Manual center must use format row,col, e.g., 62,145")

    return int(parts[0]), int(parts[1])


# ============================================================
# Radar geometry
# ============================================================

def get_polar_grid(H=120, W=240, r_max=60.0, az_min=-30.0, az_max=30.0):
    """
    Build TorNet polar-sector grid.

    Display convention:
        azimuth: -30 to 30 degrees
        range:   0 to 60 km
    """
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
        alpha=0.32,
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
        fontsize=6.2,
        fontweight="bold",
        zorder=21,
    )


# ============================================================
# Focus region
# ============================================================

def find_shear_center(v_data, z_data=None, window=9, valid_dbz_threshold=12.0):
    """
    Find a compact velocity-contrast center.

    This is only a fallback. For final paper figures, manual centers
    are often better when the automatic maximum falls on missing-data
    edges or boundary noise.
    """
    local_max = maximum_filter(v_data, size=window, mode="nearest")
    local_min = minimum_filter(v_data, size=window, mode="nearest")
    shear = local_max - local_min

    if z_data is not None:
        shear = np.where(z_data > valid_dbz_threshold, shear, 0.0)

    margin_y = 6
    margin_x = 12
    shear[:margin_y, :] = 0.0
    shear[-margin_y:, :] = 0.0
    shear[:, :margin_x] = 0.0
    shear[:, -margin_x:] = 0.0

    cy, cx = np.unravel_index(np.nanargmax(shear), shear.shape)

    return int(cy), int(cx), shear


def crop_around(data, cy, cx, half_y=18, half_x=24):
    y0 = max(0, cy - half_y)
    y1 = min(data.shape[0], cy + half_y)
    x0 = max(0, cx - half_x)
    x1 = min(data.shape[1], cx + half_x)

    return data[y0:y1, x0:x1], (y0, y1, x0, x1)


def draw_focus_box_on_full(
    ax,
    cy,
    cx,
    az_centers,
    r_centers,
    half_y=18,
    half_x=24,
):
    """
    Draw a high-contrast focus box around selected radar cells.
    The black outer stroke + white inner stroke remains visible on both
    bright reflectivity and velocity backgrounds.
    """
    corners = [
        (cy - half_y, cx - half_x),
        (cy - half_y, cx + half_x),
        (cy + half_y, cx + half_x),
        (cy + half_y, cx - half_x),
    ]

    xy = [cell_to_xy(r, c, az_centers, r_centers) for r, c in corners]

    outer = patches.Polygon(
        xy,
        closed=True,
        facecolor="none",
        edgecolor="black",
        lw=2.2,
        zorder=15,
    )
    inner = patches.Polygon(
        xy,
        closed=True,
        facecolor="none",
        edgecolor="white",
        lw=1.2,
        zorder=16,
    )

    ax.add_patch(outer)
    ax.add_patch(inner)

    x, y = cell_to_xy(cy, cx, az_centers, r_centers)

    ax.plot(x, y, marker="+", color="black", ms=7.5, mew=1.9, zorder=17)
    ax.plot(x, y, marker="+", color="white", ms=6.0, mew=1.2, zorder=18)


# ============================================================
# Plotting primitives
# ============================================================

def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        zorder=30,
    )


def plot_polar_field(ax, data, X, Y, cmap, norm, column_title=None):
    mesh = ax.pcolormesh(
        X,
        Y,
        np.ma.masked_invalid(data),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    if column_title is not None:
        ax.set_title(column_title, fontsize=8.2, fontweight="normal", pad=3.0)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(X.min() - 1.0, X.max() + 1.0)
    ax.set_ylim(Y.min() - 1.0, Y.max() + 1.0)
    ax.set_facecolor("white")

    draw_sector_boundary(ax)
    add_scale_bar(ax)

    return mesh


def plot_zoom_velocity(
    ax,
    v_data,
    cy,
    cx,
    cmap,
    norm,
    column_title=None,
    half_y=18,
    half_x=24,
):
    crop, (y0, y1, x0, x1) = crop_around(
        v_data,
        cy,
        cx,
        half_y=half_y,
        half_x=half_x,
    )

    im = ax.imshow(
        crop,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
    )

    if column_title is not None:
        ax.set_title(column_title, fontsize=8.2, fontweight="normal", pad=3.0)

    ax.set_xticks([])
    ax.set_yticks([])

    cy_crop = cy - y0
    cx_crop = cx - x0

    ax.plot(cx_crop, cy_crop, marker="+", color="black", ms=8.5, mew=1.8, zorder=8)
    ax.plot(cx_crop, cy_crop, marker="+", color="white", ms=6.8, mew=1.1, zorder=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    return im


def plot_probability_shift(ax, case, threshold=0.60, column_title=None):
    se_prob = case["se"]
    pi_prob = case["pi"]
    category = case["category"]

    if category == "TOR":
        pi_color = "#B2182B"
    else:
        pi_color = "#2166AC"

    se_color = "0.66"

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, 1.55)

    ax.barh(
        [1],
        [se_prob],
        height=0.34,
        color=se_color,
        edgecolor="none",
        zorder=2,
    )

    ax.barh(
        [0],
        [pi_prob],
        height=0.34,
        color=pi_color,
        edgecolor="none",
        zorder=2,
    )

    ax.axvline(threshold, color="0.15", ls="--", lw=0.8, zorder=1)

    ax.text(
        threshold + 0.018,
        1.42,
        "0.6",
        ha="left",
        va="center",
        fontsize=6.8,
        color="0.15",
    )

    # Put model names on the left and values on the far right.
    # This avoids overlap when PI-STDNet probability is low, e.g., 0.25.
    label_box = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.70,
        pad=0.8,
    )

    ax.text(
        0.035,
        1,
        "SE-Net",
        ha="left",
        va="center",
        fontsize=6.8,
        color="0.20",
        fontweight="bold",
        bbox=label_box,
        zorder=5,
    )

    ax.text(
        0.035,
        0,
        "PI-STDNet",
        ha="left",
        va="center",
        fontsize=6.8,
        color="0.20",
        fontweight="bold",
        bbox=label_box,
        zorder=5,
    )

    value_box = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
        pad=0.6,
    )

    ax.text(
        0.965,
        1,
        f"{se_prob:.2f}",
        ha="right",
        va="center",
        fontsize=7.2,
        color="0.10",
        bbox=value_box,
        zorder=6,
    )

    ax.text(
        0.965,
        0,
        f"{pi_prob:.2f}",
        ha="right",
        va="center",
        fontsize=7.2,
        color="0.10",
        bbox=value_box,
        zorder=6,
    )

    if column_title is not None:
        ax.set_title(column_title, fontsize=8.2, fontweight="normal", pad=3.0)

    ax.set_yticks([])
    ax.set_xticks([0.0, 0.6, 1.0])
    ax.set_xlabel("Probability", fontsize=7.0, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=6.8, direction="in", length=2.5)
    ax.grid(axis="x", ls="-", lw=0.35, alpha=0.25, zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def plot_case_label(ax, case):
    """
    Dedicated case-label card in the left column.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if case["category"] == "TOR":
        color = "#B2182B"
        face = "#F8E8E8"
        title = "Captured\nmissed TOR"
        transition = f"{case['se']:.2f} → {case['pi']:.2f}"
        condition = "crosses 0.6"
    else:
        color = "#2166AC"
        face = "#E8F0FA"
        title = "Suppressed\nWRN false alarm"
        transition = f"{case['se']:.2f} → {case['pi']:.2f}"
        condition = "falls below 0.6"

    card = patches.FancyBboxPatch(
        (0.04, 0.10),
        0.92,
        0.80,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face,
        edgecolor="none",
        alpha=1.0,
    )
    ax.add_patch(card)

    ax.text(
        0.50,
        0.63,
        title,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=color,
        linespacing=1.15,
    )

    ax.text(
        0.50,
        0.34,
        transition,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="0.15",
    )

    ax.text(
        0.50,
        0.20,
        condition,
        ha="center",
        va="center",
        fontsize=6.3,
        color="0.35",
    )


# ============================================================
# Data extraction
# ============================================================

def extract_z_v(image_tensor, sweep_idx=0, frame_idx=0):
    """
    Extract DBZ and VEL from TorNetDatasetAblation tensors.

    Channel order:
        [sweep][frame][channel]

    channel 0 = DBZ
    channel 1 = VEL
    """
    arr = image_tensor.numpy()

    cpf = getattr(train_compare, "CONFIG", {}).get("CHANNELS_PER_FRAME", 11)
    n_frames = getattr(train_compare, "CONFIG", {}).get("N_FRAMES", 4)

    z_idx = sweep_idx * (n_frames * cpf) + frame_idx * cpf + 0
    v_idx = sweep_idx * (n_frames * cpf) + frame_idx * cpf + 1

    z = arr[z_idx] * 110.0 - 30.0
    v = arr[v_idx] * 100.0 - 50.0

    return z, v


# ============================================================
# Main figure class
# ============================================================

class DecisionCaseFigureNatureV5:
    def __init__(self, save_dir="paper_figures_split_nature_v5"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_nature_style()

        self.cmap_z = plt.cm.turbo.copy()
        self.cmap_z.set_bad("white")
        self.norm_z = mcolors.Normalize(vmin=-10, vmax=65)

        self.cmap_v = plt.cm.seismic.copy()
        self.cmap_v.set_bad("white")
        self.norm_v = mcolors.TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)

    def plot(self, dataset, cases, centers=None, sweep_idx=0, frame_idx=0):
        X, Y, az_centers, r_centers = get_polar_grid()

        fig = plt.figure(figsize=(7.35, 3.65))

        gs = GridSpec(
            3,
            5,
            height_ratios=[1.0, 1.0, 0.075],
            width_ratios=[0.72, 1.15, 1.15, 1.28, 1.00],
            hspace=0.16,
            wspace=0.10,
            left=0.030,
            right=0.990,
            top=0.925,
            bottom=0.135,
        )

        im_z = None
        im_v = None
        selected_rows = []

        column_titles = [
            None,
            "Reflectivity $Z$",
            "Radial velocity $V_r$",
            "Zoomed $V_r$",
            "Probability shift",
        ]

        panel_letters = list("abcdefgh")

        for row_idx, case in enumerate(cases):
            image_tensor, _, _ = dataset[case["idx"]]

            z_data, v_data = extract_z_v(
                image_tensor,
                sweep_idx=sweep_idx,
                frame_idx=frame_idx,
            )

            manual_center = centers[row_idx] if centers is not None else None

            if manual_center is None:
                cy, cx, _ = find_shear_center(v_data, z_data=z_data, window=9)
                center_mode = "auto"
            else:
                cy, cx = manual_center
                center_mode = "manual"

            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_z = fig.add_subplot(gs[row_idx, 1])
            ax_v = fig.add_subplot(gs[row_idx, 2])
            ax_zoom = fig.add_subplot(gs[row_idx, 3])
            ax_prob = fig.add_subplot(gs[row_idx, 4])

            plot_case_label(ax_label, case)

            title_z = column_titles[1] if row_idx == 0 else None
            title_v = column_titles[2] if row_idx == 0 else None
            title_zoom = column_titles[3] if row_idx == 0 else None
            title_prob = column_titles[4] if row_idx == 0 else None

            im_z = plot_polar_field(
                ax_z,
                z_data,
                X,
                Y,
                self.cmap_z,
                self.norm_z,
                column_title=title_z,
            )
            add_panel_label(ax_z, f"({panel_letters[row_idx * 4]})")

            im_v = plot_polar_field(
                ax_v,
                v_data,
                X,
                Y,
                self.cmap_v,
                self.norm_v,
                column_title=title_v,
            )
            add_panel_label(ax_v, f"({panel_letters[row_idx * 4 + 1]})")

            draw_focus_box_on_full(ax_z, cy, cx, az_centers, r_centers)
            draw_focus_box_on_full(ax_v, cy, cx, az_centers, r_centers)

            plot_zoom_velocity(
                ax_zoom,
                v_data,
                cy,
                cx,
                self.cmap_v,
                self.norm_v,
                column_title=title_zoom,
                half_y=18,
                half_x=24,
            )
            add_panel_label(ax_zoom, f"({panel_letters[row_idx * 4 + 2]})")

            plot_probability_shift(
                ax_prob,
                case,
                threshold=0.60,
                column_title=title_prob,
            )
            add_panel_label(ax_prob, f"({panel_letters[row_idx * 4 + 3]})")

            selected_rows.append({
                "dataset_index": case["idx"],
                "category": case["category"],
                "case_title": case["title"],
                "se_probability": case["se"],
                "pi_stdnet_probability": case["pi"],
                "center_row": cy,
                "center_col": cx,
                "center_mode": center_mode,
                "sweep_idx": sweep_idx,
                "frame_idx": frame_idx,
            })

        cax_z = fig.add_subplot(gs[2, 1])
        cax_v = fig.add_subplot(gs[2, 2:4])

        cb_z = fig.colorbar(im_z, cax=cax_z, orientation="horizontal")
        cb_z.set_label("Reflectivity $Z$ (dBZ)", fontsize=7.0, labelpad=2.0)
        cb_z.ax.tick_params(labelsize=6.5, direction="in", length=2.4, width=0.55)
        cb_z.outline.set_linewidth(0.55)

        cb_v = fig.colorbar(im_v, cax=cax_v, orientation="horizontal")
        cb_v.set_label("Radial velocity $V_r$ (m s$^{-1}$)", fontsize=7.0, labelpad=2.0)
        cb_v.ax.tick_params(labelsize=6.5, direction="in", length=2.4, width=0.55)
        cb_v.outline.set_linewidth(0.55)

        blank_left = fig.add_subplot(gs[2, 0])
        blank_right = fig.add_subplot(gs[2, 4])
        blank_left.axis("off")
        blank_right.axis("off")

        pdf = self.save_dir / "Figure10_Decision_Cases_Nature_v5.pdf"
        png = self.save_dir / "Figure10_Decision_Cases_Nature_v5.png"
        csv = self.save_dir / "Figure10_Decision_Cases_Nature_v5_selected_cases.csv"

        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(selected_rows).to_csv(csv, index=False)

        print(f"Saved: {pdf}")
        print(f"Saved: {png}")
        print(f"Saved: {csv}")


# ============================================================
# Main
# ============================================================

def main(args):
    tor_case = parse_case(
        args.tor_case,
        category="TOR",
        case_title="Captured missed tornado",
    )

    wrn_case = parse_case(
        args.wrn_case,
        category="WRN",
        case_title="Suppressed false alarm",
    )

    centers = [
        parse_center(args.tor_center),
        parse_center(args.wrn_center),
    ]

    catalog_path = args.catalog_path or train_compare.CONFIG["CATALOG_PATH"]
    data_root = args.data_root or train_compare.CONFIG["DATA_ROOT"]

    dataset = train_compare.TorNetDatasetAblation(
        catalog_path,
        data_root,
        mode="test",
    )

    plotter = DecisionCaseFigureNatureV5(args.save_dir)
    plotter.plot(
        dataset,
        [tor_case, wrn_case],
        centers=centers,
        sweep_idx=args.sweep_idx,
        frame_idx=args.frame_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--catalog_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="paper_figures_split_nature_v5")

    parser.add_argument("--sweep_idx", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=0)

    parser.add_argument(
        "--tor_case",
        type=str,
        default="2075:0.3535:0.7422",
        help="Captured TOR case as idx:se_prob:pi_prob",
    )

    parser.add_argument(
        "--wrn_case",
        type=str,
        default="27306:0.6978:0.2456",
        help="Suppressed WRN case as idx:se_prob:pi_prob",
    )

    parser.add_argument(
        "--tor_center",
        type=str,
        default="auto",
        help="Optional manual center for TOR case, format row,col",
    )

    parser.add_argument(
        "--wrn_center",
        type=str,
        default="auto",
        help="Optional manual center for WRN case, format row,col",
    )

    main(parser.parse_args())
