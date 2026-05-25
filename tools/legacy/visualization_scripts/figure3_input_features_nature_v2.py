# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, binary_erosion


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


def get_shear_cmap():
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
    cmap = mcolors.LinearSegmentedColormap.from_list("shear_nature", colors, N=256)
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=0.0, vmax=25.0)
    return cmap, norm


def get_debris_cmap():
    colors = [
        "#FFFFFF",
        "#F2F0F7",
        "#DADAEB",
        "#BCBDDC",
        "#9E9AC8",
        "#756BB1",
        "#54278F",
        "#2D004B",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("debris_nature", colors, N=256)
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=0.0, vmax=0.16)
    return cmap, norm


# ============================================================
# Radar-sector geometry
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


# ============================================================
# Feature construction
# ============================================================

def compute_raw_azimuthal_shear(vel, window_size=5):
    """
    Sliding-window azimuthal velocity contrast.

    Data convention:
        axis 0: azimuth-like dimension
        axis 1: range-like dimension

    The operation is applied along the azimuth dimension.
    """
    local_max = maximum_filter(vel, size=(window_size, 1), mode="nearest")
    local_min = minimum_filter(vel, size=(window_size, 1), mode="nearest")
    raw_shear = local_max - local_min

    raw_shear = np.nan_to_num(raw_shear, nan=0.0, posinf=0.0, neginf=0.0)
    raw_shear = np.clip(raw_shear, 0.0, 50.0)

    return raw_shear


def compute_maas(raw_shear, dbz, background_size=15, valid_dbz_threshold=10.0):
    """
    Multi-scale Anomaly Azimuthal Shear display version.

    This follows the paper logic:
        anomaly = ReLU(raw shear - smoothed background shear)
        then constrained by an eroded reflectivity-valid mask.
    """
    background = uniform_filter(raw_shear, size=background_size, mode="nearest")
    anomaly = np.maximum(raw_shear - background, 0.0)

    valid_mask = dbz > valid_dbz_threshold
    eroded_mask = binary_erosion(valid_mask, structure=np.ones((3, 3)), iterations=1)

    maas = np.where(eroded_mask, anomaly, np.nan)
    maas = np.nan_to_num(maas, nan=np.nan)
    maas = np.clip(maas, 0.0, 25.0)

    return maas


def compute_debris_related_index(dbz, rhohv, valid_dbz_threshold=20.0):
    """
    Debris-related polarimetric index:
        normalized reflectivity * (1 - rhohv)

    This is displayed as a complementary polarimetric cue,
    not as a necessary condition for tornado prediction.
    """
    z_norm = np.clip((dbz + 30.0) / 110.0, 0.0, 1.0)
    rhohv = np.clip(rhohv, 0.0, 1.0)

    debris = z_norm * (1.0 - rhohv)

    valid_mask = dbz > valid_dbz_threshold
    debris = np.where(valid_mask, debris, np.nan)
    debris = np.clip(debris, 0.0, 0.16)

    return debris


def mask_weak_echo(data, dbz, threshold=-10.0):
    """
    Optional display mask to suppress empty-sector noise.
    """
    return np.where(dbz > threshold, data, np.nan)


# ============================================================
# Data loading
# ============================================================

def find_rho_variable(ds):
    for name in ["RHOHV", "RHO", "RhoHV", "CC", "RHO_HV"]:
        if name in ds:
            return name
    return None


def get_nc_path(data_root, row):
    year = pd.to_datetime(row["start_time"]).year if "start_time" in row else 2013

    path1 = os.path.join(data_root, f"tornet_{year}", row["filename"])
    path2 = os.path.join(data_root, row["filename"])

    if os.path.exists(path1):
        return path1
    if os.path.exists(path2):
        return path2

    raise FileNotFoundError(f"Cannot find file: {path1} or {path2}")


def read_case_fields(data_root, row, frame_idx=3, sweep_idx=0):
    nc_path = get_nc_path(data_root, row)

    with xr.open_dataset(nc_path, engine="netcdf4", cache=False) as ds:
        if "DBZ" not in ds or "VEL" not in ds:
            raise KeyError("DBZ or VEL is missing from the NetCDF file.")

        rho_var = find_rho_variable(ds)
        if rho_var is None:
            raise KeyError("No RHOHV-like variable found.")

        dbz_raw = ds["DBZ"].values
        vel_raw = ds["VEL"].values
        rho_raw = ds[rho_var].values

        t_idx = min(frame_idx, dbz_raw.shape[0] - 1)

        # Expected TorNet convention: [time, azimuth, range, sweep]
        dbz = np.nan_to_num(dbz_raw[t_idx, :, :, sweep_idx], nan=-30.0)
        vel = np.nan_to_num(vel_raw[t_idx, :, :, sweep_idx], nan=0.0)
        rho = np.nan_to_num(rho_raw[t_idx, :, :, sweep_idx], nan=1.0)

    return dbz, vel, rho


# ============================================================
# Case selection
# ============================================================

def score_case_for_display(dbz, vel, rho):
    """
    Select a case that can visually show both rotational and debris-related cues.

    This is only for figure display.
    """
    raw_shear = compute_raw_azimuthal_shear(vel, window_size=5)
    maas = compute_maas(raw_shear, dbz)
    debris = compute_debris_related_index(dbz, rho)

    maas_score = np.nanpercentile(maas, 99.5) if np.isfinite(maas).any() else 0.0
    debris_score = np.nanpercentile(debris, 99.5) if np.isfinite(debris).any() else 0.0
    dbz_score = np.nanpercentile(dbz, 99.0) if np.isfinite(dbz).any() else 0.0

    # A balanced score. Debris is useful for display but not required.
    score = 0.55 * maas_score + 35.0 * debris_score + 0.02 * dbz_score
    return float(score)


def select_case(catalog_path, data_root, case_idx=None, mode="test", max_scan=5000, frame_idx=3, sweep_idx=0):
    df = pd.read_csv(catalog_path)
    catalog = df[df["type"] == mode].reset_index(drop=True)

    if case_idx is not None and case_idx >= 0:
        row = catalog.iloc[case_idx]
        dbz, vel, rho = read_case_fields(data_root, row, frame_idx=frame_idx, sweep_idx=sweep_idx)
        return case_idx, row, dbz, vel, rho

    best = None
    best_score = -np.inf

    scan_limit = len(catalog) if max_scan <= 0 else min(len(catalog), max_scan)

    print(f"Scanning {scan_limit} samples to find a clear Fig. 3 display case...")

    for idx in range(scan_limit):
        row = catalog.iloc[idx]

        try:
            dbz, vel, rho = read_case_fields(data_root, row, frame_idx=frame_idx, sweep_idx=sweep_idx)
            score = score_case_for_display(dbz, vel, rho)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best = (idx, row, dbz, vel, rho)

    if best is None:
        raise RuntimeError("Could not find a valid display case. Try increasing --max_scan or specify --case_idx.")

    print(f"Selected case index: {best[0]}, display score: {best_score:.3f}")
    return best


# ============================================================
# Figure rendering
# ============================================================

class InputFeatureConstructionFigure:
    def __init__(self, save_dir="paper_figures_fig3_nature_v2"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_nature_style()

        self.dbz_cmap, self.dbz_norm = get_dbz_cmap()
        self.vel_cmap, self.vel_norm = get_vel_cmap()
        self.shear_cmap, self.shear_norm = get_shear_cmap()
        self.debris_cmap, self.debris_norm = get_debris_cmap()

    def plot(self, dbz, vel, rho, case_idx=None):
        raw_shear = compute_raw_azimuthal_shear(vel, window_size=5)
        maas = compute_maas(raw_shear, dbz, background_size=15, valid_dbz_threshold=10.0)
        debris = compute_debris_related_index(dbz, rho, valid_dbz_threshold=20.0)

        # Display masks
        raw_shear_disp = np.where(dbz > 10.0, raw_shear, np.nan)
        maas_disp = maas
        debris_disp = debris

        fields = [
            {
                "title": "Reflectivity $Z$",
                "data": dbz,
                "cmap": self.dbz_cmap,
                "norm": self.dbz_norm,
                "cbar": "Reflectivity $Z$ (dBZ)",
                "ticks": [0, 20, 40, 60],
            },
            {
                "title": "Radial velocity $V_r$",
                "data": vel,
                "cmap": self.vel_cmap,
                "norm": self.vel_norm,
                "cbar": "Radial velocity $V_r$ (m s$^{-1}$)",
                "ticks": [-30, 0, 30],
            },
            {
                "title": "Raw azimuthal shear",
                "data": raw_shear_disp,
                "cmap": self.shear_cmap,
                "norm": self.shear_norm,
                "cbar": "Shear intensity",
                "ticks": [0, 10, 20],
            },
            {
                "title": "MAAS",
                "data": maas_disp,
                "cmap": self.shear_cmap,
                "norm": self.shear_norm,
                "cbar": "Anomaly shear",
                "ticks": [0, 10, 20],
            },
            {
                "title": "Debris-related index",
                "data": debris_disp,
                "cmap": self.debris_cmap,
                "norm": self.debris_norm,
                "cbar": "Index value",
                "ticks": [0.0, 0.08, 0.16],
            },
        ]

        fig = plt.figure(figsize=(7.35, 2.05))

        gs = GridSpec(
            2,
            5,
            height_ratios=[1.0, 0.075],
            wspace=0.08,
            hspace=0.18,
            left=0.028,
            right=0.992,
            top=0.92,
            bottom=0.18,
        )

        axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
        caxes = [fig.add_subplot(gs[1, i]) for i in range(5)]

        panel_letters = list("abcde")
        meshes = []

        for i, item in enumerate(fields):
            mesh = plot_polar_channel(
                axes[i],
                item["data"],
                item["cmap"],
                item["norm"],
                column_title=item["title"],
            )
            add_panel_label(axes[i], f"({panel_letters[i]})")
            meshes.append(mesh)

        for i, item in enumerate(fields):
            cb = fig.colorbar(meshes[i], cax=caxes[i], orientation="horizontal")
            cb.set_label(item["cbar"], fontsize=6.7, labelpad=1.6)
            cb.set_ticks(item["ticks"])
            cb.ax.tick_params(labelsize=6.2, direction="in", length=2.2, width=0.5)
            cb.outline.set_linewidth(0.5)


        pdf_path = self.save_dir / "Figure3_Input_Features_Nature_v2.pdf"
        png_path = self.save_dir / "Figure3_Input_Features_Nature_v2.png"
        csv_path = self.save_dir / "Figure3_Input_Features_Nature_v2_metadata.csv"

        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        metadata = pd.DataFrame([{
            "case_idx": case_idx,
            "raw_shear_p99": float(np.nanpercentile(raw_shear_disp, 99)) if np.isfinite(raw_shear_disp).any() else np.nan,
            "maas_p99": float(np.nanpercentile(maas_disp, 99)) if np.isfinite(maas_disp).any() else np.nan,
            "debris_p99": float(np.nanpercentile(debris_disp, 99)) if np.isfinite(debris_disp).any() else np.nan,
        }])
        metadata.to_csv(csv_path, index=False)

        print(f"Saved: {pdf_path}")
        print(f"Saved: {png_path}")
        print(f"Saved: {csv_path}")


# ============================================================
# Main
# ============================================================

def main(args):
    case_idx, row, dbz, vel, rho = select_case(
        catalog_path=args.catalog_path,
        data_root=args.data_root,
        case_idx=args.case_idx,
        mode=args.mode,
        max_scan=args.max_scan,
        frame_idx=args.frame_idx,
        sweep_idx=args.sweep_idx,
    )

    plotter = InputFeatureConstructionFigure(save_dir=args.save_dir)
    plotter.plot(dbz, vel, rho, case_idx=case_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--catalog_path", type=str, default="/path/to/TorNet/catalog.csv")
    parser.add_argument("--data_root", type=str, default="/path/to/TorNet/")
    parser.add_argument("--save_dir", type=str, default="paper_figures_fig3_nature_v2")

    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--case_idx", type=int, default=-1)
    parser.add_argument("--max_scan", type=int, default=5000)

    parser.add_argument("--frame_idx", type=int, default=3)
    parser.add_argument("--sweep_idx", type=int, default=0)

    args = parser.parse_args()
    main(args)


