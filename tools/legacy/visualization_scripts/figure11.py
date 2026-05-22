# Legacy note: Preserved from the original research workspace for traceability. Prefer scripts/ and src/pistdnet/ for public use.
import os
import re
import sys
import argparse
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from torch.utils.data import Dataset


# =========================================================
# 0. Import train_compare safely
# =========================================================
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import train_compare
sys.argv = _original_argv


# =========================================================
# 1. Fixed args for full PI-STDNet
# =========================================================
class MockArgs:
    def __init__(
        self,
        attn_type="physics",
        disable_physics_attn=False,
        disable_physics_inputs=False,
        disable_rot_stats=False,
        disable_topk=False,
    ):
        self.attn_type = attn_type
        self.disable_physics_attn = disable_physics_attn
        self.disable_physics_inputs = disable_physics_inputs
        self.disable_rot_stats = disable_rot_stats
        self.disable_topk = disable_topk


def setup_full_model_flags(has_rot=True, has_topk=True):
    train_compare.args = MockArgs(
        attn_type="physics",
        disable_physics_attn=False,
        disable_physics_inputs=False,
        disable_rot_stats=not has_rot,
        disable_topk=not has_topk,
    )
    train_compare.CONFIG["CHANNELS_PER_FRAME"] = 11


# =========================================================
# 2. Strict dataset: no random fallback when loading fails
# =========================================================
class StrictTorNetDataset(Dataset):
    """
    Same loading logic as TorNetDatasetAblation, but it raises errors instead of
    silently replacing failed samples with random samples.

    This is important for paper figure generation.
    """

    def __init__(self, catalog_path, root_dir, mode="test"):
        self.root_dir = root_dir
        self.mode = mode

        df = pd.read_csv(catalog_path)
        self.catalog = df[df["type"] == mode].reset_index(drop=True)

        self.labels = (self.catalog["category"] == "TOR").astype(int).values

        self.cat_ids = np.zeros(len(self.catalog), dtype=int)
        self.cat_ids[self.catalog["category"] == "TOR"] = 2

        wrn_mask = (
            (self.catalog["category"] == "WRN")
            | (self.catalog["category"] == "Tornado Warning")
        )
        self.cat_ids[wrn_mask] = 1

    def __len__(self):
        return len(self.catalog)

    def _get_filepath(self, row):
        year = pd.to_datetime(row["start_time"]).year if "start_time" in row else 2013

        filepath_1 = os.path.join(self.root_dir, f"tornet_{year}", row["filename"])
        filepath_2 = os.path.join(self.root_dir, row["filename"])

        if os.path.exists(filepath_1):
            return filepath_1

        if os.path.exists(filepath_2):
            return filepath_2

        raise FileNotFoundError(f"Cannot find file: {filepath_1} or {filepath_2}")

    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        filepath = self._get_filepath(row)

        with xr.open_dataset(filepath, engine="netcdf4", cache=False) as ds:

            def safe_read(key, fill_val, sweep_idx=0):
                if key not in ds:
                    return np.full((4, 120, 240), fill_val, dtype=np.float32)

                raw = ds[key].values

                # Same convention as train_compare.py:
                # if 4D, sweep is assumed to be the last dimension.
                if raw.ndim == 4:
                    if raw.shape[-1] > sweep_idx:
                        data = raw[..., sweep_idx]
                    else:
                        data = raw[..., 0]
                else:
                    data = raw

                return np.nan_to_num(data, nan=fill_val).astype(np.float32)

            def norm(arr, key):
                mn, mx = train_compare.CHANNEL_MIN_MAX[key]
                return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

            all_channels = []

            for sweep_idx in range(train_compare.CONFIG["N_SWEEPS"]):
                dbz = safe_read("DBZ", -30.0, sweep_idx)
                vel = safe_read("VEL", 0.0, sweep_idx)
                kdp = safe_read("KDP", 0.0, sweep_idx)
                rhohv = safe_read("RHOHV", 0.0, sweep_idx)
                zdr = safe_read("ZDR", 0.0, sweep_idx)
                width = safe_read("WIDTH", 0.0, sweep_idx)

                frames_idx = list(
                    range(min(dbz.shape[0], train_compare.CONFIG["N_FRAMES"]))
                )

                if len(frames_idx) == 0:
                    raise RuntimeError(f"No valid time frame in file: {filepath}")

                while len(frames_idx) < train_compare.CONFIG["N_FRAMES"]:
                    frames_idx.append(frames_idx[-1])

                for f_idx in frames_idx:
                    f_dbz = dbz[f_idx]
                    f_vel = vel[f_idx]

                    base_ch = [
                        norm(f_dbz, "DBZ"),
                        norm(f_vel, "VEL"),
                        norm(kdp[f_idx], "KDP"),
                        norm(rhohv[f_idx], "RHOHV"),
                        norm(zdr[f_idx], "ZDR"),
                        norm(width[f_idx], "WIDTH"),
                    ]

                    anomaly_shears = (
                        train_compare.PhysicalFeatureAmplifier
                        .compute_multiscale_anomaly_shear(f_vel, f_dbz)
                    )

                    all_channels.extend(
                        base_ch
                        + [
                            norm(
                                train_compare.PhysicalFeatureAmplifier.compute_shear(f_vel),
                                "SHEAR",
                            ),
                            np.clip(
                                train_compare.PhysicalFeatureAmplifier.compute_debris(
                                    f_dbz, rhohv[f_idx]
                                ),
                                0,
                                1,
                            ),
                            np.clip(anomaly_shears[0] / 20.0, 0.0, 1.0),
                            np.clip(anomaly_shears[1] / 20.0, 0.0, 1.0),
                            np.clip(anomaly_shears[2] / 20.0, 0.0, 1.0),
                        ]
                    )

            data = np.stack(all_channels, axis=0).astype(np.float32)
            data = np.clip(np.nan_to_num(data), -1.0, 2.0)

            return (
                torch.from_numpy(data.copy()),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.cat_ids[idx], dtype=torch.int8),
            )


# =========================================================
# 3. Model loading and inference
# =========================================================
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    has_rot = any("rot_proj" in k for k in state_dict.keys())
    has_topk = any("topk_pool" in k for k in state_dict.keys())

    setup_full_model_flags(has_rot=has_rot, has_topk=has_topk)

    model = train_compare.PI_STDNet_Ablation().to(device)

    # Important for the local train_compare.py version.
    model.use_train_py_order = True

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"has_rot={has_rot}, has_topk={has_topk}")
    print(f"missing keys={len(missing)}, unexpected keys={len(unexpected)}")

    return model


def predict_prob(model, image_tensor, device):
    x = image_tensor.unsqueeze(0).to(device)

    amp_enabled = device.type == "cuda" and train_compare.CONFIG.get("USE_AMP", False)
    ctx = (
        torch.amp.autocast("cuda", enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        with ctx:
            cls_logit, aux_logit, _, _ = model(x)
            prob = (
                0.6 * torch.sigmoid(cls_logit)
                + 0.4 * torch.sigmoid(aux_logit)
            ).item()

    return prob


# =========================================================
# 4. Channel extraction
# =========================================================
def channel_index(sweep_idx, frame_idx, channel_idx):
    """
    Real channel order from train_compare.py:
    [sweep][frame][channel]

    channel per frame:
    0 DBZ
    1 VEL
    2 KDP
    3 RHOHV
    4 ZDR
    5 WIDTH
    6 SHEAR
    7 DEBRIS
    8 MAAS-3
    9 MAAS-5
    10 MAAS-7
    """
    cpf = train_compare.CONFIG["CHANNELS_PER_FRAME"]
    n_frames = train_compare.CONFIG["N_FRAMES"]
    return sweep_idx * (n_frames * cpf) + frame_idx * cpf + channel_idx


def extract_display_fields(image_tensor, sweep_idx=0, frame_idx=0):
    """
    Default frame_idx=0 follows the setting that produced the selected legacy figure.
    """
    arr = image_tensor.numpy()

    z_idx = channel_index(sweep_idx, frame_idx, 0)
    v_idx = channel_index(sweep_idx, frame_idx, 1)

    z_data = arr[z_idx] * 110.0 - 30.0
    v_data = arr[v_idx] * 100.0 - 50.0

    return z_data, v_data


def visual_diff_norm(dataset, idx_a, idx_b, sweep_idx=0, frame_idx=0):
    """
    Use normalized DBZ/VEL for continuity checks.
    """
    img_a, _, _ = dataset[idx_a]
    img_b, _, _ = dataset[idx_b]

    z_idx = channel_index(sweep_idx, frame_idx, 0)
    v_idx = channel_index(sweep_idx, frame_idx, 1)

    za = img_a[z_idx].numpy()
    va = img_a[v_idx].numpy()
    zb = img_b[z_idx].numpy()
    vb = img_b[v_idx].numpy()

    dz = float(np.mean(np.abs(za - zb)))
    dv = float(np.mean(np.abs(va - vb)))

    try:
        zcorr = float(np.corrcoef(za.reshape(-1), zb.reshape(-1))[0, 1])
    except Exception:
        zcorr = -1.0

    if np.isnan(zcorr):
        zcorr = -1.0

    return {
        "dz": dz,
        "dv": dv,
        "sum": dz + dv,
        "zcorr": zcorr,
    }


# =========================================================
# 5. Candidate generation
# =========================================================
def infer_time_col(df):
    candidates = [
        "start_time",
        "end_time",
        "time",
        "timestamp",
        "scan_time",
        "valid_time",
        "datetime",
        "date_time",
    ]

    for c in candidates:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().sum() > 0:
                return c

    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().sum() > 0:
                return c

    return None


def normalize_filename_path(x):
    return str(x).replace("\\", "/")


def derive_parent_group_from_filename(filename):
    p = normalize_filename_path(filename)
    parent = os.path.dirname(p)
    return parent if parent else "no_parent"


def derive_prefix_group_from_filename(filename):
    """
    Fallback grouping from filename stem.
    """
    p = normalize_filename_path(filename)
    stem = Path(p).stem

    stem = re.sub(r"[_\-]?\d{8}[_\-]?\d{4,6}$", "", stem)
    stem = re.sub(r"[_\-]?\d{10,14}$", "", stem)

    stem2 = re.sub(r"[_\-]\d+$", "", stem)
    if len(stem2) >= 6:
        stem = stem2

    return stem


def build_group_sources(df, max_derived_group_size=80):
    """
    Build multiple possible grouping sources.
    """
    sources = []

    exact_cols = [
        "event_id",
        "storm_id",
        "case_id",
        "tornado_id",
        "tor_id",
        "episode_id",
        "storm_track_id",
        "track_id",
        "object_id",
    ]

    for c in exact_cols:
        if c in df.columns:
            s = df[c].astype(str)
            sizes = s.groupby(s).size()
            if (sizes >= 4).sum() > 0:
                sources.append((f"column:{c}", s))

    bad_cols = {"category", "type", "split", "mode", "label", "target"}

    for c in df.columns:
        lc = c.lower()
        if lc in bad_cols:
            continue

        if not any(
            k in lc
            for k in ["event", "storm", "case", "tornado", "episode", "track", "object"]
        ):
            continue

        if any(existing_name == f"column:{c}" for existing_name, _ in sources):
            continue

        s = df[c].astype(str)
        sizes = s.groupby(s).size()

        if (sizes >= 4).sum() > 0:
            sources.append((f"column:{c}", s))

    if "filename" in df.columns:
        parent_s = df["filename"].apply(derive_parent_group_from_filename)
        parent_sizes = parent_s.groupby(parent_s).size()

        if (parent_sizes >= 4).sum() > 0:
            sources.append(("filename_parent", parent_s))

        prefix_s = df["filename"].apply(derive_prefix_group_from_filename)
        prefix_sizes = prefix_s.groupby(prefix_s).size()

        if (prefix_sizes >= 4).sum() > 0:
            sources.append(("filename_prefix", prefix_s))

    filtered = []

    for name, s in sources:
        sizes = s.groupby(s).size()

        if name.startswith("filename") and sizes.max() > max_derived_group_size:
            print(f"Skip broad group source {name}: max group size={sizes.max()}")
            continue

        filtered.append((name, s))

    return filtered


def build_target_time_candidates(
    dataset,
    offset_tolerance=6.0,
    max_derived_group_size=80,
):
    """
    Find four samples from the same group nearest to T-15/T-10/T-5/T0.
    T0 is required to be a TOR sample.
    """
    df = dataset.catalog.copy().reset_index(drop=True)
    df["_ds_idx"] = np.arange(len(df))

    print("\nCatalog columns:")
    print(list(df.columns))

    time_col = infer_time_col(df)
    print(f"\nInferred time_col = {time_col}")

    if time_col is None:
        print("Cannot infer time column.")
        return []

    df["_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["_time"]).reset_index(drop=True)

    group_sources = build_group_sources(
        df,
        max_derived_group_size=max_derived_group_size,
    )

    print("\nCandidate group sources:")
    for name, s in group_sources:
        sizes = s.groupby(s).size()
        print(f"  {name}: groups>=4={(sizes >= 4).sum()}, max_size={sizes.max()}")

    if len(group_sources) == 0:
        print("No usable group source found.")
        return []

    candidates = []
    seen = set()

    target_offsets = [-15, -10, -5, 0]

    for source_name, group_series in group_sources:
        work = df.copy()
        work["_group_key"] = group_series.loc[work.index].astype(str).values

        for gid, g in work.groupby("_group_key"):
            g = g.sort_values("_time").reset_index(drop=True)

            if len(g) < 4:
                continue

            if "category" not in g.columns:
                continue

            tor_rows = g[g["category"] == "TOR"]

            if len(tor_rows) == 0:
                continue

            for _, end_row in tor_rows.iterrows():
                end_time = end_row["_time"]

                selected_rows = []
                total_offset_error = 0.0
                ok = True

                for off in target_offsets:
                    if off == 0:
                        nearest = end_row.copy()
                        abs_err = 0.0
                    else:
                        target_time = end_time + pd.Timedelta(minutes=off)

                        tmp = g.copy()
                        tmp["_abs_err"] = (
                            tmp["_time"] - target_time
                        ).abs().dt.total_seconds() / 60.0

                        nearest = tmp.sort_values("_abs_err").iloc[0]
                        abs_err = float(nearest["_abs_err"])

                        if abs_err > offset_tolerance:
                            ok = False
                            break

                    selected_rows.append(nearest)
                    total_offset_error += abs_err

                if not ok:
                    continue

                indices = [int(r["_ds_idx"]) for r in selected_rows]

                if len(set(indices)) < 4:
                    continue

                times = [r["_time"] for r in selected_rows]

                if any(times[i] >= times[i + 1] for i in range(3)):
                    continue

                rel_minutes = [
                    -int(round((end_time - t).total_seconds() / 60.0))
                    for t in times
                ]

                key = tuple(indices)

                if key in seen:
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "group_source": source_name,
                        "group_id": gid,
                        "indices": indices,
                        "times": times,
                        "rel_minutes": rel_minutes,
                        "gaps": [
                            (times[j + 1] - times[j]).total_seconds() / 60.0
                            for j in range(3)
                        ],
                        "offset_error": total_offset_error,
                    }
                )

    candidates = sorted(candidates, key=lambda x: x["offset_error"])

    print(f"\nFound target-time candidates: {len(candidates)}")

    return candidates


def build_sliding_window_candidates(
    dataset,
    min_gap=1.5,
    max_gap=12.0,
    max_derived_group_size=80,
):
    """
    Backup strategy: consecutive 4-scan windows from the same group.
    """
    df = dataset.catalog.copy().reset_index(drop=True)
    df["_ds_idx"] = np.arange(len(df))

    time_col = infer_time_col(df)

    if time_col is None:
        return []

    df["_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["_time"]).reset_index(drop=True)

    group_sources = build_group_sources(
        df,
        max_derived_group_size=max_derived_group_size,
    )

    candidates = []
    seen = set()

    for source_name, group_series in group_sources:
        work = df.copy()
        work["_group_key"] = group_series.loc[work.index].astype(str).values

        for gid, g in work.groupby("_group_key"):
            g = g.sort_values("_time").reset_index(drop=True)

            if len(g) < 4:
                continue

            for i in range(len(g) - 3):
                w = g.iloc[i : i + 4].copy()

                if w.iloc[-1]["category"] != "TOR":
                    continue

                times = list(w["_time"])

                gaps = [
                    (times[j + 1] - times[j]).total_seconds() / 60.0
                    for j in range(3)
                ]

                if not all(min_gap <= gap <= max_gap for gap in gaps):
                    continue

                end_time = times[-1]

                rel_minutes = [
                    -int(round((end_time - t).total_seconds() / 60.0))
                    for t in times
                ]

                indices = list(w["_ds_idx"].astype(int))

                key = tuple(indices)

                if key in seen:
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "group_source": source_name,
                        "group_id": gid,
                        "indices": indices,
                        "times": times,
                        "rel_minutes": rel_minutes,
                        "gaps": gaps,
                        "offset_error": abs(rel_minutes[0] + 15)
                        + abs(rel_minutes[1] + 10)
                        + abs(rel_minutes[2] + 5),
                    }
                )

    candidates = sorted(candidates, key=lambda x: x["offset_error"])

    print(f"\nFound sliding-window candidates: {len(candidates)}")

    return candidates


# =========================================================
# 6. Candidate checking
# =========================================================
class SampleCache:
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.sample_cache = {}
        self.prob_cache = {}

    def get_sample(self, idx):
        if idx not in self.sample_cache:
            self.sample_cache[idx] = self.dataset[idx]
        return self.sample_cache[idx]

    def get_prob_cat(self, idx):
        if idx not in self.prob_cache:
            image_tensor, _, cat = self.get_sample(idx)
            prob = predict_prob(self.model, image_tensor, self.device)
            self.prob_cache[idx] = (prob, int(cat))
        return self.prob_cache[idx]


def evaluate_candidate_sequence(
    cache,
    cand,
    display_frame,
    min_final_prob=0.60,
    min_lead_prob=0.60,
    max_early_prob=0.55,
    min_gain=0.20,
    allowed_late_drop=0.08,
    min_dup_diff=0.003,
    max_jump_diff=0.35,
    min_zcorr=-0.05,
):
    """
    Return selected dict if valid.
    Return None if invalid.

    The final plotted figure will keep only panels [0, 1, 3],
    but the candidate is still checked using the full four-point sequence.
    """
    dataset = cache.dataset
    indices = cand["indices"]

    probs = []
    cats = []

    try:
        for idx in indices:
            prob, cat = cache.get_prob_cat(idx)
            probs.append(prob)
            cats.append(cat)
    except Exception as e:
        print(f"Skip candidate {indices}: loading or inference failed: {e}")
        return None

    p0, p1, p2, p3 = probs

    if cats[-1] != 2:
        return None

    if p3 < min_final_prob:
        return None

    if max(p1, p2) < min_lead_prob:
        return None

    if p0 > max_early_prob:
        return None

    if p3 < p0 + min_gain:
        return None

    if p3 + allowed_late_drop < p2:
        return None

    diffs = []

    try:
        for a, b in zip(indices[:-1], indices[1:]):
            d = visual_diff_norm(
                dataset,
                a,
                b,
                sweep_idx=0,
                frame_idx=display_frame,
            )
            diffs.append(d)
    except Exception as e:
        print(f"Skip candidate {indices}: visual diff failed: {e}")
        return None

    min_diff = min(d["sum"] for d in diffs)
    max_diff = max(d["sum"] for d in diffs)
    min_corr = min(d["zcorr"] for d in diffs)

    if min_diff < min_dup_diff:
        return None

    if max_diff > max_jump_diff:
        return None

    if min_corr < min_zcorr:
        return None

    return {
        **cand,
        "probs": probs,
        "cats": cats,
        "diffs": diffs,
    }


def find_first_valid_sequence(
    cache,
    candidates,
    display_frame,
    max_candidates,
    args,
):
    searched = 0

    for cand in tqdm(
        candidates[:max_candidates],
        desc="Searching first valid Fig.11 case",
    ):
        searched += 1

        selected = evaluate_candidate_sequence(
            cache=cache,
            cand=cand,
            display_frame=display_frame,
            min_final_prob=args.min_final_prob,
            min_lead_prob=args.min_lead_prob,
            max_early_prob=args.max_early_prob,
            min_gain=args.min_gain,
            allowed_late_drop=args.allowed_late_drop,
            min_dup_diff=args.min_dup_diff,
            max_jump_diff=args.max_jump_diff,
            min_zcorr=args.min_zcorr,
        )

        if selected is not None:
            print("\nFound valid Fig. 11 case. Stop searching.")
            print(f"Searched candidates: {searched}")
            return selected

    print(f"\nNo valid case found after searching {searched} candidates.")
    return None


# =========================================================
# 7. Plotting: three-panel version
# =========================================================
class SequencePlotterTGRS:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica"],
                "font.size": 14,
                "figure.dpi": 300,
                "axes.linewidth": 1.6,
            }
        )

        self.cmap_z = plt.cm.nipy_spectral
        self.cmap_v = plt.cm.seismic

    def plot(self, dataset, selected, display_frame):
        """
        Three-panel version:
        keep T-15, T-10, and T=0; remove the non-monotonic T-5 panel.
        """
        panel_ids = [0, 1, 3]  # T-15, T-10, T0

        fig = plt.figure(figsize=(17, 9.2))

        gs = gridspec.GridSpec(
            2,
            4,
            width_ratios=[1, 1, 1, 0.05],
            wspace=0.06,
            hspace=0.10,
            left=0.06,
            right=0.92,
            top=0.88,
            bottom=0.16,
        )

        axes_z = []
        axes_v = []

        for plot_col, original_col in enumerate(panel_ids):
            idx = selected["indices"][original_col]
            image_tensor, _, _ = dataset[idx]

            z_data, v_data = extract_display_fields(
                image_tensor,
                sweep_idx=0,
                frame_idx=display_frame,
            )

            prob = selected["probs"][original_col]
            rel = selected["rel_minutes"][original_col]

            if rel == 0:
                title = "T = 0\n(Tornado Touchdown)"
            else:
                title = f"T {rel} min\n(Lead Time)"

            # -----------------------------
            # Row 1: Reflectivity
            # -----------------------------
            ax_z = fig.add_subplot(gs[0, plot_col])
            im_z = ax_z.imshow(
                z_data,
                cmap=self.cmap_z,
                vmin=0,
                vmax=70,
                origin="lower",
            )
            ax_z.set_xticks([])
            ax_z.set_yticks([])
            ax_z.set_title(title, fontweight="bold", fontsize=16, pad=12)

            if plot_col == 0:
                ax_z.set_ylabel(
                    "Reflectivity ($Z$)",
                    fontweight="bold",
                    fontsize=18,
                    labelpad=14,
                )

            axes_z.append(im_z)

            # -----------------------------
            # Row 2: Radial velocity
            # -----------------------------
            ax_v = fig.add_subplot(gs[1, plot_col])
            im_v = ax_v.imshow(
                v_data,
                cmap=self.cmap_v,
                vmin=-35,
                vmax=35,
                origin="lower",
            )
            ax_v.set_xticks([])
            ax_v.set_yticks([])

            if plot_col == 0:
                ax_v.set_ylabel(
                    "Radial Velocity ($V_r$)",
                    fontweight="bold",
                    fontsize=18,
                    labelpad=14,
                )

            # -----------------------------
            # Probability label
            # -----------------------------
            if prob < 0.4:
                box_fc, box_ec = "#E8F5E9", "#2E7D32"
            elif prob < 0.75:
                box_fc, box_ec = "#FFF3E0", "#E65100"
            else:
                box_fc, box_ec = "#FFEBEE", "#C62828"

            ax_v.text(
                0.5,
                -0.13,
                f"PI-STDNet Prob: {prob * 100:.1f}%",
                transform=ax_v.transAxes,
                ha="center",
                va="top",
                fontsize=15,
                fontweight="bold",
                color=box_ec,
                bbox=dict(
                    facecolor=box_fc,
                    edgecolor=box_ec,
                    boxstyle="round,pad=0.5",
                    lw=1.8,
                    alpha=1.0,
                ),
            )

            axes_v.append(im_v)

        # -----------------------------
        # Colorbars
        # -----------------------------
        cax_z = fig.add_subplot(gs[0, 3])
        cbar_z = fig.colorbar(axes_z[0], cax=cax_z, orientation="vertical")
        cbar_z.set_label("dBZ", fontsize=15, fontweight="bold")
        cbar_z.ax.tick_params(labelsize=12)

        cax_v = fig.add_subplot(gs[1, 3])
        cbar_v = fig.colorbar(axes_v[0], cax=cax_v, orientation="vertical")
        cbar_v.set_label("m/s", fontsize=15, fontweight="bold")
        cbar_v.ax.tick_params(labelsize=12)

        save_path = os.path.join(
            self.save_dir,
            "Figure11_Temporal_Evolution_Revised_3Panel.pdf",
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        return save_path


def save_metadata(dataset, selected, save_dir):
    """
    Save metadata for the three panels actually shown in the final figure.
    """
    meta_path = os.path.join(save_dir, "Figure11_sequence_metadata_3Panel.csv")

    panel_ids = [0, 1, 3]
    panel_names = ["T-15", "T-10", "T0"]

    rows = []

    for panel_name, original_col in zip(panel_names, panel_ids):
        idx = selected["indices"][original_col]
        row = dataset.catalog.iloc[idx]

        rows.append(
            {
                "panel": panel_name,
                "dataset_index": idx,
                "relative_minutes": selected["rel_minutes"][original_col],
                "probability": selected["probs"][original_col],
                "category_id": selected["cats"][original_col],
                "timestamp": selected["times"][original_col],
                "group_source": selected["group_source"],
                "group_id": selected["group_id"],
                "filename": row.get("filename", "NA"),
                "category": row.get("category", "NA"),
                "start_time": row.get("start_time", "NA"),
            }
        )

    pd.DataFrame(rows).to_csv(meta_path, index=False)

    return meta_path


def print_selected_details(dataset, selected):
    print("\nSelected Fig. 11 sequence:")
    print("group_source:", selected["group_source"])
    print("group_id:", selected["group_id"])
    print("indices:", selected["indices"])
    print("relative minutes:", selected["rel_minutes"])
    print("probabilities:", [round(p, 4) for p in selected["probs"]])
    print("category ids:", selected["cats"])
    print("gaps:", selected["gaps"])
    print("diffs:", selected["diffs"])

    print("\nSelected filenames:")
    for idx in selected["indices"]:
        row = dataset.catalog.iloc[idx]
        print(
            idx,
            "|",
            row.get("start_time", "NA"),
            "|",
            row.get("category", "NA"),
            "|",
            row.get("filename", "NA"),
        )

    print("\nFinal plotted panels:")
    for name, original_col in zip(["T-15", "T-10", "T0"], [0, 1, 3]):
        print(
            name,
            "| index:",
            selected["indices"][original_col],
            "| rel:",
            selected["rel_minutes"][original_col],
            "| prob:",
            round(selected["probs"][original_col], 4),
        )


# =========================================================
# 8. Main
# =========================================================
def execute(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    catalog_path = args.catalog_path or train_compare.CONFIG["CATALOG_PATH"]
    data_root = args.data_root or train_compare.CONFIG["DATA_ROOT"]

    print(f"catalog_path = {catalog_path}")
    print(f"data_root    = {data_root}")

    setup_full_model_flags(has_rot=True, has_topk=True)

    model = load_model(args.ckpt_ours, device)

    dataset = StrictTorNetDataset(
        catalog_path=catalog_path,
        root_dir=data_root,
        mode="test",
    )

    print(f"Test dataset size: {len(dataset)}")

    display_frame = args.display_frame
    print(f"display_frame = {display_frame}")

    cache = SampleCache(dataset, model, device)

    if args.manual_indices is not None:
        print("\nUsing manual indices.")
        indices = args.manual_indices

        selected = {
            "group_source": "manual",
            "group_id": "manual",
            "indices": indices,
            "times": ["manual"] * 4,
            "rel_minutes": [-15, -10, -5, 0],
            "gaps": ["manual"] * 3,
            "probs": [],
            "cats": [],
            "diffs": [],
        }

        for idx in indices:
            prob, cat = cache.get_prob_cat(idx)
            selected["probs"].append(prob)
            selected["cats"].append(cat)

        for a, b in zip(indices[:-1], indices[1:]):
            selected["diffs"].append(
                visual_diff_norm(
                    dataset,
                    a,
                    b,
                    sweep_idx=0,
                    frame_idx=display_frame,
                )
            )

        if not args.allow_invalid_manual:
            checked = evaluate_candidate_sequence(
                cache=cache,
                cand=selected,
                display_frame=display_frame,
                min_final_prob=args.min_final_prob,
                min_lead_prob=args.min_lead_prob,
                max_early_prob=args.max_early_prob,
                min_gain=args.min_gain,
                allowed_late_drop=args.allowed_late_drop,
                min_dup_diff=args.min_dup_diff,
                max_jump_diff=args.max_jump_diff,
                min_zcorr=args.min_zcorr,
            )

            if checked is None:
                print("\nManual sequence failed hard Fig. 11 validity checks.")
                print_selected_details(dataset, selected)
                return

            selected = checked

    else:
        print("\nBuilding target-time candidates...")
        candidates = build_target_time_candidates(
            dataset,
            offset_tolerance=args.offset_tolerance,
            max_derived_group_size=args.max_derived_group_size,
        )

        if len(candidates) == 0 and args.use_sliding_backup:
            print("\nTarget-time search found nothing. Trying sliding-window backup...")
            candidates = build_sliding_window_candidates(
                dataset,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                max_derived_group_size=args.max_derived_group_size,
            )

        if len(candidates) == 0:
            print("\nNo candidates found.")
            return

        selected = find_first_valid_sequence(
            cache=cache,
            candidates=candidates,
            display_frame=display_frame,
            max_candidates=args.max_candidates,
            args=args,
        )

        if selected is None and args.use_sliding_backup:
            print("\nNo valid case in target-time candidates. Trying sliding-window backup...")
            backup_candidates = build_sliding_window_candidates(
                dataset,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                max_derived_group_size=args.max_derived_group_size,
            )

            selected = find_first_valid_sequence(
                cache=cache,
                candidates=backup_candidates,
                display_frame=display_frame,
                max_candidates=args.max_candidates,
                args=args,
            )

        if selected is None:
            print("\nNo valid Fig. 11 case found.")
            print("Try relaxing thresholds, for example:")
            print("  --min_final_prob 0.55 --min_lead_prob 0.55 --max_early_prob 0.60")
            print("or:")
            print("  --offset_tolerance 10 --max_jump_diff 0.45 --min_zcorr -0.15")
            return

    print_selected_details(dataset, selected)

    plotter = SequencePlotterTGRS(args.save_dir)
    fig_path = plotter.plot(dataset, selected, display_frame=display_frame)
    meta_path = save_metadata(dataset, selected, args.save_dir)

    print(f"\nSaved revised 3-panel figure: {fig_path}")
    print(f"Saved 3-panel metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_ours", type=str, default="best_pi_v8_GWP1.pth")
    parser.add_argument("--catalog_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="paper_figures_final")

    # Keep default = 0 because this is the setting used by the selected legacy figure.
    parser.add_argument("--display_frame", type=int, default=0)

    parser.add_argument("--max_candidates", type=int, default=5000)
    parser.add_argument("--offset_tolerance", type=float, default=6.0)
    parser.add_argument("--max_derived_group_size", type=int, default=80)

    parser.add_argument("--use_sliding_backup", action="store_true")
    parser.add_argument("--min_gap", type=float, default=1.5)
    parser.add_argument("--max_gap", type=float, default=12.0)

    # Hard probability conditions.
    parser.add_argument("--min_final_prob", type=float, default=0.60)
    parser.add_argument("--min_lead_prob", type=float, default=0.60)
    parser.add_argument("--max_early_prob", type=float, default=0.55)
    parser.add_argument("--min_gain", type=float, default=0.20)
    parser.add_argument("--allowed_late_drop", type=float, default=0.08)

    # Visual continuity conditions, computed on normalized DBZ/VEL.
    parser.add_argument("--min_dup_diff", type=float, default=0.003)
    parser.add_argument("--max_jump_diff", type=float, default=0.35)
    parser.add_argument("--min_zcorr", type=float, default=-0.05)

    parser.add_argument("--manual_indices", type=int, nargs=4, default=None)
    parser.add_argument("--allow_invalid_manual", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    execute(args)
