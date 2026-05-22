"""TorNet NetCDF preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from pistdnet.models.maas import PhysicalFeatureAmplifier, normalize_channel


def _safe_read(ds: xr.Dataset, key: str, fill_val: float, sweep_idx: int) -> np.ndarray:
    if key not in ds:
        return np.full((4, 120, 240), fill_val, dtype=np.float32)
    raw = ds[key].values
    if raw.ndim == 4:
        data = raw[..., sweep_idx] if raw.shape[-1] > sweep_idx else raw[..., 0]
    else:
        data = raw
    return np.nan_to_num(data, nan=fill_val).astype(np.float32)


def load_tornet_file(
    filepath: str | Path,
    n_sweeps: int = 2,
    n_frames: int = 4,
    channels_per_frame: int = 11,
) -> torch.Tensor:
    """Load one TorNet-style NetCDF sample as a normalized tensor."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"TorNet sample not found: {path}. See docs/data_preparation.md for the expected dataset layout."
        )

    all_channels = []
    try:
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            for sweep_idx in range(n_sweeps):
                dbz = _safe_read(ds, "DBZ", -30.0, sweep_idx)
                vel = _safe_read(ds, "VEL", 0.0, sweep_idx)
                kdp = _safe_read(ds, "KDP", 0.0, sweep_idx)
                rhohv = _safe_read(ds, "RHOHV", 0.0, sweep_idx)
                zdr = _safe_read(ds, "ZDR", 0.0, sweep_idx)
                width = _safe_read(ds, "WIDTH", 0.0, sweep_idx)

                frames_idx = list(range(min(dbz.shape[0], n_frames)))
                while len(frames_idx) < n_frames:
                    frames_idx.append(frames_idx[-1])

                for frame_idx in frames_idx:
                    f_dbz = dbz[frame_idx]
                    f_vel = vel[frame_idx]
                    f_shear = PhysicalFeatureAmplifier.compute_shear(f_vel)
                    f_debris = PhysicalFeatureAmplifier.compute_debris(f_dbz, rhohv[frame_idx])
                    anomaly_shears = PhysicalFeatureAmplifier.compute_multiscale_anomaly_shear(f_vel, f_dbz)

                    all_channels.extend(
                        [
                            normalize_channel(f_dbz, "DBZ"),
                            normalize_channel(f_vel, "VEL"),
                            normalize_channel(kdp[frame_idx], "KDP"),
                            normalize_channel(rhohv[frame_idx], "RHOHV"),
                            normalize_channel(zdr[frame_idx], "ZDR"),
                            normalize_channel(width[frame_idx], "WIDTH"),
                            normalize_channel(f_shear, "SHEAR"),
                            np.clip(f_debris, 0, 1),
                            np.clip(anomaly_shears[0] / 20.0, 0.0, 1.0),
                            np.clip(anomaly_shears[1] / 20.0, 0.0, 1.0),
                            np.clip(anomaly_shears[2] / 20.0, 0.0, 1.0),
                        ]
                    )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read TorNet sample {path}: {exc}") from exc

    data = np.clip(np.nan_to_num(np.stack(all_channels, axis=0).astype(np.float32), nan=0.0), -1.0, 2.0)
    expected_channels = n_sweeps * n_frames * channels_per_frame
    if data.shape[0] != expected_channels:
        raise ValueError(f"Unexpected channel count for {path}: got {data.shape[0]}, expected {expected_channels}.")
    return torch.from_numpy(data)

