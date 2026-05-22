"""Physics-derived radar feature encoders used by PI-STDNet."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage


CHANNEL_MIN_MAX = {
    "DBZ": (-30.0, 80.0),
    "VEL": (-50.0, 50.0),
    "KDP": (-2.0, 5.0),
    "RHOHV": (0.2, 1.05),
    "ZDR": (-5.0, 8.0),
    "WIDTH": (0.0, 10.0),
    "SHEAR": (-10.0, 10.0),
    "DEBRIS": (0.0, 1.0),
}


def normalize_channel(arr: np.ndarray, key: str) -> np.ndarray:
    """Normalize a radar variable to the range used by the original scripts."""
    mn, mx = CHANNEL_MIN_MAX[key]
    return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)


class PhysicalFeatureAmplifier:
    """Compute MAAS, azimuthal shear, and debris-related physics features."""

    @staticmethod
    def compute_shear(vel: np.ndarray) -> np.ndarray:
        vel = np.nan_to_num(vel, nan=0.0)
        shear = np.gradient(vel, axis=1)
        return np.nan_to_num(shear, nan=0.0)

    @staticmethod
    def compute_debris(dbz: np.ndarray, rhohv: np.ndarray) -> np.ndarray:
        dbz = np.nan_to_num(dbz, nan=-30.0)
        rhohv = np.nan_to_num(rhohv, nan=0.0)
        d_norm = np.clip((dbz - (-30)) / 110.0, 0, 1)
        r_norm = np.clip(rhohv, 0, 1)
        return d_norm * (1.0 - r_norm)

    @staticmethod
    def compute_multiscale_anomaly_shear(vel: np.ndarray, dbz: np.ndarray) -> list[np.ndarray]:
        """Compute Multi-scale Anomaly Azimuthal Shear (MAAS) maps."""
        vel = np.nan_to_num(vel, nan=0.0)
        valid_mask = dbz > -29.0
        eroded_mask = ndimage.binary_erosion(valid_mask, iterations=2)

        results = []
        for kernel in [3, 5, 7]:
            half = kernel // 2
            padded = np.pad(vel, ((half, half), (0, 0)), mode="wrap")
            az_shear = np.zeros_like(vel)
            for i in range(vel.shape[0]):
                window = padded[i : i + kernel, :]
                az_shear[i] = window.max(axis=0) - window.min(axis=0)

            background = ndimage.uniform_filter(az_shear, size=15)
            anomaly = np.clip(az_shear - background, 0, None)
            results.append(anomaly * eroded_mask)
        return results

