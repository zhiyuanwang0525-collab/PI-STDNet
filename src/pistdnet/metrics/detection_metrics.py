"""Tornado detection metrics used by the PI-STDNet scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_detection_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute CSI, POD, FAR, precision, recall, and confusion counts."""
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)
    preds = (probs > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    pod = tp / (tp + fn + 1e-7)
    far = fp / (tp + fp + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = pod
    csi = tp / (tp + fn + fp + 1e-7)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "csi": float(csi),
        "pod": float(pod),
        "far": float(far),
        "precision": float(precision),
        "recall": float(recall),
    }


def best_csi(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None) -> dict[str, float | int]:
    """Find the threshold with the best CSI."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    best = None
    for threshold in thresholds:
        metrics = binary_detection_metrics(probs, labels, float(threshold))
        if best is None or metrics["csi"] > best["csi"]:
            best = metrics
    return best or binary_detection_metrics(probs, labels, 0.5)


def safe_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return ROC AUC, or NaN when only one class is present."""
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return average precision, or NaN when only one class is present."""
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def evaluate_view(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None, threshold: float | None = None) -> dict[str, float | int]:
    """Evaluate one NC/WC view with optional fixed threshold."""
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)
    if mask is not None:
        probs = probs[mask]
        labels = labels[mask]
    metrics = binary_detection_metrics(probs, labels, threshold) if threshold is not None else best_csi(probs, labels)
    metrics["auc"] = safe_auc(probs, labels)
    metrics["ap"] = safe_average_precision(probs, labels)
    return metrics

