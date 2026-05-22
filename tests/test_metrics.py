import numpy as np

from pistdnet.metrics import best_csi, binary_detection_metrics, evaluate_view


def test_binary_detection_metrics_small_arrays():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    metrics = binary_detection_metrics(probs, labels, 0.5)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["csi"] >= 0


def test_best_csi_and_view_do_not_crash():
    probs = np.array([0.9, 0.4, 0.3, 0.2])
    labels = np.array([1, 0, 1, 0])
    assert "threshold" in best_csi(probs, labels)
    assert "auc" in evaluate_view(probs, labels)

