# Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --config configs/eval.yaml --checkpoint /path/to/checkpoint.pth
```

Main metrics:

- CSI: Critical Success Index, `TP / (TP + FP + FN)`
- POD: Probability of Detection / recall, `TP / (TP + FN)`
- FAR: False Alarm Ratio, `FP / (TP + FP)`
- AUC: ROC area under the curve
- AP: Average precision

Set `evaluation.threshold` to a fixed threshold, or leave it as `null` to report the best CSI threshold over the default sweep.

