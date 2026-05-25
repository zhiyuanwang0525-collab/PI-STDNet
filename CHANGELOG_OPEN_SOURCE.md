# Open-Source Preparation Changelog

## 2026-05-21

- Reorganized model, data, training, evaluation, metrics, utilities, and visualization modules under `src/pistdnet/`.
- Added config-based training, evaluation, inference, reproducibility, and temporal visualization entry points under `scripts/`.
- Added `README.md`, documentation files, dependency files, tests, `.gitignore`, and safety checking utilities.
- Replaced local absolute dataset paths in public configs with example paths such as `/path/to/TorNet`.
- Preserved original experimental logic by moving legacy research scripts into `tools/legacy/`.
- Moved local checkpoints, logs, generated figures, and evaluation outputs out of the public source tree.
- Added release notes for licensing, dataset access, pretrained-weight policy, hardware notes, and citation metadata.
- Added Apache-2.0 license metadata after license selection.
- Added author-person copyright notice, official TorNet data wording, and non-release policy for pretrained weights.
- Renamed citation metadata to `CITATION.cff` and removed unverified publication metadata fields.

## Legacy Script Moves

- `train.py`, `trainfix.py`, `trainV8fix.py`, `trainV8GWP.py`, `train_Ablation.py`, `train_baseline.py` -> `tools/legacy/root_scripts/`
- `ablation.py`, `ablation_v83.py`, `attentionbaseline.py`, `Evaluate fair comparison.py`, `eval_slider.py` -> `tools/legacy/root_scripts/`
- `graph.py`, `grab3.py`, `plot_figures.py`, `train_` -> `tools/legacy/root_scripts/`
- `画图/*.py` -> `tools/legacy/visualization_scripts/`
- original checkpoints, logs, generated figures, CSV outputs, `.vscode`, and caches -> external local artifact directory, not part of the public repository
