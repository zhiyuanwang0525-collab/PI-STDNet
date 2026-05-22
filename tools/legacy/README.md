# Legacy Research Scripts

These files are preserved from the original research workspace for traceability. They may contain old experiment-specific assumptions, old checkpoint names, and legacy figure workflows. Public use should prefer:

- `scripts/` for command-line entry points
- `src/pistdnet/` for importable model, data, metrics, training, evaluation, and visualization modules
- `configs/` for editable experiment settings

Local absolute data paths have been replaced with placeholders such as `/path/to/TorNet`.

## Moved Files

- root training/evaluation/plot scripts -> `tools/legacy/root_scripts/`
- original visualization scripts -> `tools/legacy/visualization_scripts/`
- checkpoints, logs, generated figures, and evaluation outputs -> external local artifact storage outside this repository
