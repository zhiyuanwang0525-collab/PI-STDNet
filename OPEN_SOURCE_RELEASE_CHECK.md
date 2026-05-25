# Open Source Release Check

## Checks Passed

- README uses restrained research-release wording and does not claim acceptance, publication, or final paper status.
- README uses "research implementation" wording.
- Code-release citation metadata is available in `CITATION.cff` and does not invent publication identifiers.
- Author and affiliation metadata are aligned with the manuscript under review.
- Path scan passed with `python tools/check_paths.py .`.
- CLI help works for `scripts/train.py`, `scripts/evaluate.py`, `scripts/infer.py`, and `scripts/visualize_temporal_case.py`.
- No full training was attempted without real TorNet data.
- Missing dependency/data paths are handled with clear messages directing users to environment files or `docs/data_preparation.md`.
- `.gitignore` covers local datasets, outputs, checkpoints, weights, generated figures, and common binary training artifacts.
- No model weights, raw radar data, logs, generated result tables, or large generated figures are present in the public source tree.
- Legacy research scripts are preserved under `tools/legacy/` with a repository-level legacy note and per-script legacy headers.

## Publication-Time Updates

- Add final manuscript bibliographic metadata after publication.
- Add exact TorNet dataset source link if the manuscript or data provider requires a specific reference.
- Add hardware and CUDA/PyTorch details after the final reproducibility statement is settled.
- Add a `NOTICE` file or source-file copyright headers if required by project owners.

## Recommended GitHub Release Steps

1. Install the release environment from `environment.yml` or `requirements.txt`.
2. Run `python -m pytest -q` in the configured environment.
3. Run `python tools/check_paths.py .`.
4. Check that `git status --short --ignored` does not show ignored source directories.
5. Check that no local dataset, checkpoint, output, or generated figure directory is inside the repository.
6. Add a `NOTICE` file or source-file copyright headers if required by project owners.
7. Update `CITATION.cff` with final paper metadata after publication.
8. Create a clean GitHub repository and push only source, configs, docs, tests, and lightweight examples.
9. Add a release note that TorNet data must be obtained officially and pretrained weights are not publicly released.

## File Types Not Recommended For Public Release

- Raw or processed radar data: `.nc`, `.h5`, `.hdf5`, `.npy`, `.npz`, `.parquet`
- Model artifacts: `.pth`, `.pt`, `.ckpt`, `.onnx`
- Training/evaluation byproducts: `.log`, generated `.csv`, generated figures, large `.png`/`.pdf`
- Local workspace folders: `.vscode/`, `.idea/`, cache folders
- Local experiment outputs: `outputs/`, `results/`, `runs/`, `logs/`, `checkpoints/`, `weights/`
- Local dataset folders: `data/`, `datasets/`, `TorNet/`

## First Release Checklist

- [x] Apache-2.0 license selected and committed.
- [x] Code-release citation metadata added; final paper metadata pending.
- [ ] README reviewed by all authors.
- [ ] Data preparation instructions reviewed against the TorNet access process.
- [x] Public checkpoint policy is settled: pretrained weights are not released.
- [ ] Tests pass in a fresh environment.
- [ ] `tools/check_paths.py .` passes.
- [ ] GitHub repository contains no local artifacts or private paths.
