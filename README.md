# PI-STDNet: Physics-Guided Spatiotemporal Framework for Tornado Detection

This repository contains a research implementation of PI-STDNet.

PI-STDNet integrates observation-derived radar physical priors with a ConvNeXt-U-Net spatiotemporal backbone for tornado detection on TorNet-style radar data. The key physics-guided components include Multi-scale Anomaly Azimuthal Shear encoding and a Physics Attention Module.

## Repository Status

This code release is prepared for research reproducibility. Dataset files, pretrained weights, training logs, and generated paper figures are not included in the public source tree. TorNet data should be obtained from the official TorNet distribution channel. Pretrained weights are not publicly released at this time. TODO: confirm hardware details and final citation.

## Installation

```bash
conda env create -f environment.yml
conda activate pistdnet
pip install -r requirements.txt
pip install -e .
python -c "import pistdnet; print(pistdnet.__version__)"
```

The PyTorch/CUDA build may need to be adjusted for your local GPU and driver.

## Data Preparation

Download or request TorNet through the official TorNet distribution channel, then prepare a TorNet-style directory outside the repository:

```text
/path/to/TorNet/
  catalog.csv
  tornet_YYYY/
    sample_file.nc
```

Then update `dataset_root` and `catalog_path` in `configs/*.yaml`. Do not commit raw radar files or local absolute paths. See `docs/data_preparation.md`.

## Training

NC setting:

```bash
python scripts/train.py --config configs/train_nc.yaml
```

WC setting:

```bash
python scripts/train.py --config configs/train_wc.yaml
```

Training parameters are configured in YAML files under `configs/`.

## Evaluation

```bash
python scripts/evaluate.py --config configs/eval.yaml --checkpoint /path/to/checkpoint.pth
```

The decision threshold can be configured with `evaluation.threshold`. When it is `null`, the evaluation script reports the best CSI threshold over the configured sweep.

## Visualization

Temporal evolution entry point:

```bash
python scripts/visualize_temporal_case.py --config configs/eval.yaml --case-id CASE_ID
```

Legacy figure scripts from the original research workspace are preserved under `tools/legacy/visualization_scripts/`.

## Reproducibility

Use the YAML config files, fixed random seed, and the same checkpoint to reproduce a run. The default seed is `42`. Environment details are captured in `requirements.txt` and `environment.yml`. TODO: confirm hardware, CUDA version, exact data split release, and final checkpoint identifiers.

## Citation

If you use this repository, please cite:

```bibtex
@article{TODO_PI_STDNet,
  title = {PI-STDNet: Physics-Guided Spatiotemporal Framework for Tornado Detection},
  author = {Wang, Zhiyuan and Zheng, Kun and Zhang, Jiaolong},
  journal = {TODO},
  year = {TODO}
}
```

Affiliation: China University of Geosciences, Wuhan. Corresponding author: Kun Zheng, ZhengK@cug.edu.cn. TODO: update the citation after publication details are confirmed.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.

Copyright 2026 Zhiyuan Wang, Kun Zheng, and Jiaolong Zhang.
