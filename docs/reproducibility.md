# Reproducibility

Reproducible runs depend on:

- the exact TorNet catalog and split columns
- the YAML config file
- the random seed
- the Python/PyTorch/CUDA environment
- the checkpoint used for evaluation

The default seed is `42`, set through `training.seed`.

TODO: Confirm the exact hardware, CUDA version, PyTorch version, released data split, and public checkpoint identifiers used for the paper experiments.

