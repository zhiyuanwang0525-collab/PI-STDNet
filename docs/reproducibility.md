# Reproducibility

Reproducible runs depend on:

- the exact TorNet catalog and split columns
- the YAML config file
- the random seed
- the Python/PyTorch/CUDA environment
- the checkpoint used for evaluation

The default seed is `42`, set through `training.seed`.

Hardware, CUDA/PyTorch details, released split notes, and checkpoint identifiers will be updated upon publication when the final reproducibility statement is settled.
