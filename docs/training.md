# Training

Train the NC setting:

```bash
python scripts/train.py --config configs/train_nc.yaml
```

Train the WC setting:

```bash
python scripts/train.py --config configs/train_wc.yaml
```

Common parameters live in `configs/default.yaml`. Edit the YAML files for batch size, learning rate, epochs, random seed, data paths, checkpoint directory, output directory, and log directory.

Outputs are written to:

- checkpoints: `./checkpoints/...`
- logs: `./logs/...`
- generated outputs: `./outputs/...`

These directories are ignored by git.

