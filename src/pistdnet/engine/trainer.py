"""Training loop for PI-STDNet."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from pistdnet.data import TorNetDataset
from pistdnet.engine.evaluator import collect_predictions
from pistdnet.metrics import evaluate_view
from pistdnet.models import EMA, PI_STDNet, V8Loss
from pistdnet.utils.checkpoint import save_checkpoint
from pistdnet.utils.config import get_model_config, resolve_catalog_path
from pistdnet.utils.logger import setup_logger
from pistdnet.utils.seed import set_seed


def _make_loader(dataset: TorNetDataset, config: dict, train: bool) -> DataLoader:
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    if train:
        sample_weights = np.zeros(len(dataset))
        sample_weights[dataset.cat_ids == 2] = 7.0
        sample_weights[dataset.cat_ids == 1] = 3.0
        sample_weights[dataset.cat_ids == 0] = 1.0
        kwargs["sampler"] = WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        kwargs["shuffle"] = False
    return DataLoader(dataset, **kwargs)


def _learning_rate(epoch: int, config: dict) -> float:
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("learning_rate", 5e-4))
    warmup = int(train_cfg.get("warmup_epochs", 5))
    epochs = int(train_cfg.get("epochs", 20))
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    return 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))


def train_model(config: dict) -> None:
    """Train PI-STDNet from a config dictionary."""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    log_dir = Path(train_cfg.get("log_dir", "./logs"))
    logger = setup_logger(log_file=log_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting PI-STDNet training on %s", device)

    model_cfg = get_model_config(config)
    catalog_path = resolve_catalog_path(config)
    dataset_root = Path(data_cfg.get("dataset_root", "/path/to/TorNet"))
    train_ds = TorNetDataset(catalog_path, dataset_root, mode=data_cfg.get("train_split", "train"), n_sweeps=model_cfg.get("n_sweeps", 2), n_frames=model_cfg.get("n_frames", 4), channels_per_frame=model_cfg.get("channels_per_frame", 11))
    test_ds = TorNetDataset(catalog_path, dataset_root, mode=data_cfg.get("val_split", "test"), n_sweeps=model_cfg.get("n_sweeps", 2), n_frames=model_cfg.get("n_frames", 4), channels_per_frame=model_cfg.get("channels_per_frame", 11))
    train_loader = _make_loader(train_ds, config, train=True)
    test_loader = _make_loader(test_ds, config, train=False)

    model = PI_STDNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.05)))
    criterion = V8Loss(label_smoothing=float(model_cfg.get("label_smoothing", 0.1))).to(device)
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.9995)))
    accum_steps = int(train_cfg.get("accum_steps", 2))
    best_wc_csi = 0.0

    for epoch in range(int(train_cfg.get("epochs", 20))):
        lr = _learning_rate(epoch, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg.get('epochs', 20)}")
        for step, (images, labels, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, _ = criterion(*model(images), labels)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
            train_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        ema.apply_shadow()
        probs, labels, cats = collect_predictions(model, test_loader, device, use_amp)
        ema.restore()
        nc = evaluate_view(probs, labels)
        wc = evaluate_view(probs, labels, mask=(cats == 2) | (cats == 1))
        logger.info("Epoch %d | loss %.4f | lr %.2e | NC CSI %.4f | WC CSI %.4f", epoch + 1, train_loss / max(1, len(train_loader)), lr, nc["csi"], wc["csi"])

        if wc["csi"] > best_wc_csi:
            best_wc_csi = float(wc["csi"])
            ema.apply_shadow()
            ckpt_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
            save_checkpoint(ckpt_dir / "best_pi_stdnet.pth", model, nc_csi=nc["csi"], wc_csi=wc["csi"], epoch=epoch + 1, config=config)
            ema.restore()
            logger.info("Saved new best checkpoint to %s", ckpt_dir / "best_pi_stdnet.pth")

