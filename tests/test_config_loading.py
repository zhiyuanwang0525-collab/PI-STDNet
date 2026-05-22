from pathlib import Path

from pistdnet.utils.config import load_config


def test_default_config_loads():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert config["data"]["dataset_root"] == "/path/to/TorNet"
    assert config["training"]["seed"] == 42


def test_inherited_train_config_loads():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "train_nc.yaml")
    assert config["model"]["use_pam"] is True
    assert config["training"]["checkpoint_dir"] == "./checkpoints/nc"
