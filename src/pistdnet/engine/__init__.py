"""Training and evaluation engines."""

from .evaluator import collect_predictions, evaluate_model
from .trainer import train_model

__all__ = ["train_model", "collect_predictions", "evaluate_model"]

