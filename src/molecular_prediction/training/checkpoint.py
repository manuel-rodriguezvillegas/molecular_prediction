"""Self-contained checkpoints.

A checkpoint stores everything needed to use the model later: weights, model
hyperparameters and the target normalisation statistics. Downstream experiments
therefore never recompute statistics or rely on the current config matching
the one used for training.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from molecular_prediction.config import Config, ModelConfig
from molecular_prediction.data.dataset import TargetStats
from molecular_prediction.models import MolecularGNN, build_model

CHECKPOINT_NAME: str = "checkpoint.pt"


def save_checkpoint(
    path: Path,
    model: MolecularGNN,
    model_name: str,
    config: Config,
    stats: TargetStats,
    epoch: int,
    val_mae: float,
) -> None:
    """Persist the model weights together with the metadata to rebuild it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "model_config": asdict(config.model),
        "target_indices": list(config.data.target_indices),
        "target_stats": stats.to_dict(),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_mae": val_mae,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[MolecularGNN, TargetStats, dict[str, Any]]:
    """Rebuild a trained model from its checkpoint, in eval mode.

    Returns:
        The model on ``device``, its normalisation statistics, and the raw
        checkpoint metadata (epoch, validation MAE, ...).
    """
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)

    config: Config = Config()
    config.model = ModelConfig(**payload["model_config"])
    config.data.target_indices = tuple(payload["target_indices"])

    model: MolecularGNN = build_model(payload["model_name"], config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    stats: TargetStats = TargetStats.from_dict(payload["target_stats"])
    metadata: dict[str, Any] = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, stats, metadata
