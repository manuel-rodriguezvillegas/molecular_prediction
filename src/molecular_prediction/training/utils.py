"""Reproducibility, device and I/O helpers."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger: logging.Logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    This fixes weight initialisation and data order. Bit-exact reproducibility
    on GPU is not guaranteed: scatter-sum relies on atomic additions whose order,
    and therefore rounding, varies between runs. Differences are at float
    precision, far below seed-to-seed variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU if it is unavailable."""
    if requested == "cuda" and torch.cuda.is_available():
        # TF32 matmuls on Ampere/Hopper: ~2x faster, ~1e-3 relative precision,
        # irrelevant next to the noise in the targets.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested != "cpu":
        logger.warning("Device '%s' not available, falling back to CPU.", requested)
    return torch.device("cpu")


def save_json(payload: Any, path: Path) -> None:
    """Write ``payload`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open() as handle:
        return json.load(handle)


def run_dir(runs_root: Path, model_name: str, seed: int) -> Path:
    """Directory holding every artefact of one (model, seed) training run."""
    return runs_root / model_name / f"seed_{seed}"
