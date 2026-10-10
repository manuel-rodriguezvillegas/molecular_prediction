"""Inference helpers shared by the downstream experiments."""

import logging
from pathlib import Path

import torch
from torch_geometric.data import Batch, Dataset

from molecular_prediction.config import Config
from molecular_prediction.data.transforms import add_coordinate_noise
from molecular_prediction.models import MolecularGNN
from molecular_prediction.training.checkpoint import CHECKPOINT_NAME
from molecular_prediction.training.trainer import make_loader
from molecular_prediction.training.utils import run_dir

logger: logging.Logger = logging.getLogger(__name__)


def collate_batches(dataset: Dataset, batch_size: int) -> list[Batch]:
    """Group a split into batches once, in a fixed order, without worker processes.

    The downstream experiments run hundreds of inference passes over the same
    test set (models x seeds x noise levels x realisations). Building a new
    multi-worker DataLoader for each pass leaks file descriptors through
    PyTorch's shared-memory tensor passing until the process hits its limit
    ("received 0 items of ancdata"). Collating once and reusing the batches
    avoids that, and is faster.
    """
    return list(make_loader(dataset, batch_size, shuffle=False, num_workers=0))


@torch.no_grad()
def predict(
    model: MolecularGNN,
    batches: list[Batch],
    device: torch.device,
    sigma: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions and targets for every molecule of ``batches``, in order.

    Args:
        model: Trained model (put in eval mode here).
        batches: Pre-collated batches with normalised targets (see :func:`collate_batches`).
        device: Inference device.
        sigma: Standard deviation of the coordinate noise, in Angstrom.
        generator: Source of the noise. With the same seed, every model receives
            the same perturbed geometries, which makes comparisons paired.

    Returns:
        Normalised predictions and targets, both [num_molecules, num_targets].
    """
    if sigma > 0.0 and generator is None:
        raise ValueError("A seeded generator is required when sigma > 0.")

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in batches:
        if sigma > 0.0:
            batch = add_coordinate_noise(batch, sigma, generator)
        # In place for the shared batches: they move to the device on the first
        # pass and stay there, so later passes skip the host-to-device copy.
        batch = batch.to(device, non_blocking=True)
        predictions.append(model(batch).cpu())
        targets.append(batch.y.cpu())
    return torch.cat(predictions), torch.cat(targets)


def available_checkpoints(config: Config, model_name: str) -> dict[int, Path]:
    """Checkpoints of the configured seeds that exist on disk, keyed by seed."""
    found: dict[int, Path] = {}
    for seed in config.training.seeds:
        path: Path = run_dir(config.paths.runs_dir, model_name, seed) / CHECKPOINT_NAME
        if path.exists():
            found[seed] = path
        else:
            logger.warning("Missing checkpoint for %s seed %d: %s", model_name, seed, path)
    if not found:
        raise FileNotFoundError(
            f"No trained checkpoint for {model_name}. Run `python main.py train` first."
        )
    return found
