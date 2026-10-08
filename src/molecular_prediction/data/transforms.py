"""Per-molecule and per-batch data transforms."""

import copy

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform


class SelectAndNormaliseTargets(BaseTransform):
    """Keep only the selected QM9 targets and standardise them.

    Applied lazily on every ``dataset[i]``. The operation is out-of-place on
    purpose: PyG caches Data objects that share storage with the dataset, so
    writing into ``data.y`` in place would silently corrupt the cache.

    Args:
        target_indices: QM9 columns to keep.
        mean: Per-target mean of the training split, shape [num_targets].
        std: Per-target standard deviation of the training split.
    """

    def __init__(self, target_indices: list[int], mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_indices: list[int] = list(target_indices)
        self.mean: torch.Tensor = mean
        self.std: torch.Tensor = std

    def forward(self, data: Data) -> Data:
        """Replace ``data.y`` [1, 19] by its normalised selection [1, num_targets]."""
        data.y = (data.y[:, self.target_indices] - self.mean) / self.std
        return data


def add_coordinate_noise(batch: Batch, sigma: float, generator: torch.Generator) -> Batch:
    """Return a copy of ``batch`` with isotropic Gaussian noise on ``pos``.

    The noise is drawn on the CPU from an explicit generator, so for a fixed
    seed and a fixed (unshuffled) loader every model sees exactly the same
    perturbed geometries. That turns the noise ablation into a paired
    comparison between models.

    Args:
        batch: Batch of molecules.
        sigma: Standard deviation of the noise, in Angstrom.
        generator: CPU random generator.

    Returns:
        A shallow copy of the batch whose ``pos`` has been perturbed.
    """
    noise: torch.Tensor = torch.randn(batch.pos.shape, generator=generator, dtype=batch.pos.dtype)
    noisy: Batch = copy.copy(batch)  # shares every tensor except the new pos
    noisy.pos = batch.pos + sigma * noise.to(batch.pos.device)
    return noisy
