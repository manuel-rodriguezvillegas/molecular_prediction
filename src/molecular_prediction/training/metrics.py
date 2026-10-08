"""Streaming error metrics."""

import torch


class MAEAccumulator:
    """Per-target mean absolute error accumulated over batches.

    Errors are summed per molecule and divided by the total number of molecules
    at the end, so a small final batch weighs exactly as much as its size.
    Averaging per-batch means instead would over-weight that last batch.

    Args:
        num_targets: Number of predicted properties.
    """

    def __init__(self, num_targets: int) -> None:
        self.abs_error_sum: torch.Tensor = torch.zeros(num_targets, dtype=torch.float64)
        self.count: int = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of [batch_size, num_targets] predictions and targets."""
        errors: torch.Tensor = (predictions.detach() - targets).abs().sum(dim=0)
        self.abs_error_sum += errors.double().cpu()
        self.count += targets.size(0)

    def per_target(self) -> torch.Tensor:
        """MAE of each target, shape [num_targets]."""
        if self.count == 0:
            raise RuntimeError("No batch has been accumulated.")
        return (self.abs_error_sum / self.count).float()

    def combined(self) -> float:
        """Mean of the per-target MAEs.

        Only meaningful in normalised units, where every target has unit
        variance; in physical units it would mix Debye, eV, ...
        """
        return float(self.per_target().mean())
