"""QM9 loading, random splitting and target normalisation."""

from dataclasses import dataclass

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

from molecular_prediction.config import Config
from molecular_prediction.data.transforms import SelectAndNormaliseTargets

# QM9 node features (PyG): one-hot over (H, C, N, O, F), atomic number,
# aromatic flag, sp/sp2/sp3 hybridisation flags and number of attached H.
NODE_FEATURE_DIM: int = 11
EDGE_FEATURE_DIM: int = 4  # one-hot bond type: single, double, triple, aromatic
HYDROGEN_FEATURE: int = 0  # column of the one-hot that marks hydrogen atoms


@dataclass(frozen=True)
class TargetStats:
    """Per-target mean and standard deviation, computed on the train split only."""

    mean: torch.Tensor  # [num_targets]
    std: torch.Tensor  # [num_targets]

    def normalise(self, y: torch.Tensor) -> torch.Tensor:
        """Map targets in physical units to zero mean, unit variance."""
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def denormalise(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalise`."""
        return y * self.std.to(y.device) + self.mean.to(y.device)

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-serialisable representation, stored next to every checkpoint."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, raw: dict[str, list[float]]) -> "TargetStats":
        """Inverse of :meth:`to_dict`."""
        return cls(mean=torch.tensor(raw["mean"]), std=torch.tensor(raw["std"]))

    @classmethod
    def fit(cls, y: torch.Tensor) -> "TargetStats":
        """Estimate statistics from a [num_molecules, num_targets] matrix."""
        return cls(mean=y.mean(dim=0), std=y.std(dim=0))


@dataclass
class QM9Splits:
    """The three splits with targets already selected and normalised."""

    train: Dataset
    val: Dataset
    test: Dataset
    stats: TargetStats


def split_indices(
    num_items: int,
    sizes: tuple[int, int],
    seed: int,
    test_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random train/val/test partition of ``range(num_items)``.

    QM9 is stored in GDB enumeration order, which is correlated with molecular
    size, so a contiguous slice would put systematically different molecules in
    the test set. A seeded permutation makes the split i.i.d. and reproducible.

    Args:
        num_items: Size of the dataset.
        sizes: Number of train and validation items; the rest goes to test.
        seed: Seed of the permutation.
        test_size: Optional cap on the number of test items.

    Returns:
        Disjoint index tensors (train, val, test).
    """
    num_train, num_val = sizes
    if num_train + num_val >= num_items:
        raise ValueError(f"Split {sizes} leaves no test data out of {num_items}.")

    generator: torch.Generator = torch.Generator().manual_seed(seed)
    perm: torch.Tensor = torch.randperm(num_items, generator=generator)
    test_end: int | None = None if test_size is None else num_train + num_val + test_size
    return (
        perm[:num_train],
        perm[num_train : num_train + num_val],
        perm[num_train + num_val : test_end],
    )


def load_splits(config: Config) -> QM9Splits:
    """Load QM9, split it randomly and normalise targets with train statistics.

    After this call every ``data.y`` has shape [1, num_targets] and contains only
    the selected targets in normalised units.
    """
    dataset: QM9 = QM9(str(config.paths.data_dir))
    target_indices: list[int] = list(config.data.target_indices)

    train_idx, val_idx, test_idx = split_indices(
        len(dataset), config.data.split_sizes, config.data.split_seed, config.data.test_size
    )

    # ``dataset.y`` is the full [N, 19] target matrix, read without building
    # 130k Data objects. Statistics come from the training molecules only.
    all_targets: torch.Tensor = dataset.y[:, target_indices]
    stats: TargetStats = TargetStats.fit(all_targets[train_idx])

    transform: SelectAndNormaliseTargets = SelectAndNormaliseTargets(
        target_indices, stats.mean, stats.std
    )
    splits: list[Dataset] = []
    for indices in (train_idx, val_idx, test_idx):
        subset: Dataset = dataset.index_select(indices)
        subset.transform = transform
        splits.append(subset)

    return QM9Splits(train=splits[0], val=splits[1], test=splits[2], stats=stats)
