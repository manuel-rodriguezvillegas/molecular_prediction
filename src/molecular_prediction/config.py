"""Experiment configuration.

All hyperparameters live in typed dataclasses. ``Config()`` gives the defaults
used for the reported results; the command-line interface only overrides the
handful of fields that change between runs (models, seeds, epochs, paths).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# QM9 column indices (PyG ordering) and their short names.
TARGET_INDICES: tuple[int, ...] = (0, 1, 4, 11)
TARGET_NAMES: tuple[str, ...] = ("mu", "alpha", "gap", "Cv")
TARGET_UNITS: tuple[str, ...] = ("D", "a0^3", "eV", "cal/mol/K")

# Ordered from least to most geometric information, so that every adjacent
# pair of models differs in exactly one ingredient.
MODEL_NAMES: tuple[str, ...] = ("GIN", "GINEdge", "GINDist", "EGNNStatic", "EGNN")

LossName = Literal["l1", "smooth_l1", "mse"]


@dataclass
class PathsConfig:
    """Where data is read from and where every artefact is written.

    Two roots with different lifetimes:

    * ``output_dir`` (git-ignored): heavy or machine-specific artefacts such as
      checkpoints, TensorBoard logs and caches.
    * ``results_dir`` (committed): small JSON summaries, per-run metrics and
      curves, and the key figures, so results travel with the repository.
    """

    data_dir: Path = Path("data/qm9")
    output_dir: Path = Path("outputs")
    results_dir: Path = Path("results")

    @property
    def runs_dir(self) -> Path:
        """One sub-directory per (model, seed) with its checkpoint and TensorBoard logs."""
        return self.output_dir / "runs"

    @property
    def run_results_dir(self) -> Path:
        """One sub-directory per (model, seed) with its metrics and training curves."""
        return self.results_dir / "runs"

    @property
    def figures_dir(self) -> Path:
        """Generated plots."""
        return self.results_dir / "figures"

    @property
    def cache_dir(self) -> Path:
        """Expensive intermediate results (e.g. curvature per molecule)."""
        return self.output_dir / "cache"


@dataclass
class DataConfig:
    """Dataset split and target selection."""

    target_indices: tuple[int, ...] = TARGET_INDICES
    # Train / validation sizes; the remaining ~10.8k molecules form the test set.
    split_sizes: tuple[int, int] = (110_000, 10_000)
    # The split seed is fixed and independent of the training seed, so every
    # model and every training seed is evaluated on exactly the same molecules.
    split_seed: int = 42
    # Optional cap on the test set (only used for quick smoke runs).
    test_size: int | None = None
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    ``hidden_dims`` is set per model so that all models have a similar number
    of trainable parameters (~200k +-1.5%, see ``python main.py params``).
    Without this, EGNN would have ~2.7x more parameters than GIN at equal width and
    any gain could be attributed to capacity rather than geometry.
    """

    hidden_dims: dict[str, int] = field(
        default_factory=lambda: {
            "GIN": 128,  # 200.7k parameters
            "GINEdge": 122,  # 198.1k
            "GINDist": 122,  # 202.0k
            "EGNNStatic": 82,  # 202.7k
            "EGNN": 76,  # 198.8k
        }
    )
    num_layers: int = 4
    num_rbf: int = 32
    rbf_cutoff: float = 5.0  # Angstrom; covers bonds and moderately noisy geometries
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Optimisation hyperparameters, identical for every model."""

    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 300
    batch_size: int = 96
    lr: float = 5e-4
    weight_decay: float = 0.0
    loss: LossName = "l1"
    grad_clip_norm: float = 10.0
    # Early stopping on validation MAE; ``None`` disables it. The best
    # checkpoint is always tracked and restored regardless of this value.
    patience: int | None = 50
    min_delta: float = 1e-4


@dataclass
class NoiseConfig:
    """Inference-time coordinate-noise ablation."""

    sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)  # Angstrom
    num_realisations: int = 5
    noise_seed: int = 1234


@dataclass
class CurvatureConfig:
    """Ollivier-Ricci over-squashing analysis."""

    alpha: float = 0.5  # laziness of the random walk
    num_quantiles: int = 4
    num_processes: int = 8


@dataclass
class Config:
    """Root configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration (paths become strings)."""
        raw: dict[str, Any] = asdict(self)
        raw["paths"] = {key: str(value) for key, value in raw["paths"].items()}
        return raw
