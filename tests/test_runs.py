"""Run labels: architectures trained with or without coordinate-noise augmentation."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from molecular_prediction.config import Config, parse_run_label
from molecular_prediction.data.dataset import TargetStats
from molecular_prediction.experiments.main_comparison import METRICS_NAME, discover_runs
from molecular_prediction.models import build_model
from molecular_prediction.training.checkpoint import CHECKPOINT_NAME, load_checkpoint
from molecular_prediction.training.trainer import Trainer
from tests.conftest import random_molecule


def test_parse_run_label() -> None:
    assert parse_run_label("EGNN") == ("EGNN", 0.0)
    assert parse_run_label("GINDist+noise0.05") == ("GINDist", 0.05)
    for bad in ("EGNN+noise", "Transformer", "EGNN+noise-0.1", "egnn"):
        with pytest.raises(ValueError):
            parse_run_label(bad)


def _train(config: Config, label: str, output_dir: Path) -> dict:
    molecules: list[Data] = [random_molecule(6 + i % 4, seed=i) for i in range(40)]
    stats: TargetStats = TargetStats(mean=torch.zeros(4), std=torch.ones(4))
    torch.manual_seed(0)
    model = build_model(parse_run_label(label)[0], config)
    trainer = Trainer(
        model,
        label,
        molecules[:32],
        molecules[32:],
        stats,
        config,
        torch.device("cpu"),
        output_dir,
        seed=0,
    )
    return trainer.fit()


def test_training_noise_changes_training_only(small_config: Config, tmp_path: Path) -> None:
    small_config.training.epochs = 2
    small_config.training.batch_size = 8
    clean = _train(small_config, "EGNN", tmp_path / "clean")

    small_config.training.train_noise_sigma = 0.3
    noisy = _train(small_config, "EGNN+noise0.3", tmp_path / "noisy")
    assert noisy["train_loss"] != clean["train_loss"], "noise must reach the training batches"

    # The labelled checkpoint rebuilds the right architecture and records the noise.
    model, _, metadata = load_checkpoint(tmp_path / "noisy" / CHECKPOINT_NAME, torch.device("cpu"))
    assert metadata["architecture"] == "EGNN"
    assert metadata["train_noise_sigma"] == 0.3
    assert type(model).__name__ == "EGNN"


def test_discover_runs_follows_the_ladder(small_config: Config, tmp_path: Path) -> None:
    small_config.paths.results_dir = tmp_path
    for label in ("EGNN+noise0.05", "GIN", "EGNN", "GINDist+noise0.05", "GINDist"):
        metrics: Path = small_config.paths.run_results_dir / label / "seed_0" / METRICS_NAME
        metrics.parent.mkdir(parents=True)
        metrics.write_text("{}")
    (small_config.paths.run_results_dir / "EGNNStatic" / "seed_0").mkdir(parents=True)  # unfinished

    assert discover_runs(small_config) == [
        "GIN",
        "GINDist",
        "GINDist+noise0.05",
        "EGNN",
        "EGNN+noise0.05",
    ]
