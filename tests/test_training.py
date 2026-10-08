"""Metric aggregation, checkpoint policy and an end-to-end training run."""

from pathlib import Path

import torch
from torch_geometric.data import Data

from molecular_prediction.config import Config
from molecular_prediction.data.dataset import TargetStats
from molecular_prediction.models import build_model
from molecular_prediction.training.checkpoint import CHECKPOINT_NAME, load_checkpoint
from molecular_prediction.training.metrics import MAEAccumulator
from molecular_prediction.training.trainer import EarlyStopping, Trainer, make_loader
from tests.conftest import random_molecule


def test_mae_is_weighted_by_molecules_not_batches() -> None:
    predictions: torch.Tensor = torch.tensor([[1.0], [1.0], [1.0], [5.0]])
    targets: torch.Tensor = torch.zeros(4, 1)
    accumulator: MAEAccumulator = MAEAccumulator(1)
    accumulator.update(predictions[:3], targets[:3])
    accumulator.update(predictions[3:], targets[3:])
    assert accumulator.combined() == 2.0  # (1 + 1 + 1 + 5) / 4, not (1 + 5) / 2


def test_early_stopping() -> None:
    stopper: EarlyStopping = EarlyStopping(patience=2, min_delta=0.0)
    assert [stopper.step(v) for v in (3.0, 2.0, 2.5, 2.1)] == [True, True, False, False]
    assert stopper.should_stop and stopper.best == 2.0
    assert not EarlyStopping(patience=None, min_delta=0.0).should_stop


def test_trainer_restores_and_persists_the_best_model(small_config: Config, tmp_path: Path) -> None:
    small_config.training.epochs = 4
    small_config.training.batch_size = 8
    small_config.training.lr = 1e-2
    molecules: list[Data] = [random_molecule(6 + i % 4, seed=i) for i in range(48)]
    train, val = molecules[:32], molecules[32:]
    stats: TargetStats = TargetStats(mean=torch.zeros(4), std=torch.ones(4))

    torch.manual_seed(0)
    model = build_model("EGNN", small_config)
    trainer: Trainer = Trainer(
        model, "EGNN", train, val, stats, small_config, torch.device("cpu"), tmp_path, seed=0
    )
    history = trainer.fit()

    val_curve: list[float] = history["val_mae"]
    assert history["best_epoch"] == 1 + min(range(len(val_curve)), key=val_curve.__getitem__)

    # The in-memory model is the best checkpoint, not the last epoch.
    val_loader = make_loader(val, 8, shuffle=False, num_workers=0)
    restored_mae: float = float(trainer.evaluate(val_loader).mean())
    assert abs(restored_mae - min(val_curve)) < 1e-5

    # The checkpoint alone is enough to rebuild an identical model.
    reloaded, reloaded_stats, metadata = load_checkpoint(
        tmp_path / CHECKPOINT_NAME, torch.device("cpu")
    )
    assert metadata["epoch"] == history["best_epoch"]
    torch.testing.assert_close(reloaded_stats.std, stats.std)
    batch = next(iter(val_loader))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(batch), trainer.model(batch))
