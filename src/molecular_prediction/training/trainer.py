"""Model-agnostic training loop.

All models go through this exact loop (same optimiser, schedule, loss, data
order and checkpoint policy), so differences between them cannot come from
differences in training code.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from molecular_prediction.config import TARGET_NAMES, Config, LossName
from molecular_prediction.data.dataset import TargetStats
from molecular_prediction.models import MolecularGNN
from molecular_prediction.training.checkpoint import (
    CHECKPOINT_NAME,
    save_checkpoint,
)
from molecular_prediction.training.metrics import MAEAccumulator

logger: logging.Logger = logging.getLogger(__name__)

RESUME_NAME: str = "last.pt"

History = dict[str, list[float] | dict[str, list[float]]]


def build_loss(name: LossName) -> nn.Module:
    """Training objective, applied in normalised target space.

    * ``l1``: MAE, the QM9 evaluation metric; robust to outliers but its
      gradient has constant magnitude, so it does not slow down near the optimum.
    * ``smooth_l1``: Huber with delta = 1 (one standard deviation): quadratic for
      small errors (smooth convergence), linear for large ones (robust).
    * ``mse``: penalises large errors heavily; sensitive to outliers.
    """
    losses: dict[str, nn.Module] = {
        "l1": nn.L1Loss(),
        "smooth_l1": nn.SmoothL1Loss(beta=1.0),
        "mse": nn.MSELoss(),
    }
    return losses[name]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int = 0,
) -> DataLoader:
    """DataLoader with worker processes and a seeded shuffling order."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


class EarlyStopping:
    """Track the best validation score and stop after ``patience`` stale epochs.

    Args:
        patience: Epochs without improvement before stopping; ``None`` disables
            stopping (the best score is still tracked).
        min_delta: Minimum decrease that counts as an improvement.
    """

    def __init__(self, patience: int | None, min_delta: float) -> None:
        self.patience: int | None = patience
        self.min_delta: float = min_delta
        self.best: float = float("inf")
        self.stale_epochs: int = 0

    def step(self, value: float) -> bool:
        """Register a new validation score; return whether it is a new best."""
        if value < self.best - self.min_delta:
            self.best = value
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.stale_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"best": self.best, "stale_epochs": self.stale_epochs}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.best = float(state["best"])
        self.stale_epochs = int(state["stale_epochs"])


class Trainer:
    """Train one model with the shared protocol and keep its best checkpoint.

    Args:
        model: Untrained model.
        model_name: Registry name of the model.
        train_dataset: Training split (normalised targets).
        val_dataset: Validation split, used for model selection only.
        stats: Normalisation statistics, stored in the checkpoint.
        config: Full configuration.
        device: Device to train on.
        output_dir: Run directory (checkpoint and TensorBoard logs).
        seed: Seed of the shuffling order (weights are seeded by the caller).
        resume: Continue from ``last.pt`` in ``output_dir`` if it exists.
    """

    def __init__(
        self,
        model: MolecularGNN,
        model_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        stats: TargetStats,
        config: Config,
        device: torch.device,
        output_dir: Path,
        seed: int,
        resume: bool = True,
    ) -> None:
        self.model: MolecularGNN = model.to(device)
        self.model_name: str = model_name
        self.stats: TargetStats = stats
        self.config: Config = config
        self.device: torch.device = device
        self.checkpoint_path: Path = output_dir / CHECKPOINT_NAME
        self.resume_path: Path = output_dir / RESUME_NAME
        self.tensorboard_dir: Path = output_dir / "tensorboard"
        if not resume:
            self.resume_path.unlink(missing_ok=True)

        train_cfg = config.training
        self.epochs: int = train_cfg.epochs
        self.grad_clip_norm: float = train_cfg.grad_clip_norm
        self.num_targets: int = len(config.data.target_indices)
        self.target_names: tuple[str, ...] = TARGET_NAMES

        self.train_loader: DataLoader = make_loader(
            train_dataset, train_cfg.batch_size, True, config.data.num_workers, seed
        )
        self.val_loader: DataLoader = make_loader(
            val_dataset, train_cfg.batch_size, False, config.data.num_workers
        )

        self.criterion: nn.Module = build_loss(train_cfg.loss)
        self.optimizer: Adam = Adam(
            self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.early_stopping: EarlyStopping = EarlyStopping(train_cfg.patience, train_cfg.min_delta)

    def _train_epoch(self) -> tuple[float, torch.Tensor]:
        """One pass over the training set.

        Returns:
            Sample-weighted mean training loss and per-target training MAE
            (computed with the model in train mode, i.e. batch-norm batch stats).
        """
        self.model.train()
        mae: MAEAccumulator = MAEAccumulator(self.num_targets)
        loss_sum: float = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            predictions: torch.Tensor = self.model(batch)
            loss: torch.Tensor = self.criterion(predictions, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            loss_sum += loss.item() * batch.num_graphs
            mae.update(predictions, batch.y)

        return loss_sum / mae.count, mae.per_target()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> torch.Tensor:
        """Per-target MAE (normalised units) of the current model on ``loader``."""
        self.model.eval()
        mae: MAEAccumulator = MAEAccumulator(self.num_targets)
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            mae.update(self.model(batch), batch.y)
        return mae.per_target()

    def _log_epoch(
        self, epoch: int, train_loss: float, train_mae: torch.Tensor, val_mae: torch.Tensor
    ) -> None:
        lr: float = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("mae/train", float(train_mae.mean()), epoch)
        self.writer.add_scalar("mae/val", float(val_mae.mean()), epoch)
        self.writer.add_scalar("lr", lr, epoch)
        for i, name in enumerate(self.target_names):
            self.writer.add_scalar(f"mae_train/{name}", float(train_mae[i]), epoch)
            self.writer.add_scalar(f"mae_val/{name}", float(val_mae[i]), epoch)

    def _save_resume_state(
        self, epoch: int, history: History, best_epoch: int, train_seconds: float
    ) -> None:
        """Persist everything needed to continue exactly after ``epoch``.

        Written to a temporary file and atomically renamed, so a job killed
        mid-write never leaves a corrupt state behind.
        """
        state: dict[str, Any] = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "early_stopping": self.early_stopping.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_rng": self.train_loader.generator.get_state(),
        }
        tmp_path: Path = self.resume_path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.resume_path)

    def _load_resume_state(self) -> tuple[int, History, int, float]:
        """Restore the state saved by :meth:`_save_resume_state`."""
        state: dict[str, Any] = torch.load(
            self.resume_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.early_stopping.load_state_dict(state["early_stopping"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and len(state["cuda_rng"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        self.train_loader.generator.set_state(state["loader_rng"].cpu())
        return state["epoch"], state["history"], state["best_epoch"], state["train_seconds"]

    def fit(self) -> History:
        """Train for ``epochs`` epochs (or until early stopping), resuming if possible.

        The checkpoint on disk is only ever overwritten by a better validation
        MAE, and the best weights are loaded back into the model at the end, so
        any evaluation after ``fit`` uses the selected model, not the last one.

        After every epoch the full training state (weights, optimiser, schedule,
        early stopping, curves, random generators) is saved to ``last.pt``. If
        the job is killed, calling ``fit`` again continues from the last
        completed epoch with the same data order.

        Returns:
            Per-epoch curves: train loss, train/val MAE (combined and per target),
            learning rate, plus the best epoch and training time.
        """
        history: History = {
            "train_loss": [],
            "train_mae": [],
            "val_mae": [],
            "lr": [],
            "train_mae_per_target": {name: [] for name in self.target_names},
            "val_mae_per_target": {name: [] for name in self.target_names},
        }
        best_epoch: int = -1
        done_epochs: int = 0
        previous_seconds: float = 0.0
        if self.resume_path.exists():
            done_epochs, history, best_epoch, previous_seconds = self._load_resume_state()
            logger.info("%s: resuming after epoch %d.", self.model_name, done_epochs)

        # purge_step drops TensorBoard events logged after the resumed epoch.
        self.writer: SummaryWriter = SummaryWriter(
            log_dir=str(self.tensorboard_dir), purge_step=done_epochs + 1 if done_epochs else None
        )
        start: float = time.perf_counter()
        stopped: bool = self.early_stopping.should_stop

        for epoch in range(done_epochs + 1, self.epochs + 1):
            if stopped:
                break
            train_loss, train_mae = self._train_epoch()
            val_mae: torch.Tensor = self.evaluate(self.val_loader)
            val_combined: float = float(val_mae.mean())

            history["train_loss"].append(train_loss)
            history["train_mae"].append(float(train_mae.mean()))
            history["val_mae"].append(val_combined)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])
            for i, name in enumerate(self.target_names):
                history["train_mae_per_target"][name].append(float(train_mae[i]))
                history["val_mae_per_target"][name].append(float(val_mae[i]))
            self._log_epoch(epoch, train_loss, train_mae, val_mae)

            improved: bool = self.early_stopping.step(val_combined)
            if improved:
                best_epoch = epoch
                save_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    self.model_name,
                    self.config,
                    self.stats,
                    epoch,
                    val_combined,
                )

            logger.info(
                "%s | epoch %d/%d | train loss %.4f | val MAE %.4f%s",
                self.model_name,
                epoch,
                self.epochs,
                train_loss,
                val_combined,
                " *" if improved else "",
            )

            self.scheduler.step()
            elapsed: float = previous_seconds + time.perf_counter() - start
            self._save_resume_state(epoch, history, best_epoch, elapsed)
            stopped = self.early_stopping.should_stop
            if stopped:
                logger.info("Early stopping at epoch %d (best: %d).", epoch, best_epoch)

        self._restore_best()
        self.writer.close()

        history["best_epoch"] = best_epoch
        history["train_seconds"] = previous_seconds + time.perf_counter() - start
        return history

    def discard_resume_state(self) -> None:
        """Delete ``last.pt`` once the run's results have been saved."""
        self.resume_path.unlink(missing_ok=True)

    def _restore_best(self) -> None:
        """Load the best checkpoint back into the in-memory model."""
        payload: dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(payload["state_dict"])
