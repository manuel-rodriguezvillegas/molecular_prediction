"""Model-agnostic training loop.

All models go through this exact loop (same optimiser, schedule, loss, data
order and checkpoint policy), so differences between them cannot come from
differences in training code.
"""

import logging
import time
from pathlib import Path

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
    ) -> None:
        self.model: MolecularGNN = model.to(device)
        self.model_name: str = model_name
        self.stats: TargetStats = stats
        self.config: Config = config
        self.device: torch.device = device
        self.checkpoint_path: Path = output_dir / CHECKPOINT_NAME

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
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

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

    def fit(self) -> History:
        """Train for ``epochs`` epochs (or until early stopping).

        The checkpoint on disk is only ever overwritten by a better validation
        MAE, and the best weights are loaded back into the model at the end, so
        any evaluation after ``fit`` uses the selected model, not the last one.

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
        start: float = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
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
            if self.early_stopping.should_stop:
                logger.info("Early stopping at epoch %d (best: %d).", epoch, best_epoch)
                break

        self._restore_best()
        self.writer.close()

        history["best_epoch"] = best_epoch
        history["train_seconds"] = time.perf_counter() - start
        return history

    def _restore_best(self) -> None:
        """Load the best checkpoint back into the in-memory model."""
        payload: dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(payload["state_dict"])
