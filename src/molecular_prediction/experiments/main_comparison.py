"""Experiment 1: train every model with several seeds and compare test MAE."""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molecular_prediction.config import (
    TARGET_NAMES,
    TARGET_UNITS,
    Config,
    parse_run_label,
    run_sort_key,
)
from molecular_prediction.data.dataset import QM9Splits, load_splits
from molecular_prediction.experiments import plotting
from molecular_prediction.models import MolecularGNN, build_model, count_parameters
from molecular_prediction.training.trainer import Trainer, make_loader
from molecular_prediction.training.utils import load_json, run_dir, save_json, seed_everything

logger: logging.Logger = logging.getLogger(__name__)

METRICS_NAME: str = "metrics.json"
HISTORY_NAME: str = "history.json"


def discover_runs(config: Config) -> list[str]:
    """Labels of every run with at least one finished seed, in ladder order."""
    root: Path = config.paths.run_results_dir
    if not root.exists():
        return []
    labels: list[str] = [
        directory.name
        for directory in root.iterdir()
        if any(directory.glob(f"seed_*/{METRICS_NAME}"))
    ]
    return sorted(labels, key=run_sort_key)


def mean_std(values: list[float]) -> dict[str, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    array: np.ndarray = np.asarray(values, dtype=float)
    std: float = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std}


def train_run(
    config: Config,
    run_label: str,
    seed: int,
    splits: QM9Splits,
    device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    """Train one (run, seed) pair, evaluate it on test and persist everything.

    ``run_label`` is an architecture name, optionally with training noise
    (e.g. ``"EGNN+noise0.05"``). An interrupted run is continued from its last
    completed epoch unless ``resume`` is False.

    The checkpoint and TensorBoard logs go to ``outputs/runs/<run>/seed_<seed>/``
    (git-ignored); ``metrics.json`` and ``history.json`` go to
    ``results/runs/<run>/seed_<seed>/`` (committed).
    """
    architecture, train_noise_sigma = parse_run_label(run_label)
    run_config: Config = copy.deepcopy(config)
    run_config.training.train_noise_sigma = train_noise_sigma

    output_dir: Path = run_dir(config.paths.runs_dir, run_label, seed)
    results_dir: Path = run_dir(config.paths.run_results_dir, run_label, seed)
    seed_everything(seed)
    model: MolecularGNN = build_model(architecture, run_config)
    num_parameters: int = count_parameters(model)
    logger.info("Training %s (seed %d, %d parameters)", run_label, seed, num_parameters)

    trainer: Trainer = Trainer(
        model,
        run_label,
        splits.train,
        splits.val,
        splits.stats,
        run_config,
        device,
        output_dir,
        seed,
        resume=resume,
    )
    history: dict[str, Any] = trainer.fit()

    test_loader = make_loader(
        splits.test, config.training.batch_size, False, config.data.num_workers
    )
    test_mae: torch.Tensor = trainer.evaluate(test_loader)  # best checkpoint, normalised
    # MAE is linear in the targets, so physical-unit MAE is normalised MAE * std.
    test_mae_physical: torch.Tensor = test_mae * splits.stats.std

    metrics: dict[str, Any] = {
        "model_name": run_label,
        "architecture": architecture,
        "train_noise_sigma": train_noise_sigma,
        "seed": seed,
        "num_parameters": num_parameters,
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["val_mae"]),
        "train_seconds": history["train_seconds"],
        "best_val_mae": min(history["val_mae"]),
        "test_mae": float(test_mae.mean()),
        "test_mae_per_target": dict(zip(TARGET_NAMES, test_mae.tolist(), strict=True)),
        "test_mae_physical": dict(zip(TARGET_NAMES, test_mae_physical.tolist(), strict=True)),
    }
    save_json(history, results_dir / HISTORY_NAME)
    save_json(metrics, results_dir / METRICS_NAME)
    trainer.discard_resume_state()  # only once the results are safely on disk
    logger.info("%s seed %d: test MAE %.4f", run_label, seed, metrics["test_mae"])
    return metrics


def train_all(
    config: Config,
    model_names: list[str],
    device: torch.device,
    overwrite: bool = False,
) -> None:
    """Train every requested (model, seed) pair.

    Finished runs (with ``metrics.json``) are skipped and interrupted ones are
    resumed; ``overwrite`` retrains everything from scratch.
    """
    splits: QM9Splits = load_splits(config)
    for model_name in model_names:
        for seed in config.training.seeds:
            done: Path = run_dir(config.paths.run_results_dir, model_name, seed) / METRICS_NAME
            if done.exists() and not overwrite:
                logger.info("Skipping %s seed %d: already trained.", model_name, seed)
                continue
            train_run(config, model_name, seed, splits, device, resume=not overwrite)


def aggregate_comparison(config: Config, model_names: list[str]) -> dict[str, Any]:
    """Collect the finished runs, summarise them across seeds and plot.

    Returns:
        Summary with, for each model, mean +- std over seeds of the test MAE in
        normalised units (combined and per target) and in physical units.
    """
    summary: dict[str, Any] = {
        "units": dict(zip(TARGET_NAMES, TARGET_UNITS, strict=True)),
        "models": {},
    }
    histories: dict[str, list[dict[str, Any]]] = {}

    for model_name in model_names:
        runs: list[dict[str, Any]] = []
        histories[model_name] = []
        for seed in config.training.seeds:
            directory: Path = run_dir(config.paths.run_results_dir, model_name, seed)
            if not (directory / METRICS_NAME).exists():
                logger.warning("Missing run %s seed %d, excluded.", model_name, seed)
                continue
            runs.append(load_json(directory / METRICS_NAME))
            histories[model_name].append(load_json(directory / HISTORY_NAME))
        if not runs:
            raise FileNotFoundError(f"No finished run for {model_name}.")

        summary["models"][model_name] = {
            "seeds": [run["seed"] for run in runs],
            "num_parameters": runs[0]["num_parameters"],
            "best_epoch": [run["best_epoch"] for run in runs],
            "train_hours": mean_std([run["train_seconds"] / 3600 for run in runs]),
            "test_mae": mean_std([run["test_mae"] for run in runs]),
            "test_mae_per_target": {
                t: mean_std([run["test_mae_per_target"][t] for run in runs]) for t in TARGET_NAMES
            },
            "test_mae_physical": {
                t: mean_std([run["test_mae_physical"][t] for run in runs]) for t in TARGET_NAMES
            },
        }

    save_json(summary, config.paths.results_dir / "comparison.json")
    print_comparison_table(summary)
    plotting.plot_training_curves(histories, config.paths.figures_dir)
    plotting.plot_test_mae(summary, config.paths.figures_dir)
    return summary


def print_comparison_table(summary: dict[str, Any]) -> None:
    """Log a mean +- std table: combined (normalised) and per target (physical)."""
    units: dict[str, str] = summary["units"]
    header: str = f"{'Model':<11} {'Params':>8} {'Combined':>15}" + "".join(
        f" {f'{t} [{units[t]}]':>19}" for t in TARGET_NAMES
    )
    lines: list[str] = [header, "-" * len(header)]
    for name, result in summary["models"].items():
        row: str = f"{name:<11} {result['num_parameters']:>8,}"
        row += f" {_fmt(result['test_mae']):>15}"
        for t in TARGET_NAMES:
            row += f" {_fmt(result['test_mae_physical'][t]):>19}"
        lines.append(row)
    logger.info("Test MAE (mean +- std over seeds)\n%s", "\n".join(lines))


def _fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f}+-{stat['std']:.4f}"
