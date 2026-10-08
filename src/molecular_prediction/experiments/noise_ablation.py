"""Experiment 2: robustness to coordinate noise at inference time.

Trained models are evaluated on the test set after adding isotropic Gaussian
noise of standard deviation sigma to every atomic coordinate. Nothing is
retrained and the bond graph is untouched: the models see the right molecule
with wrong geometry, which isolates the geometric channel.

Design choices:
* GIN and GINEdge never read coordinates, so their curves must be flat. This is
  a built-in sanity check of the pipeline (a slope would reveal a leak).
* The noise of realisation r at level sigma is drawn from a generator seeded
  with (noise_seed, sigma index, r), identical for every model and seed, so the
  comparison between models is paired.
* Each sigma is averaged over several noise realisations; the reported spread
  is across training seeds.
"""

import logging
from typing import Any

import numpy as np
import torch

from molecular_prediction.config import TARGET_NAMES, Config
from molecular_prediction.data.dataset import QM9Splits, load_splits
from molecular_prediction.experiments import plotting
from molecular_prediction.experiments.evaluation import available_checkpoints, predict
from molecular_prediction.experiments.main_comparison import mean_std
from molecular_prediction.training.checkpoint import load_checkpoint
from molecular_prediction.training.utils import save_json

logger: logging.Logger = logging.getLogger(__name__)


def noise_generator(base_seed: int, sigma_index: int, realisation: int) -> torch.Generator:
    """Deterministic generator shared by all models for a given (sigma, realisation)."""
    return torch.Generator().manual_seed(base_seed + 1_000 * sigma_index + realisation)


def evaluate_under_noise(
    config: Config,
    model_name: str,
    splits: QM9Splits,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Per-target test MAE of every seed at every sigma.

    Returns:
        ``{seed: array[num_sigmas, num_targets]}`` in normalised units, each
        entry averaged over the noise realisations.
    """
    noise_cfg = config.noise
    per_seed: dict[int, np.ndarray] = {}

    for seed, path in available_checkpoints(config, model_name).items():
        model, _, _ = load_checkpoint(path, device)
        rows: list[np.ndarray] = []
        for sigma_index, sigma in enumerate(noise_cfg.sigmas):
            realisations: int = 1 if sigma == 0.0 else noise_cfg.num_realisations
            maes: list[np.ndarray] = []
            for r in range(realisations):
                generator = noise_generator(noise_cfg.noise_seed, sigma_index, r)
                preds, targets = predict(model, splits.test, config, device, sigma, generator)
                maes.append((preds - targets).abs().mean(dim=0).numpy())
            rows.append(np.mean(maes, axis=0))
            logger.info("%s seed %d sigma %.2f: MAE %.4f", model_name, seed, sigma, rows[-1].mean())
        per_seed[seed] = np.stack(rows)

    return per_seed


def run_noise_ablation(
    config: Config, model_names: list[str], device: torch.device
) -> dict[str, Any]:
    """Run the sweep for every model, save a JSON summary and plot it.

    Returns:
        ``{"sigmas": [...], "models": {name: {...}}}`` where each model holds,
        per sigma, mean +- std over seeds of the combined MAE, of the
        per-target MAE and of the relative degradation MAE(sigma) / MAE(0).
    """
    splits: QM9Splits = load_splits(config)
    sigmas: list[float] = list(config.noise.sigmas)
    summary: dict[str, Any] = {"sigmas": sigmas, "models": {}}

    for model_name in model_names:
        per_seed: dict[int, np.ndarray] = evaluate_under_noise(config, model_name, splits, device)
        stacked: np.ndarray = np.stack(list(per_seed.values()))  # [seeds, sigmas, targets]
        combined: np.ndarray = stacked.mean(axis=2)  # [seeds, sigmas]
        relative: np.ndarray = combined / combined[:, :1]

        summary["models"][model_name] = {
            "seeds": list(per_seed.keys()),
            "combined": [mean_std(combined[:, s].tolist()) for s in range(len(sigmas))],
            "relative": [mean_std(relative[:, s].tolist()) for s in range(len(sigmas))],
            "per_target": {
                t: [mean_std(stacked[:, s, i].tolist()) for s in range(len(sigmas))]
                for i, t in enumerate(TARGET_NAMES)
            },
        }

    save_json(summary, config.paths.results_dir / "noise_ablation.json")
    plotting.plot_noise_ablation(summary, config.paths.figures_dir)
    return summary
