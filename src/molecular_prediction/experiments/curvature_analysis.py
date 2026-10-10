"""Experiment 3: does geometry help more on molecules with topological bottlenecks?

Hypothesis (over-squashing): message passing on the bond graph struggles where
the graph has bridge-like edges (very negative Ollivier-Ricci curvature), so the
advantage of geometric models over GIN should grow with bottleneck severity.

Procedure:
1. Minimum and mean edge curvature of every test molecule (cached on disk).
2. Molecules are ranked by bottleneck severity (min curvature, ties broken by
   mean curvature) and cut into equal-sized quantiles: Q0 = least bottlenecked.
3. Test MAE per quantile for every model (mean +- std over seeds), and the
   relative improvement of each model over GIN with bootstrap 95% CIs.
4. Size control. Larger molecules have both more negative curvature and harder
   targets, so a trend across quantiles could be a size effect. We regress the
   per-molecule improvement over GIN on standardised min curvature, with and
   without the number of atoms as a covariate; if the curvature coefficient
   vanishes once size is included, the trend was size, not over-squashing.

All errors are in normalised units, so the combined score weighs every target
equally (averaging physical units would be dominated by alpha).
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Dataset

from molecular_prediction.config import TARGET_NAMES, Config
from molecular_prediction.data.dataset import HYDROGEN_FEATURE, QM9Splits, load_splits
from molecular_prediction.experiments import plotting
from molecular_prediction.experiments.curvature import CurvatureSummary, summarise_molecule
from molecular_prediction.experiments.evaluation import (
    available_checkpoints,
    collate_batches,
    predict,
)
from molecular_prediction.experiments.main_comparison import mean_std
from molecular_prediction.training.checkpoint import load_checkpoint
from molecular_prediction.training.utils import save_json

logger: logging.Logger = logging.getLogger(__name__)

REFERENCE_MODEL: str = "GIN"
NUM_BOOTSTRAP: int = 1_000


def compute_test_curvatures(config: Config, test_dataset: Dataset) -> dict[str, np.ndarray]:
    """Curvature summary and size of every test molecule, cached as ``.npz``.

    Returns:
        Arrays of length num_test: ``min_curvature``, ``mean_curvature``,
        ``num_edges``, ``num_atoms`` and ``num_heavy_atoms``.
    """
    cache: Path = config.paths.cache_dir / (
        f"curvature_split{config.data.split_seed}_n{len(test_dataset)}"
        f"_alpha{config.curvature.alpha}.npz"
    )
    if cache.exists():
        logger.info("Loading cached curvatures from %s", cache)
        return dict(np.load(cache))

    graphs: list[tuple[np.ndarray, int, float]] = []
    num_heavy: list[int] = []
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        graphs.append((data.edge_index.numpy(), data.num_nodes, config.curvature.alpha))
        num_heavy.append(int((data.x[:, HYDROGEN_FEATURE] == 0).sum()))

    logger.info("Computing Ollivier-Ricci curvature for %d molecules...", len(graphs))
    with Pool(config.curvature.num_processes) as pool:
        summaries: list[CurvatureSummary] = pool.starmap(summarise_molecule, graphs, chunksize=256)

    result: dict[str, np.ndarray] = {
        "min_curvature": np.array([s.min_curvature for s in summaries]),
        "mean_curvature": np.array([s.mean_curvature for s in summaries]),
        "num_edges": np.array([s.num_edges for s in summaries]),
        "num_atoms": np.array([g[1] for g in graphs]),
        "num_heavy_atoms": np.array(num_heavy),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **result)
    return result


def assign_quantiles(
    min_curvature: np.ndarray, mean_curvature: np.ndarray, num_quantiles: int
) -> np.ndarray:
    """Equal-sized bottleneck quantiles; 0 = least bottlenecked.

    Min curvature takes few distinct values on small molecules, so percentile
    cut-offs would produce very unbalanced bins. Ranking by (min, mean)
    curvature and splitting the ranking into equal chunks keeps the bins
    balanced, and the secondary key resolves most ties meaningfully.
    """
    # lexsort sorts by the last key first: descending min, then descending mean.
    order: np.ndarray = np.lexsort((-mean_curvature, -min_curvature))
    labels: np.ndarray = np.empty(len(order), dtype=int)
    for quantile, chunk in enumerate(np.array_split(order, num_quantiles)):
        labels[chunk] = quantile
    return labels


def per_molecule_errors(
    config: Config, model_name: str, test_batches: list[Batch], device: torch.device
) -> np.ndarray:
    """Absolute errors in normalised units, shape [num_seeds, num_test, num_targets]."""
    errors: list[np.ndarray] = []
    for path in available_checkpoints(config, model_name).values():
        model, _, _ = load_checkpoint(path, device)
        preds, targets = predict(model, test_batches, device)
        errors.append((preds - targets).abs().numpy())
    return np.stack(errors)


def bootstrap_relative_improvement(
    reference: np.ndarray, candidate: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Relative MAE improvement of ``candidate`` over ``reference`` with a 95% CI.

    Both inputs are per-molecule errors of the same molecules, so molecules are
    resampled jointly (paired bootstrap).
    """
    point: float = 1.0 - candidate.mean() / reference.mean()
    indices: np.ndarray = rng.integers(0, len(reference), size=(NUM_BOOTSTRAP, len(reference)))
    samples: np.ndarray = 1.0 - candidate[indices].mean(axis=1) / reference[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(point), "ci_low": float(low), "ci_high": float(high)}


def ols(design: np.ndarray, response: np.ndarray) -> dict[str, list[float]]:
    """Ordinary least squares coefficients and classical standard errors."""
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals: np.ndarray = response - design @ coefficients
    dof: int = len(response) - design.shape[1]
    variance: float = float(residuals @ residuals) / dof
    covariance: np.ndarray = variance * np.linalg.inv(design.T @ design)
    return {"coef": coefficients.tolist(), "se": np.sqrt(np.diag(covariance)).tolist()}


def size_controlled_regression(
    improvement: np.ndarray, min_curvature: np.ndarray, num_atoms: np.ndarray
) -> dict[str, Any]:
    """Regress per-molecule improvement on curvature, with and without size.

    Both covariates are standardised and the curvature is negated, so a
    positive coefficient means "more bottlenecked -> larger improvement".
    """

    def zscore(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std()

    severity: np.ndarray = zscore(-min_curvature)
    size: np.ndarray = zscore(num_atoms.astype(float))
    ones: np.ndarray = np.ones_like(severity)

    curvature_only = ols(np.column_stack([ones, severity]), improvement)
    with_size = ols(np.column_stack([ones, severity, size]), improvement)
    return {
        "severity_coef": curvature_only["coef"][1],
        "severity_se": curvature_only["se"][1],
        "severity_coef_given_size": with_size["coef"][1],
        "severity_se_given_size": with_size["se"][1],
        "size_coef": with_size["coef"][2],
        "size_se": with_size["se"][2],
    }


def run_curvature_analysis(
    config: Config, model_names: list[str], device: torch.device
) -> dict[str, Any]:
    """Run the full analysis, save a JSON summary and plot it."""
    splits: QM9Splits = load_splits(config)
    molecules: dict[str, np.ndarray] = compute_test_curvatures(config, splits.test)
    num_quantiles: int = config.curvature.num_quantiles
    quantiles: np.ndarray = assign_quantiles(
        molecules["min_curvature"], molecules["mean_curvature"], num_quantiles
    )
    std: np.ndarray = splits.stats.std.numpy()

    test_batches: list[Batch] = collate_batches(splits.test, config.training.batch_size)
    errors: dict[str, np.ndarray] = {
        name: per_molecule_errors(config, name, test_batches, device) for name in model_names
    }

    summary: dict[str, Any] = {
        "quantile_sizes": np.bincount(quantiles, minlength=num_quantiles).tolist(),
        "quantile_min_curvature": [
            mean_std(molecules["min_curvature"][quantiles == q].tolist())
            for q in range(num_quantiles)
        ],
        "quantile_num_atoms": [
            mean_std(molecules["num_atoms"][quantiles == q].tolist()) for q in range(num_quantiles)
        ],
        "corr_min_curvature_num_atoms": float(
            np.corrcoef(molecules["min_curvature"], molecules["num_atoms"])[0, 1]
        ),
        "mae_by_quantile": {},
        "relative_improvement": {},
        "regression": {},
    }

    for name, err in errors.items():
        by_quantile: dict[str, Any] = {"combined": [], "per_target_physical": {}}
        for q in range(num_quantiles):
            mask: np.ndarray = quantiles == q
            per_seed_combined: np.ndarray = err[:, mask, :].mean(axis=(1, 2))
            by_quantile["combined"].append(mean_std(per_seed_combined.tolist()))
        for i, target in enumerate(TARGET_NAMES):
            by_quantile["per_target_physical"][target] = [
                mean_std((err[:, quantiles == q, i].mean(axis=1) * std[i]).tolist())
                for q in range(num_quantiles)
            ]
        summary["mae_by_quantile"][name] = by_quantile

    if REFERENCE_MODEL in errors:
        rng: np.random.Generator = np.random.default_rng(0)
        # Seed-averaged per-molecule error, combined over targets: [num_test].
        reference: np.ndarray = errors[REFERENCE_MODEL].mean(axis=(0, 2))
        for name, err in errors.items():
            if name == REFERENCE_MODEL:
                continue
            candidate: np.ndarray = err.mean(axis=(0, 2))
            summary["relative_improvement"][name] = [
                bootstrap_relative_improvement(
                    reference[quantiles == q], candidate[quantiles == q], rng
                )
                for q in range(num_quantiles)
            ]
            summary["regression"][name] = size_controlled_regression(
                reference - candidate, molecules["min_curvature"], molecules["num_atoms"]
            )

    save_json(summary, config.paths.results_dir / "curvature_analysis.json")
    plotting.plot_curvature_distribution(molecules, config.paths.figures_dir)
    plotting.plot_mae_by_quantile(summary, config.paths.figures_dir)
    if summary["relative_improvement"]:
        plotting.plot_relative_improvement(summary, config.paths.figures_dir)
    _log_summary(summary)
    return summary


def _log_summary(summary: dict[str, Any]) -> None:
    lines: list[str] = [f"Quantile sizes: {summary['quantile_sizes']}"]
    lines.append(f"corr(min curvature, num atoms) = {summary['corr_min_curvature_num_atoms']:.3f}")
    for name, reg in summary["regression"].items():
        lines.append(
            f"{name} vs {REFERENCE_MODEL}: severity coef {reg['severity_coef']:+.4f} "
            f"(se {reg['severity_se']:.4f}); given size {reg['severity_coef_given_size']:+.4f} "
            f"(se {reg['severity_se_given_size']:.4f})"
        )
    logger.info("Curvature analysis\n%s", "\n".join(lines))
