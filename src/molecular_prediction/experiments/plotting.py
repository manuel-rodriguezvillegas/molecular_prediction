"""Figures of every experiment. Error bars are always +-1 std over training seeds.

Each architecture has a fixed colour; runs trained with coordinate noise
(e.g. "EGNN+noise0.05") reuse it with dashed lines or hatched bars.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the DGX has no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from molecular_prediction.config import TARGET_NAMES, parse_run_label  # noqa: E402

# One fixed colour per model, from no geometry (grey) to full equivariance (red).
MODEL_COLOURS: dict[str, str] = {
    "GIN": "#8C8C8C",
    "GINEdge": "#4C72B0",
    "GINDist": "#55A868",
    "EGNNStatic": "#DD8452",
    "EGNN": "#C44E52",
}


def colour(run: str) -> str | None:
    """Colour of a run's architecture (noise-trained variants share it)."""
    return MODEL_COLOURS.get(parse_run_label(run)[0])


def is_noise_trained(run: str) -> bool:
    return parse_run_label(run)[1] > 0.0


def line_style(run: str) -> str:
    """Dashed lines for noise-trained runs, solid otherwise."""
    return "--" if is_noise_trained(run) else "-"


def hatch(run: str) -> str | None:
    """Hatched bars for noise-trained runs."""
    return "//" if is_noise_trained(run) else None


def _save(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)


def _grouped_bars(
    ax: plt.Axes,
    models: list[str],
    groups: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    """Bars grouped by ``groups`` (x axis), one colour per model. Arrays: [models, groups]."""
    width: float = 0.8 / len(models)
    x: np.ndarray = np.arange(len(groups))
    for i, model in enumerate(models):
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            means[i],
            width,
            yerr=stds[i],
            capsize=2,
            label=model,
            color=colour(model),
            hatch=hatch(model),
            edgecolor="white",
        )
    ax.set_xticks(x, groups)
    ax.grid(True, axis="y", alpha=0.3)


def plot_training_curves(histories: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Validation MAE per epoch (combined and per target), mean +- std over seeds."""
    panels: list[str] = ["combined", *TARGET_NAMES]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharex=True)
    for model, runs in histories.items():
        for ax, panel in zip(axes, panels, strict=True):
            curves: list[list[float]] = [
                run["val_mae"] if panel == "combined" else run["val_mae_per_target"][panel]
                for run in runs
            ]
            length: int = min(len(c) for c in curves)  # early stopping may differ per seed
            stacked: np.ndarray = np.array([c[:length] for c in curves])
            epochs: np.ndarray = np.arange(1, length + 1)
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
            ax.plot(epochs, mean, label=model, color=colour(model), linestyle=line_style(model))
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=colour(model))
    for ax, panel in zip(axes, panels, strict=True):
        ax.set_title(f"Validation MAE - {panel}")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MAE (normalised units)")
    axes[0].legend()
    _save(fig, output_dir, "training_curves.png")


def plot_test_mae(summary: dict[str, Any], output_dir: Path) -> None:
    """Test MAE per model: combined (normalised) and each target (physical units)."""
    models: list[str] = list(summary["models"])
    panels: list[str] = ["combined", *TARGET_NAMES]
    # 2 x 3 grid (last cell holds the legend): readable on a printed page.
    fig, grid = plt.subplots(2, 3, figsize=(13, 8))
    axes: list[plt.Axes] = list(grid.flat[: len(panels)])
    for ax, panel in zip(axes, panels, strict=True):
        stats: list[dict[str, float]] = [
            summary["models"][m]["test_mae"]
            if panel == "combined"
            else summary["models"][m]["test_mae_physical"][panel]
            for m in models
        ]
        bars = ax.bar(
            models,
            [s["mean"] for s in stats],
            yerr=[s["std"] for s in stats],
            capsize=3,
            color=[colour(m) for m in models],
            edgecolor="white",
            label=models,
        )
        for bar, model in zip(bars, models, strict=True):
            bar.set_hatch(hatch(model))
        unit: str = "normalised" if panel == "combined" else summary["units"][panel]
        ax.set_title(f"{panel} [{unit}]")
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Test MAE")
    legend_ax: plt.Axes = grid.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)
    _save(fig, output_dir, "test_mae.png")


def plot_noise_ablation(summary: dict[str, Any], output_dir: Path) -> None:
    """Absolute and relative MAE vs coordinate noise, plus one panel per target."""
    sigmas: np.ndarray = np.array(summary["sigmas"])
    panels: list[tuple[str, str]] = [
        ("combined", "MAE (normalised)"),
        ("relative", "MAE(sigma) / MAE(0)"),
        *[(t, f"MAE {t} (normalised)") for t in TARGET_NAMES],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, ylabel) in zip(axes.flat, panels, strict=True):
        for model, result in summary["models"].items():
            series: list[dict[str, float]] = (
                result[key] if key in ("combined", "relative") else result["per_target"][key]
            )
            ax.errorbar(
                sigmas,
                [s["mean"] for s in series],
                yerr=[s["std"] for s in series],
                marker="o",
                capsize=3,
                label=model,
                color=colour(model),
                linestyle=line_style(model),
            )
        ax.set_xlabel("Noise sigma (Angstrom)")
        ax.set_ylabel(ylabel)
        # Errors span four orders of magnitude across models: log scale is required.
        ax.set_yscale("log")
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
    # Curves cross every panel, so the legend goes below the figure.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _save(fig, output_dir, "noise_ablation.png", rect=(0.0, 0.05, 1.0, 1.0))


def plot_noise_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Single-panel version of the noise ablation: combined MAE vs sigma, every run."""
    sigmas: np.ndarray = np.array(summary["sigmas"])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for model, result in summary["models"].items():
        ax.plot(
            sigmas,
            [s["mean"] for s in result["combined"]],
            marker="o",
            markersize=4,
            label=model,
            color=colour(model),
            linestyle=line_style(model),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Test-time coordinate noise sigma (Angstrom)")
    ax.set_ylabel("Test MAE (normalised, combined)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _save(fig, output_dir, "noise_summary.png")


def plot_curvature_distribution(
    molecules: dict[str, np.ndarray], output_dir: Path, suffix: str = ""
) -> None:
    """Histogram of min curvature and its relation to molecule size (the confounder)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(molecules["min_curvature"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Minimum edge Ollivier-Ricci curvature")
    axes[0].set_ylabel("Molecules")
    graph: str = "heavy-atom graph" if suffix == "_heavy" else "all-atom graph"
    axes[0].set_title(f"Bottleneck severity across test molecules ({graph})")

    sizes: np.ndarray = molecules["num_atoms"]
    unique_sizes: np.ndarray = np.unique(sizes)
    axes[1].boxplot(
        [molecules["min_curvature"][sizes == s] for s in unique_sizes],
        positions=unique_sizes,
        widths=0.6,
        showfliers=False,
    )
    axes[1].set_xlabel("Number of atoms")
    axes[1].set_ylabel("Minimum curvature")
    axes[1].set_title(f"Curvature vs size, the confounder ({graph})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f"curvature_distribution{suffix}.png")


def plot_mae_by_quantile(summary: dict[str, Any], output_dir: Path, suffix: str = "") -> None:
    """MAE per bottleneck quantile: combined (normalised) and per target (physical)."""
    models: list[str] = list(summary["mae_by_quantile"])
    labels: list[str] = [f"Q{q}\nn={n}" for q, n in enumerate(summary["quantile_sizes"])]
    panels: list[str] = ["combined", *TARGET_NAMES]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.5))
    for ax, panel in zip(axes, panels, strict=True):
        series: list[list[dict[str, float]]] = [
            summary["mae_by_quantile"][m]["combined"]
            if panel == "combined"
            else summary["mae_by_quantile"][m]["per_target_physical"][panel]
            for m in models
        ]
        means: np.ndarray = np.array([[s["mean"] for s in row] for row in series])
        stds: np.ndarray = np.array([[s["std"] for s in row] for row in series])
        _grouped_bars(ax, models, labels, means, stds)
        ax.set_title(panel)
        ax.set_xlabel("Bottleneck quantile (Q0 = least)")
    axes[0].set_ylabel("MAE")
    axes[0].legend(fontsize=8)
    _save(fig, output_dir, f"mae_by_quantile{suffix}.png")


def plot_relative_improvement(summary: dict[str, Any], output_dir: Path, suffix: str = "") -> None:
    """Relative improvement over GIN per quantile with paired-bootstrap 95% CIs."""
    improvements: dict[str, list[dict[str, float]]] = summary["relative_improvement"]
    models: list[str] = list(improvements)
    num_quantiles: int = len(summary["quantile_sizes"])
    labels: list[str] = [f"Q{q}" for q in range(num_quantiles)]

    means: np.ndarray = np.array([[100 * s["mean"] for s in improvements[m]] for m in models])
    low: np.ndarray = np.array([[100 * s["ci_low"] for s in improvements[m]] for m in models])
    high: np.ndarray = np.array([[100 * s["ci_high"] for s in improvements[m]] for m in models])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width: float = 0.8 / len(models)
    x: np.ndarray = np.arange(num_quantiles)
    for i, model in enumerate(models):
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            means[i],
            width,
            yerr=[means[i] - low[i], high[i] - means[i]],
            capsize=2,
            label=model,
            color=colour(model),
            hatch=hatch(model),
            edgecolor="white",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Bottleneck quantile (Q0 = least, Q3 = most)")
    ax.set_ylabel("Improvement over GIN (%)")
    ax.set_title("Relative MAE improvement over GIN (combined, 95% bootstrap CI)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, output_dir, f"relative_improvement_by_quantile{suffix}.png")
