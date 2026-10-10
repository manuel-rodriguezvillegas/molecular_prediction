"""Command-line interface.

Examples:
    python main.py params
    python main.py data                 # download and process QM9 once
    python main.py train --models GIN EGNN --seeds 0 1 2 --device cuda
    python main.py train --models EGNN+noise0.05    # train with coordinate-noise augmentation
    python main.py compare
    python main.py noise
    python main.py curvature
    python main.py curvature --heavy-atoms   # curvature of the hydrogen-free skeleton
    python main.py plots                     # redraw every figure from results/*.json
    python main.py all --smoke          # 2-epoch end-to-end check on a tiny subset
"""

import argparse
import logging
from pathlib import Path

from molecular_prediction.config import MODEL_NAMES, Config, parse_run_label
from molecular_prediction.data.dataset import load_splits
from molecular_prediction.experiments import plotting
from molecular_prediction.experiments.curvature_analysis import run_curvature_analysis
from molecular_prediction.experiments.main_comparison import (
    aggregate_comparison,
    discover_runs,
    train_all,
)
from molecular_prediction.experiments.noise_ablation import run_noise_ablation
from molecular_prediction.models import build_model, count_parameters
from molecular_prediction.training.utils import load_json, resolve_device, save_json

COMMANDS: tuple[str, ...] = (
    "params",
    "data",
    "train",
    "compare",
    "noise",
    "curvature",
    "plots",
    "all",
)


def _run_label(value: str) -> str:
    """argparse type: accept only well-formed run labels."""
    try:
        parse_run_label(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-aware GNNs for molecular property prediction on QM9."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--models",
        nargs="+",
        type=_run_label,
        metavar="RUN",
        help=(
            f"Architectures from {', '.join(MODEL_NAMES)}, optionally trained with "
            "coordinate noise, e.g. EGNN+noise0.05. Default: the five architectures for "
            "training, every finished run for the analyses."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="Training seeds (default: config).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker processes.")
    parser.add_argument("--data-dir", type=Path, help="Where QM9 is (or will be) stored.")
    parser.add_argument("--output-dir", type=Path, help="Checkpoints, logs, caches.")
    parser.add_argument("--results-dir", type=Path, help="JSON results and figures.")
    parser.add_argument("--overwrite", action="store_true", help="Retrain finished runs.")
    parser.add_argument(
        "--heavy-atoms",
        action="store_true",
        help="Curvature analysis on the heavy-atom graph (hydrogens removed).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny split, 2 epochs, writes only to outputs_smoke/: checks the whole pipeline.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Default configuration with the command-line overrides applied."""
    config: Config = Config()
    if args.smoke:
        config.data.split_sizes = (1_000, 200)
        config.data.test_size = 300
        config.training.epochs = 2
        config.noise.num_realisations = 2
        config.paths.output_dir = Path("outputs_smoke")
        config.paths.results_dir = Path("outputs_smoke/results")
    if args.seeds is not None:
        config.training.seeds = tuple(args.seeds)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.data_dir is not None:
        config.paths.data_dir = args.data_dir
    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir
    if args.results_dir is not None:
        config.paths.results_dir = args.results_dir
    config.curvature.heavy_atoms_only = args.heavy_atoms
    return config


def print_parameter_counts(config: Config, model_names: list[str]) -> None:
    for name in model_names:
        architecture: str = parse_run_label(name)[0]
        model = build_model(architecture, config)
        print(
            f"{name:<18} hidden={config.model.hidden_dims[architecture]:>4}  "
            f"parameters={count_parameters(model):>9,}"
        )


def redraw_figures(config: Config, model_names: list[str]) -> None:
    """Regenerate the figures from the saved JSON results, without checkpoints.

    The curvature histogram needs per-molecule data from the cache, so it is
    only redrawn by the ``curvature`` command itself.
    """
    results: Path = config.paths.results_dir
    figures: Path = config.paths.figures_dir
    aggregate_comparison(config, model_names)
    if (results / "noise_ablation.json").exists():
        noise_summary = load_json(results / "noise_ablation.json")
        plotting.plot_noise_ablation(noise_summary, figures)
        plotting.plot_noise_summary(noise_summary, figures)
    for suffix in ("", "_heavy"):
        path: Path = results / f"curvature_analysis{suffix}.json"
        if path.exists():
            summary = load_json(path)
            plotting.plot_mae_by_quantile(summary, figures, suffix)
            plotting.plot_relative_improvement(summary, figures, suffix)


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config: Config = build_config(args)
    requested: list[str] | None = args.models

    def analysis_runs() -> list[str]:
        # Evaluated lazily: after `all` has trained, the new runs are included.
        return requested if requested is not None else discover_runs(config)

    if args.command == "params":
        print_parameter_counts(config, requested or list(MODEL_NAMES))
        return

    if args.command == "plots":
        redraw_figures(config, analysis_runs())
        return

    if args.command == "data":
        splits = load_splits(config)
        print(f"QM9 ready: {len(splits.train)} / {len(splits.val)} / {len(splits.test)}")
        return

    device = resolve_device(args.device)
    logging.getLogger(__name__).info("Command %s on %s", args.command, device)
    save_json(config.to_dict(), config.paths.results_dir / "config.json")

    if args.command in ("train", "all"):
        train_all(config, requested or list(MODEL_NAMES), device, overwrite=args.overwrite)
    if args.command in ("compare", "all"):
        aggregate_comparison(config, analysis_runs())
    if args.command in ("noise", "all"):
        run_noise_ablation(config, analysis_runs(), device)
    if args.command in ("curvature", "all"):
        run_curvature_analysis(config, analysis_runs(), device)


if __name__ == "__main__":
    main()
