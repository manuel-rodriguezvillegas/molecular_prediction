"""Command-line interface.

Examples:
    python main.py params
    python main.py data                 # download and process QM9 once
    python main.py train --models GIN EGNN --seeds 0 1 2 --device cuda
    python main.py compare
    python main.py noise
    python main.py curvature
    python main.py all --smoke          # 2-epoch end-to-end check on a tiny subset
"""

import argparse
import logging
from pathlib import Path

from molecular_prediction.config import MODEL_NAMES, Config
from molecular_prediction.data.dataset import load_splits
from molecular_prediction.experiments.curvature_analysis import run_curvature_analysis
from molecular_prediction.experiments.main_comparison import aggregate_comparison, train_all
from molecular_prediction.experiments.noise_ablation import run_noise_ablation
from molecular_prediction.models import build_model, count_parameters
from molecular_prediction.training.utils import resolve_device, save_json

COMMANDS: tuple[str, ...] = ("params", "data", "train", "compare", "noise", "curvature", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-aware GNNs for molecular property prediction on QM9."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, help="Training seeds (default: config).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker processes.")
    parser.add_argument("--data-dir", type=Path, help="Where QM9 is (or will be) stored.")
    parser.add_argument("--output-dir", type=Path, help="Checkpoints, logs, caches.")
    parser.add_argument("--results-dir", type=Path, help="JSON results and figures.")
    parser.add_argument("--overwrite", action="store_true", help="Retrain finished runs.")
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
    return config


def print_parameter_counts(config: Config, model_names: list[str]) -> None:
    for name in model_names:
        model = build_model(name, config)
        print(
            f"{name:<11} hidden={config.model.hidden_dims[name]:>4}  "
            f"parameters={count_parameters(model):>9,}"
        )


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config: Config = build_config(args)
    models: list[str] = list(args.models)

    if args.command == "params":
        print_parameter_counts(config, models)
        return

    if args.command == "data":
        splits = load_splits(config)
        print(f"QM9 ready: {len(splits.train)} / {len(splits.val)} / {len(splits.test)}")
        return

    device = resolve_device(args.device)
    logging.getLogger(__name__).info("Command %s on %s", args.command, device)
    save_json(config.to_dict(), config.paths.results_dir / "config.json")

    if args.command in ("train", "all"):
        train_all(config, models, device, overwrite=args.overwrite)
    if args.command in ("compare", "all"):
        aggregate_comparison(config, models)
    if args.command in ("noise", "all"):
        run_noise_ablation(config, models, device)
    if args.command in ("curvature", "all"):
        run_curvature_analysis(config, models, device)


if __name__ == "__main__":
    main()
