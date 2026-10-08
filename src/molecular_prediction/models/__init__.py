"""Model registry."""

import torch.nn as nn

from molecular_prediction.config import MODEL_NAMES, Config
from molecular_prediction.models.base import MolecularGNN
from molecular_prediction.models.egnn import EGNN, EGNNStatic
from molecular_prediction.models.gin import GIN, GINDist, GINEdge

__all__ = [
    "EGNN",
    "GIN",
    "EGNNStatic",
    "GINDist",
    "GINEdge",
    "MolecularGNN",
    "build_model",
    "count_parameters",
]


def build_model(name: str, config: Config) -> MolecularGNN:
    """Instantiate an untrained model with the hyperparameters of ``config``."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")

    hidden_dim: int = config.model.hidden_dims[name]
    num_layers: int = config.model.num_layers
    num_targets: int = len(config.data.target_indices)

    if name == "GIN":
        return GIN(hidden_dim, num_layers, num_targets)
    if name == "GINEdge":
        return GINEdge(hidden_dim, num_layers, num_targets)

    geometric: dict[str, type[MolecularGNN]] = {
        "GINDist": GINDist,
        "EGNNStatic": EGNNStatic,
        "EGNN": EGNN,
    }
    return geometric[name](
        hidden_dim,
        num_layers,
        num_targets,
        num_rbf=config.model.num_rbf,
        rbf_cutoff=config.model.rbf_cutoff,
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
