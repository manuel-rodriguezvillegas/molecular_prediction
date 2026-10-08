"""Common interface of every model: encode atoms, then read out properties."""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn
from torch_geometric.data import Data

from molecular_prediction.data.dataset import NODE_FEATURE_DIM
from molecular_prediction.models.layers import MultiTaskHead


class MolecularGNN(nn.Module, ABC):
    """Template for all models.

    Subclasses only implement :meth:`encode`, which turns a (batched) molecular
    graph into per-atom embeddings. The atom embedding and the readout head are
    shared code, so any difference between models comes from the encoder,
    which is precisely the variable under study.

    Args:
        hidden_dim: Width of the atom embeddings.
        num_targets: Number of properties predicted jointly.
    """

    #: Whether the model reads ``data.pos``. Used by the tests and the noise
    #: ablation: a model that does not use coordinates must be unaffected by them.
    uses_coordinates: ClassVar[bool]

    def __init__(self, hidden_dim: int, num_targets: int) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.atom_embedding: nn.Linear = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.head: MultiTaskHead = MultiTaskHead(hidden_dim, num_targets)

    @abstractmethod
    def encode(self, data: Data) -> torch.Tensor:
        """Return per-atom embeddings of shape [num_nodes, hidden_dim]."""

    def forward(self, data: Data) -> torch.Tensor:
        """Predict normalised targets, shape [num_graphs, num_targets]."""
        h: torch.Tensor = self.encode(data)
        batch: torch.Tensor = (
            data.batch
            if data.batch is not None
            else torch.zeros(data.num_nodes, dtype=torch.long, device=h.device)
        )
        return self.head(h, batch)
