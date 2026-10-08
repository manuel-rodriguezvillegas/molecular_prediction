"""Graph Isomorphism Network family (Xu et al., ICLR 2019).

Three models share the exact same trunk and differ only in what they know about
each bond:

* ``GIN``     - nothing: pure covalent topology (GINConv).
* ``GINEdge`` - the bond type (single / double / triple / aromatic) (GINEConv).
* ``GINDist`` - the bond type plus the bond length, RBF-expanded (GINEConv).

GIN -> GINEdge isolates the value of bond types; GINEdge -> GINDist isolates the
value of interatomic distances. None of them uses angles.
"""

from typing import ClassVar

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GINEConv

from molecular_prediction.data.dataset import EDGE_FEATURE_DIM
from molecular_prediction.models.base import MolecularGNN
from molecular_prediction.models.layers import RadialBasis, edge_vectors, mlp


def _gin_mlp(hidden_dim: int) -> nn.Sequential:
    """Update MLP of a GIN layer, with batch norm as in the original paper."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GIN(MolecularGNN):
    """Topology-only baseline: atom features and bond connectivity, no geometry.

    It never reads ``data.pos``, so its predictions are trivially invariant to
    any transformation of the coordinates, including random noise.
    """

    uses_coordinates: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, num_layers: int, num_targets: int) -> None:
        super().__init__(hidden_dim, num_targets)
        self.convs: nn.ModuleList = nn.ModuleList(
            GINConv(_gin_mlp(hidden_dim), train_eps=False) for _ in range(num_layers)
        )

    def encode(self, data: Data) -> torch.Tensor:
        h: torch.Tensor = self.atom_embedding(data.x)
        for conv in self.convs:
            h = h + conv(h, data.edge_index)  # residual connection
        return h


class GINEdge(MolecularGNN):
    """GIN with the bond type as edge feature, still without geometry."""

    uses_coordinates: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, num_layers: int, num_targets: int) -> None:
        super().__init__(hidden_dim, num_targets)
        self.edge_encoder: nn.Sequential = mlp(EDGE_FEATURE_DIM, hidden_dim, hidden_dim)
        self.convs: nn.ModuleList = nn.ModuleList(
            GINEConv(_gin_mlp(hidden_dim), train_eps=False) for _ in range(num_layers)
        )

    def edge_features(self, data: Data) -> torch.Tensor:
        """Raw per-edge input of the edge encoder."""
        return data.edge_attr

    def encode(self, data: Data) -> torch.Tensor:
        h: torch.Tensor = self.atom_embedding(data.x)
        e: torch.Tensor = self.edge_encoder(self.edge_features(data))
        for conv in self.convs:
            h = h + conv(h, data.edge_index, e)
        return h


class GINDist(GINEdge):
    """GINEdge plus RBF-expanded bond lengths: E(3)-invariant, no angles.

    Distances are unchanged by rotations, translations and reflections, so the
    model is invariant to E(3) by construction. What it cannot see is angular
    information: bond lengths alone do not determine bond or dihedral angles.
    """

    uses_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_targets: int,
        num_rbf: int,
        rbf_cutoff: float,
    ) -> None:
        super().__init__(hidden_dim, num_layers, num_targets)
        self.rbf: RadialBasis = RadialBasis(num_rbf, rbf_cutoff)
        # Wider input than GINEdge: bond one-hot concatenated with the RBF features.
        self.edge_encoder = mlp(EDGE_FEATURE_DIM + num_rbf, hidden_dim, hidden_dim)

    def edge_features(self, data: Data) -> torch.Tensor:
        _, dist = edge_vectors(data.pos, data.edge_index)
        return torch.cat([data.edge_attr, self.rbf(dist)], dim=-1)
