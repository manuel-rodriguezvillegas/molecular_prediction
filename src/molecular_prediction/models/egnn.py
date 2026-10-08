"""E(n) Equivariant Graph Neural Network (Satorras, Hoogeboom & Welling, ICML 2021).

For an edge j -> i, one layer computes

    m_ij      = phi_e(h_i, h_j, RBF(||x_i - x_j||), a_ij)
    h_i'      = h_i + phi_h(h_i, sum_j m_ij)
    x_i'      = x_i + mean_j  (x_i - x_j) / (||x_i - x_j|| + 1) * phi_x(m_ij)

Messages depend only on invariants, so ``h`` is E(3)-invariant; coordinates move
only along relative-position vectors scaled by invariant scalars, so ``x`` is
E(3)-equivariant. The readout uses ``h`` alone, hence predictions are invariant.

Two variants are exposed:

* ``EGNN``       - coordinates are updated (equation above).
* ``EGNNStatic`` - coordinates are frozen. It is then an invariant MPNN over
  bond lengths, i.e. it sees exactly the same information as ``GINDist``. The
  EGNNStatic -> EGNN comparison isolates the effect of the equivariant
  coordinate update, which is the only way this model can access angular
  information on a bond graph.

Deviations from the paper, all deliberate: the graph is the covalent-bond graph
instead of a fully connected one (same topology for every model, required by
the curvature study); distances are RBF-expanded (same featurisation as
GINDist); predictions are multi-task with per-target heads.
"""

from typing import ClassVar

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from molecular_prediction.data.dataset import EDGE_FEATURE_DIM
from molecular_prediction.models.base import MolecularGNN
from molecular_prediction.models.layers import RadialBasis, edge_vectors, mlp


class EGNNLayer(nn.Module):
    """One E(n)-equivariant message passing layer.

    Messages are computed once and reused for both the feature update and the
    coordinate update, with a single ``target = edge_index[1]`` convention.

    Args:
        hidden_dim: Width of the atom embeddings.
        num_rbf: Size of the radial basis expansion.
        update_coords: Whether this layer moves the atoms.
    """

    def __init__(self, hidden_dim: int, num_rbf: int, update_coords: bool) -> None:
        super().__init__()
        self.update_coords: bool = update_coords
        self.phi_e: nn.Sequential = mlp(
            2 * hidden_dim + num_rbf + EDGE_FEATURE_DIM,
            hidden_dim,
            hidden_dim,
            final_activation=True,
        )
        self.phi_h: nn.Sequential = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.phi_x: nn.Sequential | None = None
        if update_coords:
            self.phi_x = mlp(hidden_dim, hidden_dim, 1)
            # Start with near-zero displacements (as in the reference code), so
            # early training does not scramble the geometry it relies on.
            nn.init.xavier_uniform_(self.phi_x[-1].weight, gain=1e-3)
            nn.init.zeros_(self.phi_x[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        rbf: RadialBasis,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update atom embeddings and (optionally) coordinates.

        Args:
            h: Atom embeddings [num_nodes, hidden_dim].
            pos: Coordinates [num_nodes, 3].
            edge_index: Directed edges [2, num_edges], source -> target.
            edge_attr: Bond-type one-hot [num_edges, 4].
            rbf: Radial basis shared by all layers.

        Returns:
            Updated (h, pos) with unchanged shapes.
        """
        source, target = edge_index
        rel, dist = edge_vectors(pos, edge_index)

        messages: torch.Tensor = self.phi_e(
            torch.cat([h[target], h[source], rbf(dist), edge_attr], dim=-1)
        )
        aggregated: torch.Tensor = scatter(
            messages, target, dim=0, dim_size=h.size(0), reduce="sum"
        )
        h = h + self.phi_h(torch.cat([h, aggregated], dim=-1))

        if self.phi_x is not None:
            # Normalising by (d + 1) bounds each step even for distant pairs;
            # averaging over neighbours makes it independent of the degree.
            displacement: torch.Tensor = rel / (dist + 1.0) * self.phi_x(messages)
            pos = pos + scatter(displacement, target, dim=0, dim_size=pos.size(0), reduce="mean")

        return h, pos


class EGNN(MolecularGNN):
    """EGNN that updates the coordinates at every layer."""

    uses_coordinates: ClassVar[bool] = True
    update_coords: ClassVar[bool] = True

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_targets: int,
        num_rbf: int,
        rbf_cutoff: float,
    ) -> None:
        super().__init__(hidden_dim, num_targets)
        self.rbf: RadialBasis = RadialBasis(num_rbf, rbf_cutoff)
        self.layers: nn.ModuleList = nn.ModuleList(
            EGNNLayer(hidden_dim, num_rbf, self.update_coords) for _ in range(num_layers)
        )

    def encode(self, data: Data) -> torch.Tensor:
        return self.encode_with_coordinates(data)[0]

    def encode_with_coordinates(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Run every layer and also return the final (equivariant) coordinates."""
        h: torch.Tensor = self.atom_embedding(data.x)
        pos: torch.Tensor = data.pos
        for layer in self.layers:
            h, pos = layer(h, pos, data.edge_index, data.edge_attr, self.rbf)
        return h, pos


class EGNNStatic(EGNN):
    """EGNN with frozen coordinates: an E(3)-invariant MPNN over bond lengths."""

    update_coords: ClassVar[bool] = False
