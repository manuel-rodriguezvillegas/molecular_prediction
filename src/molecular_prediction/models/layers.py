"""Building blocks shared by every model."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool


def mlp(
    in_dim: int, hidden_dim: int, out_dim: int, final_activation: bool = False
) -> nn.Sequential:
    """Two-layer perceptron with SiLU, the default non-linearity of every model."""
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if final_activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative positions and lengths of every directed edge j -> i.

    PyG convention: ``edge_index[0]`` is the source j, ``edge_index[1]`` the
    target i that receives the message. Every model in the project uses it.

    Returns:
        ``x_i - x_j`` of shape [num_edges, 3] and ``||x_i - x_j||`` of shape
        [num_edges, 1]. A tiny epsilon keeps the square root differentiable if
        two atoms ever coincide (e.g. under strong coordinate noise).
    """
    source, target = edge_index
    rel: torch.Tensor = pos[target] - pos[source]
    dist: torch.Tensor = torch.sqrt((rel**2).sum(dim=-1, keepdim=True) + 1e-8)
    return rel, dist


class RadialBasis(nn.Module):
    """Gaussian radial basis expansion of interatomic distances.

    A single scalar distance is a poor input for an MLP: the network has to
    learn sharp non-linear responses (e.g. a C=C vs C-C bond differs by 0.2 A)
    from one number. Expanding ``d`` onto Gaussians centred on a grid gives a
    smooth, localised "one-hot" of the distance that linear layers can use
    directly (SchNet, DimeNet, PaiNN all do this).

    Args:
        num_rbf: Number of Gaussians.
        cutoff: Largest centre, in Angstrom.
    """

    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        centres: torch.Tensor = torch.linspace(0.0, cutoff, num_rbf)
        width: float = cutoff / (num_rbf - 1)
        self.register_buffer("centres", centres)
        self.gamma: float = 0.5 / width**2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Map [num_edges, 1] distances to [num_edges, num_rbf] features."""
        return torch.exp(-self.gamma * (dist - self.centres) ** 2)


class MultiTaskHead(nn.Module):
    """Graph readout followed by one small MLP per target property.

    The readout concatenates sum and mean pooling. Sum pooling scales with the
    number of atoms, the right inductive bias for extensive properties
    (polarisability, heat capacity); mean pooling does not, which suits
    intensive ones (HOMO-LUMO gap). Each per-target MLP learns its own mix of
    the two, and separate heads stop the targets from competing for the same
    output weights while the message-passing trunk stays shared.

    Args:
        hidden_dim: Width of the node embeddings.
        num_targets: Number of properties to predict.
    """

    def __init__(self, hidden_dim: int, num_targets: int) -> None:
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList(
            mlp(2 * hidden_dim, hidden_dim // 2, 1) for _ in range(num_targets)
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Map node embeddings [num_nodes, hidden] to predictions [num_graphs, num_targets]."""
        graph: torch.Tensor = torch.cat(
            [global_add_pool(h, batch), global_mean_pool(h, batch)], dim=-1
        )
        return torch.cat([head(graph) for head in self.heads], dim=-1)
