"""Ollivier-Ricci curvature of molecular graphs.

For an edge (u, v) of an unweighted graph,

    kappa(u, v) = 1 - W1(mu_u, mu_v) / d(u, v),   d(u, v) = 1,

where mu_x is the lazy random walk at x (mass alpha on x, (1 - alpha) / deg(x)
on each neighbour) and W1 is the Wasserstein-1 distance under the shortest-path
metric. Positive curvature: dense, clique-like neighbourhoods with many
alternative paths. Negative curvature: tree-like bridges through which all
information must flow, the edges responsible for over-squashing (Topping et al.,
ICLR 2022).

The transport problems are solved exactly with POT's network simplex (``ot.emd2``).
Molecular graphs are tiny (<= 29 atoms, degree <= 4), so each problem has at most
5 x 5 variables: exact Ollivier-Ricci is affordable here, whereas on large graphs
one would use the closed-form Balanced Forman curvature instead.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import ot


@dataclass(frozen=True)
class CurvatureSummary:
    """Per-molecule summary of its edge curvatures."""

    min_curvature: float
    mean_curvature: float
    num_edges: int


def to_networkx(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    """Undirected NetworkX graph from a (symmetric) [2, num_edges] edge index."""
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    return graph


def lazy_random_walk(graph: nx.Graph, node: int, alpha: float) -> tuple[list[int], np.ndarray]:
    """Support and probabilities of the lazy random walk measure at ``node``."""
    neighbours: list[int] = list(graph.neighbors(node))
    if not neighbours:
        return [node], np.array([1.0])
    weights: np.ndarray = np.full(len(neighbours) + 1, (1.0 - alpha) / len(neighbours))
    weights[0] = alpha
    return [node, *neighbours], weights


def ollivier_ricci_curvature(graph: nx.Graph, alpha: float = 0.5) -> dict[tuple[int, int], float]:
    """Curvature of every edge of an unweighted, undirected graph."""
    if graph.number_of_edges() == 0:
        return {}

    # All-pairs shortest paths once per molecule; each edge then only slices
    # the small cost sub-matrix between the two supports.
    num_nodes: int = graph.number_of_nodes()
    distances: np.ndarray = np.full((num_nodes, num_nodes), float(num_nodes))
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in lengths.items():
            distances[source, target] = float(length)

    curvatures: dict[tuple[int, int], float] = {}
    for u, v in graph.edges():
        support_u, mass_u = lazy_random_walk(graph, u, alpha)
        support_v, mass_v = lazy_random_walk(graph, v, alpha)
        cost: np.ndarray = distances[np.ix_(support_u, support_v)]
        curvatures[(u, v)] = 1.0 - float(ot.emd2(mass_u, mass_v, cost))
    return curvatures


def summarise_molecule(edge_index: np.ndarray, num_nodes: int, alpha: float) -> CurvatureSummary:
    """Minimum and mean edge curvature of one molecule (picklable for multiprocessing)."""
    curvatures: dict[tuple[int, int], float] = ollivier_ricci_curvature(
        to_networkx(edge_index, num_nodes), alpha
    )
    if not curvatures:
        return CurvatureSummary(min_curvature=0.0, mean_curvature=0.0, num_edges=0)
    values: np.ndarray = np.fromiter(curvatures.values(), dtype=float)
    return CurvatureSummary(
        min_curvature=float(values.min()),
        mean_curvature=float(values.mean()),
        num_edges=len(values),
    )
