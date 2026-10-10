"""Ollivier-Ricci curvature on graphs with hand-computed values, and the size-control logic."""

import networkx as nx
import numpy as np
import pytest

from molecular_prediction.experiments.curvature import ollivier_ricci_curvature
from molecular_prediction.experiments.curvature_analysis import (
    assign_quantiles,
    size_controlled_regression,
)


def _curvature(graph: nx.Graph, u: int, v: int) -> float:
    curvatures = ollivier_ricci_curvature(graph, alpha=0.5)
    return curvatures[(u, v)] if (u, v) in curvatures else curvatures[(v, u)]


def test_isolated_edge_has_curvature_one() -> None:
    # mu_u = mu_v = {u: 1/2, v: 1/2}: nothing to transport.
    assert _curvature(nx.path_graph(2), 0, 1) == pytest.approx(1.0)


def test_triangle_edge() -> None:
    # Move 1/4 of mass one step: W1 = 1/4, kappa = 3/4.
    assert _curvature(nx.complete_graph(3), 0, 1) == pytest.approx(0.75)


def test_bridge_between_triangles_is_negatively_curved() -> None:
    # Two triangles {0,1,2} and {3,4,5} joined by the bridge 2-3.
    # W1 = 1/3 (bridge endpoints) + 2 * 1/6 * 3 (far vertices) = 4/3 -> kappa = -1/3.
    graph: nx.Graph = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    assert _curvature(graph, 2, 3) == pytest.approx(-1.0 / 3.0)
    assert _curvature(graph, 0, 1) == pytest.approx(0.75)


def test_quantiles_are_balanced_and_ordered() -> None:
    rng = np.random.default_rng(0)
    min_curvature: np.ndarray = rng.choice([-0.5, -0.25, 0.0, 0.25], size=1_001)  # many ties
    mean_curvature: np.ndarray = rng.normal(size=1_001)
    labels: np.ndarray = assign_quantiles(min_curvature, mean_curvature, 4)

    sizes: np.ndarray = np.bincount(labels)
    assert sizes.max() - sizes.min() <= 1
    medians: list[float] = [float(np.median(min_curvature[labels == q])) for q in range(4)]
    assert medians == sorted(medians, reverse=True), "Q0 must be the least bottlenecked"


def test_regression_removes_a_pure_size_effect() -> None:
    """If improvement depends only on size, curvature must not survive size control."""
    rng = np.random.default_rng(0)
    num_atoms: np.ndarray = rng.integers(5, 30, size=5_000)
    min_curvature: np.ndarray = -0.05 * num_atoms + rng.normal(scale=0.3, size=5_000)
    improvement: np.ndarray = 0.01 * num_atoms + rng.normal(scale=0.02, size=5_000)

    result = size_controlled_regression(improvement, min_curvature, num_atoms)
    assert result["severity_coef"] > 5 * result["severity_se"]  # confounded on its own
    assert abs(result["severity_coef_given_size"]) < 3 * result["severity_se_given_size"]


def test_sp3_sp3_bond_curvature_depends_on_explicit_hydrogens() -> None:
    """Ethane: with hydrogens the C-C bond looks like a bottleneck, without them it does not.

    Each carbon has degree 4, so its walk spreads 1/8 on each neighbour; the three
    hydrogens of one carbon must travel 3 hops to those of the other:
    W1 = 3/8 + 3 * 3/8 = 3/2, kappa = -1/2. This is why most saturated molecules
    share a minimum curvature of exactly -0.5 on the all-atom graph.
    """
    carbon_carbon: list[tuple[int, int]] = [(0, 1)]
    hydrogens: list[tuple[int, int]] = [(0, h) for h in (2, 3, 4)] + [(1, h) for h in (5, 6, 7)]
    assert _curvature(nx.Graph(carbon_carbon + hydrogens), 0, 1) == pytest.approx(-0.5)
    assert _curvature(nx.Graph(carbon_carbon), 0, 1) == pytest.approx(1.0)
