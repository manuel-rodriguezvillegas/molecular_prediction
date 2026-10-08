"""Synthetic molecules so that the test-suite runs in seconds without QM9."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from molecular_prediction.config import Config
from molecular_prediction.data.dataset import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


def random_molecule(num_atoms: int, seed: int, num_targets: int = 4) -> Data:
    """A connected molecule-like graph: a chain closed into a ring, plus random features."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    bonds: list[tuple[int, int]] = [(i, i + 1) for i in range(num_atoms - 1)]
    bonds.append((0, num_atoms - 1))  # ring closure
    source: list[int] = [a for a, b in bonds] + [b for a, b in bonds]
    target: list[int] = [b for a, b in bonds] + [a for a, b in bonds]
    edge_index: torch.Tensor = torch.tensor([source, target])

    elements: torch.Tensor = torch.randint(0, 5, (num_atoms,), generator=generator)
    x: torch.Tensor = torch.zeros(num_atoms, NODE_FEATURE_DIM)
    x[torch.arange(num_atoms), elements] = 1.0
    x[:, 5:] = torch.rand(num_atoms, NODE_FEATURE_DIM - 5, generator=generator)

    bond_types: torch.Tensor = torch.randint(
        0, EDGE_FEATURE_DIM, (len(bonds),), generator=generator
    )
    edge_attr: torch.Tensor = torch.zeros(len(bonds), EDGE_FEATURE_DIM)
    edge_attr[torch.arange(len(bonds)), bond_types] = 1.0
    edge_attr = torch.cat([edge_attr, edge_attr])  # same type in both directions

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=1.5 * torch.randn(num_atoms, 3, generator=generator),
        y=torch.randn(1, num_targets, generator=generator),
    )


def random_rotation(seed: int, reflect: bool = False) -> torch.Tensor:
    """Random 3x3 orthogonal matrix; with ``reflect`` its determinant is -1."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))  # unique, uniformly distributed
    if (torch.det(q) < 0) != reflect:
        q[:, 0] = -q[:, 0]
    return q


@pytest.fixture
def small_config() -> Config:
    """Default configuration with narrow models, for speed."""
    config: Config = Config()
    config.model.hidden_dims = dict.fromkeys(config.model.hidden_dims, 16)
    config.model.num_layers = 3
    config.data.num_workers = 0
    return config


@pytest.fixture
def batch() -> Batch:
    """Three molecules of different sizes, in float64 for tight tolerances."""
    molecules: list[Data] = [random_molecule(n, seed=n) for n in (5, 7, 9)]
    data: Batch = Batch.from_data_list(molecules)
    for key in ("x", "edge_attr", "pos", "y"):
        data[key] = data[key].double()
    return data
