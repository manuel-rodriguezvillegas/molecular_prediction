"""Symmetry tests: the mathematical properties that motivate each architecture."""

import pytest
import torch
from torch_geometric.data import Batch

from molecular_prediction.config import MODEL_NAMES, Config
from molecular_prediction.models import EGNN, MolecularGNN, build_model, count_parameters
from tests.conftest import random_rotation

ATOL: float = 1e-9


def _model(name: str, config: Config) -> MolecularGNN:
    torch.manual_seed(0)
    return build_model(name, config).double().eval()


def _transform(batch: Batch, rotation: torch.Tensor, shift: torch.Tensor) -> Batch:
    moved: Batch = batch.clone()
    moved.pos = batch.pos @ rotation.T + shift
    return moved


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape(name: str, small_config: Config, batch: Batch) -> None:
    assert _model(name, small_config)(batch).shape == (3, 4)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("reflect", [False, True])
def test_predictions_are_e3_invariant(
    name: str, reflect: bool, small_config: Config, batch: Batch
) -> None:
    """Rotating, reflecting or translating a molecule never changes its properties."""
    model: MolecularGNN = _model(name, small_config)
    rotation: torch.Tensor = random_rotation(seed=1, reflect=reflect)
    shift: torch.Tensor = torch.tensor([3.0, -1.0, 2.5], dtype=torch.float64)
    with torch.no_grad():
        original: torch.Tensor = model(batch)
        transformed: torch.Tensor = model(_transform(batch, rotation, shift))
    torch.testing.assert_close(original, transformed, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_permutation_invariant(
    name: str, small_config: Config, batch: Batch
) -> None:
    """Relabelling the atoms is not a different molecule."""
    model: MolecularGNN = _model(name, small_config)
    perm: torch.Tensor = torch.cat(
        [
            torch.randperm(int(n)) + int(offset)
            for n, offset in zip(batch.ptr.diff(), batch.ptr[:-1], strict=True)
        ]
    )
    inverse: torch.Tensor = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))

    permuted: Batch = batch.clone()
    for key in ("x", "pos"):
        permuted[key] = batch[key][perm]
    permuted.edge_index = inverse[batch.edge_index]
    permuted.batch = batch.batch[perm]

    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(permuted), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_coordinate_usage(name: str, small_config: Config, batch: Batch) -> None:
    """Topological models ignore coordinates entirely; geometric ones depend on them."""
    model: MolecularGNN = _model(name, small_config)
    noisy: Batch = batch.clone()
    noisy.pos = batch.pos + torch.randn_like(batch.pos)
    with torch.no_grad():
        unchanged: bool = torch.allclose(model(batch), model(noisy), atol=ATOL)
    assert unchanged == (not model.uses_coordinates)


def test_egnn_coordinates_are_e3_equivariant(small_config: Config, batch: Batch) -> None:
    """x_out(R x + t) == R x_out(x) + t, with non-negligible coordinate updates."""
    model: EGNN = _model("EGNN", small_config)
    for layer in model.layers:
        torch.nn.init.normal_(layer.phi_x[-1].weight, std=0.5)

    rotation: torch.Tensor = random_rotation(seed=2, reflect=True)
    shift: torch.Tensor = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float64)
    with torch.no_grad():
        _, pos_out = model.encode_with_coordinates(batch)
        _, pos_out_moved = model.encode_with_coordinates(_transform(batch, rotation, shift))

    assert not torch.allclose(pos_out, batch.pos, atol=1e-3), "coordinates should move"
    torch.testing.assert_close(pos_out_moved, pos_out @ rotation.T + shift, atol=ATOL, rtol=0)


def test_egnn_static_keeps_coordinates(small_config: Config, batch: Batch) -> None:
    model: EGNN = _model("EGNNStatic", small_config)
    with torch.no_grad():
        _, pos_out = model.encode_with_coordinates(batch)
    torch.testing.assert_close(pos_out, batch.pos)


def test_default_models_have_matched_parameter_counts() -> None:
    """The comparison is only fair if capacity is comparable (within 3%)."""
    config: Config = Config()
    counts: list[int] = [count_parameters(build_model(n, config)) for n in MODEL_NAMES]
    assert max(counts) / min(counts) < 1.03, dict(zip(MODEL_NAMES, counts, strict=True))
