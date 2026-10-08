"""Splitting, normalisation and noise."""

import torch
from torch_geometric.data import Batch, Data

from molecular_prediction.data.dataset import TargetStats, split_indices
from molecular_prediction.data.transforms import SelectAndNormaliseTargets, add_coordinate_noise
from tests.conftest import random_molecule


def test_split_is_a_disjoint_deterministic_partition() -> None:
    train, val, test = split_indices(1_000, (700, 100), seed=42)
    assert (len(train), len(val), len(test)) == (700, 100, 200)
    assert torch.equal(torch.cat([train, val, test]).sort().values, torch.arange(1_000))

    again = split_indices(1_000, (700, 100), seed=42)
    assert all(torch.equal(a, b) for a, b in zip((train, val, test), again, strict=True))
    assert not torch.equal(train, split_indices(1_000, (700, 100), seed=7)[0])


def test_split_is_shuffled_and_test_size_is_capped() -> None:
    train, _, test = split_indices(1_000, (700, 100), seed=0, test_size=50)
    assert len(test) == 50
    assert not torch.equal(train, torch.arange(700)), "split must not be contiguous"


def test_target_stats_round_trip() -> None:
    y: torch.Tensor = torch.randn(500, 4) * torch.tensor([1.0, 10.0, 0.1, 5.0]) + 3.0
    stats: TargetStats = TargetStats.fit(y)
    normalised: torch.Tensor = stats.normalise(y)
    torch.testing.assert_close(normalised.mean(dim=0), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(normalised.std(dim=0), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(stats.denormalise(normalised), y, atol=1e-4, rtol=0)
    restored: TargetStats = TargetStats.from_dict(stats.to_dict())
    torch.testing.assert_close(restored.mean, stats.mean)


def test_target_transform_selects_and_does_not_mutate_storage() -> None:
    raw_y: torch.Tensor = torch.arange(19, dtype=torch.float32).unsqueeze(0)
    data: Data = Data(y=raw_y)
    transform = SelectAndNormaliseTargets(
        [0, 1, 4, 11], mean=torch.tensor([0.0, 1.0, 2.0, 3.0]), std=torch.full((4,), 2.0)
    )
    out: Data = transform(data)
    torch.testing.assert_close(out.y, torch.tensor([[0.0, 0.0, 1.0, 4.0]]))
    torch.testing.assert_close(raw_y, torch.arange(19, dtype=torch.float32).unsqueeze(0))


def test_noise_is_reproducible_and_leaves_input_untouched() -> None:
    batch: Batch = Batch.from_data_list([random_molecule(200, seed=0)])
    original: torch.Tensor = batch.pos.clone()

    first = add_coordinate_noise(batch, 0.5, torch.Generator().manual_seed(3))
    second = add_coordinate_noise(batch, 0.5, torch.Generator().manual_seed(3))

    torch.testing.assert_close(first.pos, second.pos)
    torch.testing.assert_close(batch.pos, original)
    assert abs(float((first.pos - original).std()) - 0.5) < 0.05
    assert first.edge_index is batch.edge_index  # topology is shared, not perturbed
