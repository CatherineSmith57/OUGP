from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ougp.data import CitationGraph


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_case_study.py"
SPEC = importlib.util.spec_from_file_location("run_case_study", SCRIPT_PATH)
assert SPEC is not None
run_case_study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_case_study)


def _toy_graph() -> CitationGraph:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 5, 7],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 2, 4, 6],
        ],
        dtype=torch.long,
    )
    return CitationGraph(
        x=torch.arange(40, dtype=torch.float32).view(10, 4),
        y=torch.arange(10, dtype=torch.long),
        edge_index=edge_index,
        train_mask=torch.tensor([True, True, False, False, False, False, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False, False, False, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True, True, True, True, True]),
    )


def test_node_sampling_builds_consistent_induced_subgraph() -> None:
    sampled = run_case_study.maybe_sample_nodes(_toy_graph(), sample_size=6, seed=0)

    assert sampled.num_nodes == 6
    assert sampled.edge_index.size(1) > 0
    assert sampled.edge_index.min() >= 0
    assert sampled.edge_index.max() < sampled.num_nodes
    assert sampled.x.shape == (6, 4)
    assert sampled.y.shape == (6,)
    assert sampled.train_mask.sum() > 0
    assert sampled.val_mask.sum() > 0
    assert sampled.test_mask.sum() > 0


def test_frontier_node_sampling_builds_consistent_induced_subgraph() -> None:
    sampled = run_case_study.maybe_sample_nodes(_toy_graph(), sample_size=6, seed=0, mode="frontier")

    assert sampled.num_nodes == 6
    assert sampled.edge_index.size(1) > 0
    assert sampled.edge_index.min() >= 0
    assert sampled.edge_index.max() < sampled.num_nodes
    assert sampled.train_mask.sum() > 0
    assert sampled.val_mask.sum() > 0
    assert sampled.test_mask.sum() > 0


def test_node_sampling_disabled_returns_original_dataset() -> None:
    dataset = _toy_graph()

    assert run_case_study.maybe_sample_nodes(dataset, sample_size=0, seed=0) is dataset
    assert run_case_study.maybe_sample_nodes(dataset, sample_size=dataset.num_nodes, seed=0) is dataset


def test_node_sampling_rejects_unknown_mode() -> None:
    try:
        run_case_study.maybe_sample_nodes(_toy_graph(), sample_size=6, seed=0, mode="bad")
    except ValueError as exc:
        assert "--node-sample-mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown node sample mode.")
