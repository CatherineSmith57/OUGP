from __future__ import annotations

import torch
import torch.nn.functional as F

from ougp.model import OUGPConfig, OUGPGCN, OnlinePruningMemory


def test_recall_memory_writes_positive_recovery_bias() -> None:
    memory = OnlinePruningMemory(
        context_dim=3,
        rank=2,
        write_beta=0.1,
        write_lambda=0.9,
        recall_items=4,
        recall_beta=0.5,
        recall_decay=0.9,
    )
    stats = memory.write_recall(torch.tensor([1.0, 0.0, 2.0, 0.0]), top_k=0)

    correction = memory.recall_correction(torch.zeros(4))
    assert stats["updates"] == 2.0
    assert correction[0] > 0
    assert correction[2] > correction[0]
    assert correction[1] == 0


def test_steering_memory_updates_after_backward() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 5)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 2],
            [1, 2, 3, 4, 5, 0, 0, 1],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=5,
        hidden_dim=4,
        out_dim=2,
        num_nodes=6,
        num_edges=edge_index.size(1),
        memory_rank=3,
        graph_target_keep=0.6,
        param_target_keep=0.5,
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=True,
        use_cross=True,
        recall_gamma=0.2,
        recall_top_k=0,
        use_steering_memory=True,
        steer_gamma=0.1,
        steer_beta=0.2,
        steer_lambda=0.9,
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)

    logits, stats = model(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    memory_stats = model.write_memories()

    assert "steering_steer_state_norm" in stats
    assert "steering_memory_state_norm" in memory_stats
    assert model.steering_memory.state.norm() > 0
    assert "graph_recall_updates" in memory_stats
    assert "param_recall_updates" in memory_stats
