from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from ougp.model import ChannelPruningMemory, OUGPConfig, OUGPGCN, OnlinePruningMemory


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


def test_channel_pruning_memory_keeps_per_channel_state() -> None:
    torch.manual_seed(0)
    memory = ChannelPruningMemory(
        context_dim=3,
        rank=2,
        num_channels=4,
        write_beta=0.5,
        write_lambda=0.9,
        recall_beta=0.5,
        recall_decay=0.9,
    )
    context = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    utility = torch.tensor([3.0, -1.0, 0.5, -2.0])

    before_corr, _, _, _ = memory.read(context)
    stats = memory.write(context, utility)
    after_corr, _, _, _ = memory.read(context)

    assert memory.state.shape == (4, 2, 2)
    assert stats["channel_state_norm_std"] > 0
    assert not torch.allclose(memory.state[0], memory.state[1])
    assert not torch.allclose(before_corr, after_corr)


def test_memory_write_modes_control_state_updates() -> None:
    torch.manual_seed(0)
    context = torch.randn(4, 3)
    utility = torch.tensor([1.0, -1.0, 0.5, 2.0])

    residual_memory = OnlinePruningMemory(context_dim=3, rank=2, write_beta=0.5, write_lambda=0.9)
    feature_memory = OnlinePruningMemory(context_dim=3, rank=2, write_beta=0.5, write_lambda=0.9)
    none_memory = OnlinePruningMemory(context_dim=3, rank=2, write_beta=0.5, write_lambda=0.9)

    residual_stats = residual_memory.write(context, utility, mode="residual")
    feature_stats = feature_memory.write(context, utility, mode="feature")
    none_stats = none_memory.write(context, utility, mode="none")

    assert residual_stats["write_mode"] == 1.0
    assert feature_stats["write_mode"] == 2.0
    assert none_stats["write_mode"] == 0.0
    assert residual_memory.state.norm() > 0
    assert feature_memory.state.norm() > 0
    assert none_memory.state.norm() == 0
    assert not torch.allclose(residual_memory.state, feature_memory.state)


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


@pytest.mark.parametrize("backbone", ["gcn", "sage", "gat"])
def test_backbone_uses_same_pruning_memory_path(backbone: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(7, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            [1, 2, 3, 4, 5, 6, 0, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=4,
        hidden_dim=5,
        out_dim=3,
        num_nodes=7,
        num_edges=edge_index.size(1),
        memory_rank=3,
        graph_target_keep=0.7,
        param_target_keep=0.6,
        graph_gamma=0.2,
        param_gamma=0.2,
        recall_gamma=0.1,
        backbone=backbone,
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)

    logits, stats = model(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    memory_stats = model.write_memories()

    assert logits.shape == (7, 3)
    if backbone == "sage":
        assert model.sage_lin1_neigh is not None
        assert model.sage_lin2_neigh is not None
    if backbone == "gat":
        assert model.gat_attn1_src is not None
        assert model.gat_attn2_src is not None
    assert stats["graph_keep"] < 1.0
    assert stats["param_keep"] < 1.0
    assert "graph_memory_state_norm" in memory_stats
    assert "param_memory_state_norm" in memory_stats


def test_deepgcn_backbone_uses_residual_hidden_blocks() -> None:
    torch.manual_seed(5)
    x = torch.randn(7, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            [1, 2, 3, 4, 5, 6, 0, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=4,
        hidden_dim=5,
        out_dim=3,
        num_nodes=7,
        num_edges=edge_index.size(1),
        num_gnn_layers=4,
        memory_rank=3,
        graph_target_keep=0.7,
        param_target_keep=0.6,
        backbone="deepgcn",
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)

    logits, stats = model(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    memory_stats = model.write_memories()
    resource_stats = model.resource_stats()

    assert logits.shape == (7, 3)
    assert len(model.deep_hidden_lins) == 2
    assert stats["graph_keep"] < 1.0
    assert stats["param_keep"] < 1.0
    assert resource_stats["dense_parameter_count"] > 4 * 5 + 5 * 3
    assert "graph_memory_state_norm" in memory_stats
    assert "param_memory_state_norm" in memory_stats


def test_resource_stats_track_message_and_parameter_reduction() -> None:
    torch.manual_seed(2)
    x = torch.randn(8, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=5,
        hidden_dim=6,
        out_dim=3,
        num_nodes=8,
        num_edges=edge_index.size(1),
        graph_target_keep=0.5,
        param_target_keep=0.5,
        backbone="gcn",
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)

    _logits, _stats = model(x, temperature=1.0)
    resource_stats = model.resource_stats()
    budget_reg = model.resource_regularization()

    assert resource_stats["effective_message_cost"] < resource_stats["dense_message_cost"]
    assert resource_stats["effective_parameter_count"] < resource_stats["dense_parameter_count"]
    assert 0.0 < resource_stats["message_cost_ratio"] < 1.0
    assert 0.0 < resource_stats["parameter_cost_ratio"] < 1.0
    assert resource_stats["memory_state_items"] > 0
    assert budget_reg.ndim == 0


def test_model_memory_write_none_skips_online_state_updates() -> None:
    torch.manual_seed(3)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=4,
        hidden_dim=5,
        out_dim=2,
        num_nodes=6,
        num_edges=edge_index.size(1),
        graph_target_keep=0.7,
        param_target_keep=0.6,
        memory_write_mode="none",
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)

    logits, _stats = model(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    memory_stats = model.write_memories()

    assert memory_stats["memory_write_skipped"] == 1.0
    assert model.graph_memory.state.norm() == 0
    assert model.param_memory.state.norm() == 0
    assert model.graph_memory.recall_bias.norm() == 0
    assert model.param_memory.recall_bias.norm() == 0


def test_static_score_initialization_and_freezing() -> None:
    torch.manual_seed(4)
    x = torch.eye(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 2],
            [1, 2, 3, 4, 5, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    cfg = OUGPConfig(
        in_dim=4,
        hidden_dim=5,
        out_dim=2,
        num_nodes=6,
        num_edges=edge_index.size(1),
        graph_target_keep=0.7,
        param_target_keep=0.6,
        graph_score_init="degree",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    )
    model = OUGPGCN(cfg, edge_index=edge_index, x=x)
    edge_before = model.edge_logits.detach().clone()
    param_before = model.param_logits.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logits, _stats = model(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    assert model.edge_logits.requires_grad is False
    assert model.param_logits.requires_grad is False
    assert edge_before.std(unbiased=False) > 0
    assert param_before.std(unbiased=False) > 0
    assert torch.allclose(model.edge_logits, edge_before)
    assert torch.allclose(model.param_logits, param_before)
