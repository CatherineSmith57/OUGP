"""OUGP modules for the first citation-network case study."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def ste_sigmoid(logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
    soft = torch.sigmoid(logits / temperature)
    if not hard:
        return soft
    hard_mask = (soft >= 0.5).to(soft.dtype)
    return hard_mask.detach() - soft.detach() + soft


def budgeted_sigmoid(logits: torch.Tensor, keep_rate: float, temperature: float, hard: bool) -> torch.Tensor:
    keep_rate = float(min(0.999, max(0.001, keep_rate)))
    threshold = torch.quantile(logits.detach().float(), 1.0 - keep_rate)
    prior = torch.logit(logits.new_tensor(keep_rate), eps=1e-6)
    soft = torch.sigmoid((logits - threshold) / temperature + prior)
    soft = (soft * (keep_rate / soft.detach().mean().clamp_min(1e-6))).clamp(0.0, 1.0)
    if not hard:
        return soft
    hard_mask = (logits >= threshold).to(logits.dtype)
    return hard_mask.detach() - soft.detach() + soft


def symmetric_norm(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    degree = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    degree.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def sparse_gcn_mm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=x.device)
    return torch.sparse.mm(adj.coalesce(), x)


def sparse_mean_mm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, _ = edge_index
    degree = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    degree.scatter_add_(0, row, edge_weight)
    norm_weight = edge_weight / degree[row].clamp_min(1e-12)
    return sparse_gcn_mm(edge_index, norm_weight, x, num_nodes)


def sparse_gat_mm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    x: torch.Tensor,
    attn_src: torch.Tensor,
    attn_dst: torch.Tensor,
    num_nodes: int,
    negative_slope: float = 0.2,
) -> torch.Tensor:
    row, col = edge_index
    logits = (x[col] * attn_src).sum(dim=-1) + (x[row] * attn_dst).sum(dim=-1)
    logits = F.leaky_relu(logits, negative_slope=negative_slope)
    max_per_row = torch.full((num_nodes,), -torch.inf, device=x.device, dtype=x.dtype)
    max_per_row.scatter_reduce_(0, row, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_per_row[row]) * edge_weight.clamp_min(0.0)
    denom = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    denom.scatter_add_(0, row, exp_logits)
    attn_weight = exp_logits / denom[row].clamp_min(1e-12)
    return sparse_gcn_mm(edge_index, attn_weight, x, num_nodes)


@dataclass(frozen=True)
class OUGPConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_nodes: int
    num_edges: int
    num_gnn_layers: int = 2
    edge_context_dim: int = 15
    param_context_dim: int = 6
    memory_rank: int = 8
    feature_context_dim: int = 6
    graph_target_keep: float = 0.70
    param_target_keep: float = 0.70
    graph_gamma: float = 0.35
    param_gamma: float = 0.35
    param_score_scale_decay: float = 0.95
    param_score_scale_min: float = 0.02
    param_score_scale_max: float = 0.50
    param_correction_clip: float = 2.0
    cross_gamma: float = 0.20
    write_beta: float = 0.12
    write_lambda: float = 0.98
    event_gamma: float = 0.0
    event_beta: float = 0.10
    event_decay: float = 0.95
    event_top_k: int = 2000
    recall_gamma: float = 0.0
    recall_beta: float = 0.10
    recall_decay: float = 0.95
    recall_top_k: int = 2000
    use_steering_memory: bool = False
    steer_context_dim: int = 10
    steer_gamma: float = 0.0
    steer_beta: float = 0.10
    steer_lambda: float = 0.95
    hard_masks: bool = False
    use_graph_pruning: bool = True
    use_param_pruning: bool = True
    use_memory: bool = True
    use_cross: bool = True
    backbone: str = "gcn"
    budget_target: float = 0.70
    memory_write_mode: str = "residual"
    graph_score_init: str = "constant"
    param_score_init: str = "constant"
    freeze_pruning_scores: bool = False
    seed: int = 0


class OnlinePruningMemory(nn.Module):
    """Fixed-capacity residual utility memory."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        write_beta: float,
        write_lambda: float,
        event_items: int = 0,
        event_beta: float = 0.10,
        event_decay: float = 0.95,
        recall_items: int = 0,
        recall_beta: float = 0.10,
        recall_decay: float = 0.95,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.event_beta = event_beta
        self.event_decay = event_decay
        self.recall_beta = recall_beta
        self.recall_decay = recall_decay
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.read_head = nn.Linear(rank, 1, bias=False)
        self.utility_head = nn.Linear(rank, 1, bias=False)
        self.register_buffer("state", torch.zeros(rank, rank))
        self.register_buffer("event_bias", torch.zeros(max(0, event_items)))
        self.register_buffer("recall_bias", torch.zeros(max(0, recall_items)))

    def reset_state(self) -> None:
        self.state.zero_()
        self.event_bias.zero_()
        self.recall_bias.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        return q, k, v

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.project_qkv(context)
        read_vec = q @ self.state.t()
        correction = self.read_head(read_vec).squeeze(-1)
        return correction, k, v, read_vec

    @torch.no_grad()
    def write(self, context: torch.Tensor, utility: torch.Tensor, mode: str = "residual") -> dict[str, float]:
        if mode not in {"residual", "feature", "none"}:
            raise ValueError("memory write mode must be one of: residual, feature, none.")
        if mode == "none":
            return {
                "write_mode": 0.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
            }
        if context.numel() == 0:
            return {
                "write_mode": 1.0 if mode == "residual" else 2.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
            }
        _, k, v = self.project_qkv(context.detach())
        pred_vec = k @ self.state.t()
        pred = self.utility_head(pred_vec).squeeze(-1)
        utility = utility.detach().float()
        utility = (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)
        residual = utility - pred
        if mode == "residual":
            target_v = v * residual.tanh().unsqueeze(-1)
        else:
            target_v = v
        current = k @ self.state.t()
        delta_vec = target_v - current
        delta_state = torch.einsum("br,bk->rk", delta_vec, k) / max(1, context.size(0))
        self.state.mul_(self.write_lambda).add_(self.write_beta * delta_state)
        self.state.clamp_(-5.0, 5.0)
        return {
            "write_mode": 1.0 if mode == "residual" else 2.0,
            "utility_mean": float(utility.mean().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "state_norm": float(self.state.norm().item()),
        }

    @torch.no_grad()
    def event_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.event_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.event_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def recall_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.recall_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.recall_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def write_events(self, event_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.event_bias.numel() == 0 or event_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.event_bias.norm().item())}
        if self.event_bias.numel() != event_delta.numel():
            raise ValueError("event_delta shape must match event_bias shape.")

        self.event_bias.mul_(self.event_decay)
        event_delta = event_delta.detach().float().to(self.event_bias.device)
        active = event_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.event_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(event_delta.abs(), k=top_k).indices
            self.event_bias[indices] += self.event_beta * event_delta[indices]
            updates = float(indices.numel())
        else:
            self.event_bias.add_(self.event_beta * event_delta)
            updates = float(active.sum().item())
        self.event_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.event_bias.mean().item()),
            "bias_abs_mean": float(self.event_bias.abs().mean().item()),
            "bias_norm": float(self.event_bias.norm().item()),
        }

    @torch.no_grad()
    def write_recall(self, recall_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.recall_bias.numel() == 0 or recall_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}
        if self.recall_bias.numel() != recall_delta.numel():
            raise ValueError("recall_delta shape must match recall_bias shape.")

        self.recall_bias.mul_(self.recall_decay)
        recall_delta = recall_delta.detach().float().to(self.recall_bias.device)
        active = recall_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(recall_delta.abs(), k=top_k).indices
            self.recall_bias[indices] += self.recall_beta * recall_delta[indices]
            updates = float(indices.numel())
        else:
            self.recall_bias.add_(self.recall_beta * recall_delta)
            updates = float(active.sum().item())
        self.recall_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.recall_bias.mean().item()),
            "bias_abs_mean": float(self.recall_bias.abs().mean().item()),
            "bias_norm": float(self.recall_bias.norm().item()),
        }


class ChannelPruningMemory(nn.Module):
    """Channel-specific residual utility memory for parameter pruning."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        num_channels: int,
        write_beta: float,
        write_lambda: float,
        recall_beta: float = 0.10,
        recall_decay: float = 0.95,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.num_channels = num_channels
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.recall_beta = recall_beta
        self.recall_decay = recall_decay
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.read_head = nn.Linear(rank, 1, bias=False)
        self.utility_head = nn.Linear(rank, 1, bias=False)
        self.register_buffer("state", torch.zeros(num_channels, rank, rank))
        self.register_buffer("event_bias", torch.zeros(0))
        self.register_buffer("recall_bias", torch.zeros(num_channels))

    def reset_state(self) -> None:
        self.state.zero_()
        self.recall_bias.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        return q, k, v

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.size(0) != self.num_channels:
            raise ValueError("ChannelPruningMemory context rows must match num_channels.")

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_context(context)
        q, k, v = self.project_qkv(context)
        read_vec = torch.einsum("ck,cok->co", q, self.state)
        correction = self.read_head(read_vec).squeeze(-1)
        return correction, k, v, read_vec

    @torch.no_grad()
    def write(self, context: torch.Tensor, utility: torch.Tensor, mode: str = "residual") -> dict[str, float]:
        if mode not in {"residual", "feature", "none"}:
            raise ValueError("memory write mode must be one of: residual, feature, none.")
        if mode == "none":
            channel_norm = self.state.detach().float().norm(dim=(1, 2))
            return {
                "write_mode": 0.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
                "channel_state_norm_mean": float(channel_norm.mean().item()),
                "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
            }
        if context.numel() == 0:
            channel_norm = self.state.detach().float().norm(dim=(1, 2))
            return {
                "write_mode": 1.0 if mode == "residual" else 2.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
                "channel_state_norm_mean": float(channel_norm.mean().item()),
                "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
            }
        self._validate_context(context)
        _, k, v = self.project_qkv(context.detach())
        pred_vec = torch.einsum("ck,cok->co", k, self.state)
        pred = self.utility_head(pred_vec).squeeze(-1)
        utility = utility.detach().float()
        utility = (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)
        residual = utility - pred
        if mode == "residual":
            target_v = v * residual.tanh().unsqueeze(-1)
        else:
            target_v = v
        current = torch.einsum("ck,cok->co", k, self.state)
        delta_vec = target_v - current
        delta_state = torch.einsum("co,ck->cok", delta_vec, k)
        self.state.mul_(self.write_lambda).add_(self.write_beta * delta_state)
        self.state.clamp_(-5.0, 5.0)
        channel_norm = self.state.detach().float().norm(dim=(1, 2))
        return {
            "write_mode": 1.0 if mode == "residual" else 2.0,
            "utility_mean": float(utility.mean().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "state_norm": float(self.state.norm().item()),
            "channel_state_norm_mean": float(channel_norm.mean().item()),
            "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
        }

    @torch.no_grad()
    def recall_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.recall_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.recall_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def write_recall(self, recall_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.recall_bias.numel() == 0 or recall_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}
        if self.recall_bias.numel() != recall_delta.numel():
            raise ValueError("recall_delta shape must match recall_bias shape.")

        self.recall_bias.mul_(self.recall_decay)
        recall_delta = recall_delta.detach().float().to(self.recall_bias.device)
        active = recall_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(recall_delta.abs(), k=top_k).indices
            self.recall_bias[indices] += self.recall_beta * recall_delta[indices]
            updates = float(indices.numel())
        else:
            self.recall_bias.add_(self.recall_beta * recall_delta)
            updates = float(active.sum().item())
        self.recall_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.recall_bias.mean().item()),
            "bias_abs_mean": float(self.recall_bias.abs().mean().item()),
            "bias_norm": float(self.recall_bias.norm().item()),
        }


class MemorySteeringMLP(nn.Module):
    """Delta-memory-style global steering state for hidden representation repair."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        hidden_dim: int,
        write_beta: float,
        write_lambda: float,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.beta_proj = nn.Linear(context_dim, rank)
        self.target_proj = nn.Linear(hidden_dim, rank)
        self.steer_head = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.steer_head[-1].weight)
        nn.init.zeros_(self.steer_head[-1].bias)
        self.register_buffer("state", torch.zeros(rank, rank))

    def reset_state(self) -> None:
        self.state.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        beta = torch.sigmoid(self.beta_proj(context))
        return q, k, v, beta

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        q, _, _, _ = self.project_qkv(context)
        read_vec = q @ self.state.t()
        delta_h = self.steer_head(read_vec).squeeze(0)
        stats = {
            "steer_delta_norm": float(delta_h.detach().float().norm().item()),
            "steer_state_norm": float(self.state.detach().float().norm().item()),
        }
        return delta_h, stats

    @torch.no_grad()
    def write(self, context: torch.Tensor, target_delta: torch.Tensor) -> dict[str, float]:
        if context.numel() == 0 or target_delta.numel() == 0:
            return {"target_norm": 0.0, "residual_mean": 0.0, "state_norm": float(self.state.norm().item())}
        _, k, v, beta = self.project_qkv(context.detach())
        target_value = self.target_proj(target_delta.detach().float().unsqueeze(0))
        pred_value = k @ self.state.t()
        residual = target_value - pred_value
        write_value = v + residual.tanh()
        delta_state = torch.einsum("br,bk->rk", write_value, k) / max(1, context.size(0))
        beta_vec = beta.mean(dim=0).unsqueeze(-1)
        self.state.mul_(self.write_lambda).add_(self.write_beta * beta_vec * delta_state)
        self.state.clamp_(-5.0, 5.0)
        return {
            "target_norm": float(target_delta.detach().float().norm().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "beta_mean": float(beta.mean().item()),
            "state_norm": float(self.state.norm().item()),
        }


class OUGPGCN(nn.Module):
    """Masked GNN backbones with online graph and channel pruning memories."""

    def __init__(self, cfg: OUGPConfig, edge_index: torch.Tensor, x: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        if cfg.memory_write_mode not in {"residual", "feature", "none"}:
            raise ValueError("cfg.memory_write_mode must be one of: residual, feature, none.")
        if cfg.graph_score_init not in {"constant", "random", "degree", "similarity"}:
            raise ValueError("cfg.graph_score_init must be one of: constant, random, degree, similarity.")
        if cfg.param_score_init not in {"constant", "random", "magnitude"}:
            raise ValueError("cfg.param_score_init must be one of: constant, random, magnitude.")
        self.register_buffer("base_edge_index", edge_index)
        self.register_buffer("x_ref", x)
        self.lin1 = nn.Linear(cfg.in_dim, cfg.hidden_dim, bias=False)
        self.lin2 = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        if cfg.backbone not in {"gcn", "sage", "gat", "deepgcn"}:
            raise ValueError("cfg.backbone must be one of: 'gcn', 'sage', 'gat', 'deepgcn'.")
        if cfg.backbone != "deepgcn" and cfg.num_gnn_layers != 2:
            raise ValueError("cfg.num_gnn_layers is currently only configurable for the 'deepgcn' backbone.")
        if cfg.backbone == "deepgcn" and cfg.num_gnn_layers < 3:
            raise ValueError("deepgcn requires cfg.num_gnn_layers >= 3.")
        self.deep_hidden_lins = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False) for _ in range(max(0, cfg.num_gnn_layers - 2))]
        )
        if cfg.backbone == "sage":
            self.sage_lin1_neigh = nn.Linear(cfg.in_dim, cfg.hidden_dim, bias=False)
            self.sage_lin2_neigh = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        else:
            self.sage_lin1_neigh = None
            self.sage_lin2_neigh = None
        if cfg.backbone == "gat":
            self.gat_attn1_src = nn.Parameter(torch.empty(cfg.hidden_dim))
            self.gat_attn1_dst = nn.Parameter(torch.empty(cfg.hidden_dim))
            self.gat_attn2_src = nn.Parameter(torch.empty(cfg.out_dim))
            self.gat_attn2_dst = nn.Parameter(torch.empty(cfg.out_dim))
            nn.init.xavier_uniform_(self.gat_attn1_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn1_dst.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn2_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn2_dst.unsqueeze(0))
        else:
            self.gat_attn1_src = None
            self.gat_attn1_dst = None
            self.gat_attn2_src = None
            self.gat_attn2_dst = None
        edge_init = self.initial_edge_scores(cfg.graph_score_init)
        param_init = self.initial_param_scores(cfg.param_score_init)
        self.edge_logits = nn.Parameter(edge_init, requires_grad=not cfg.freeze_pruning_scores)
        self.param_logits = nn.Parameter(param_init, requires_grad=not cfg.freeze_pruning_scores)
        self.register_buffer("param_logit_scale_ema", torch.tensor(float(cfg.param_score_scale_min)))

        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        rand = torch.randn(cfg.in_dim, cfg.feature_context_dim, generator=generator) / cfg.in_dim**0.5
        self.register_buffer("feature_proj", rand)

        self.graph_memory = OnlinePruningMemory(
            cfg.edge_context_dim,
            cfg.memory_rank,
            cfg.write_beta,
            cfg.write_lambda,
            event_items=cfg.num_edges,
            event_beta=cfg.event_beta,
            event_decay=cfg.event_decay,
            recall_items=cfg.num_edges,
            recall_beta=cfg.recall_beta,
            recall_decay=cfg.recall_decay,
        )
        self.param_memory = ChannelPruningMemory(
            cfg.param_context_dim,
            cfg.memory_rank,
            cfg.hidden_dim,
            cfg.write_beta,
            cfg.write_lambda,
            recall_beta=cfg.recall_beta,
            recall_decay=cfg.recall_decay,
        )
        self.steering_memory = MemorySteeringMLP(
            cfg.steer_context_dim,
            cfg.memory_rank,
            cfg.hidden_dim,
            cfg.steer_beta,
            cfg.steer_lambda,
        )

        self.last_graph_score: torch.Tensor | None = None
        self.last_param_score: torch.Tensor | None = None
        self.last_graph_mask: torch.Tensor | None = None
        self.last_param_mask: torch.Tensor | None = None
        self.last_graph_utility: torch.Tensor | None = None
        self.last_param_utility: torch.Tensor | None = None
        self.prev_graph_mask: torch.Tensor | None = None
        self.prev_param_mask: torch.Tensor | None = None
        self.trace_prev_graph_mask: torch.Tensor | None = None
        self.event_prev_graph_mask: torch.Tensor | None = None
        self.recall_prev_graph_mask: torch.Tensor | None = None
        self.recall_prev_param_mask: torch.Tensor | None = None
        self.last_steering_context: torch.Tensor | None = None
        self.last_steered_hidden: torch.Tensor | None = None

    def normalized_init_scores(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float()
        std = scores.std(unbiased=False)
        if float(std.item()) <= 1e-8:
            return torch.full_like(scores, 2.0)
        return (scores - scores.mean()) / std.clamp_min(1e-8)

    def initial_edge_scores(self, mode: str) -> torch.Tensor:
        if mode == "constant":
            return torch.full((self.cfg.num_edges,), 2.0)
        generator = torch.Generator(device=self.base_edge_index.device)
        generator.manual_seed(self.cfg.seed)
        if mode == "random":
            scores = torch.randn(self.cfg.num_edges, generator=generator, device=self.base_edge_index.device)
        elif mode == "degree":
            row, col = self.base_edge_index
            degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(self.base_edge_index.device)
            scores = degree[row] + degree[col]
        elif mode == "similarity":
            row, col = self.base_edge_index
            x_norm = l2_normalize(self.x_ref.detach().float())
            scores = (x_norm[row] * x_norm[col]).sum(dim=-1)
        else:
            raise ValueError(f"Unknown graph score init mode {mode!r}.")
        return self.normalized_init_scores(scores)

    def initial_param_scores(self, mode: str) -> torch.Tensor:
        if mode == "constant":
            return torch.full((self.cfg.hidden_dim,), 2.0)
        generator = torch.Generator(device=self.lin1.weight.device)
        generator.manual_seed(self.cfg.seed + 17)
        if mode == "random":
            scores = torch.randn(self.cfg.hidden_dim, generator=generator, device=self.lin1.weight.device)
        elif mode == "magnitude":
            w1_norm = self.lin1.weight.detach().t().norm(dim=0)
            w2_norm = self.lin2.weight.detach().norm(dim=0)
            if self.cfg.backbone == "deepgcn" and len(self.deep_hidden_lins) > 0:
                hidden_norms = [layer.weight.detach().norm(dim=0) for layer in self.deep_hidden_lins]
                w2_norm = torch.stack([w2_norm, *hidden_norms], dim=0).mean(dim=0)
            scores = w1_norm + w2_norm
        else:
            raise ValueError(f"Unknown parameter score init mode {mode!r}.")
        return self.normalized_init_scores(scores)

    def reset_memory(self) -> None:
        self.graph_memory.reset_state()
        self.param_memory.reset_state()
        self.steering_memory.reset_state()
        self.param_logit_scale_ema.fill_(float(self.cfg.param_score_scale_min))

    def dense_edge_count(self) -> torch.Tensor:
        edge_count = self.edge_logits.new_tensor(float(self.cfg.num_edges))
        if self.cfg.backbone in {"gcn", "gat", "deepgcn"}:
            edge_count = edge_count + float(self.cfg.num_nodes)
        return edge_count

    def effective_edge_count(self) -> torch.Tensor:
        if self.last_graph_mask is None:
            kept = self.edge_logits.new_tensor(float(self.cfg.num_edges))
        else:
            kept = self.last_graph_mask.sum()
        if self.cfg.backbone in {"gcn", "gat", "deepgcn"}:
            kept = kept + float(self.cfg.num_nodes)
        return kept

    def dense_parameter_count(self) -> torch.Tensor:
        base = self.edge_logits.new_tensor(float(self.cfg.in_dim * self.cfg.hidden_dim + self.cfg.hidden_dim * self.cfg.out_dim))
        if self.cfg.backbone == "sage":
            base = base * 2.0
        elif self.cfg.backbone == "gat":
            base = base + float(2 * self.cfg.hidden_dim + 2 * self.cfg.out_dim)
        elif self.cfg.backbone == "deepgcn":
            base = base + float(max(0, self.cfg.num_gnn_layers - 2) * self.cfg.hidden_dim * self.cfg.hidden_dim)
        return base

    def effective_parameter_count(self) -> torch.Tensor:
        if self.last_param_mask is None:
            hidden_keep = self.edge_logits.new_tensor(float(self.cfg.hidden_dim))
        else:
            hidden_keep = self.last_param_mask.sum()
        base = self.edge_logits.new_tensor(float(self.cfg.in_dim)) * hidden_keep
        base = base + hidden_keep * float(self.cfg.out_dim)
        if self.cfg.backbone == "sage":
            base = base * 2.0
        elif self.cfg.backbone == "gat":
            base = base + 2.0 * hidden_keep + float(2 * self.cfg.out_dim)
        elif self.cfg.backbone == "deepgcn":
            base = base + float(max(0, self.cfg.num_gnn_layers - 2)) * hidden_keep * hidden_keep
        return base

    def dense_message_cost(self) -> torch.Tensor:
        dense_edges = self.dense_edge_count()
        if self.cfg.backbone in {"gcn", "deepgcn"}:
            dims = float(self.cfg.in_dim + (self.cfg.num_gnn_layers - 1) * self.cfg.hidden_dim)
        elif self.cfg.backbone == "sage":
            dims = float(self.cfg.in_dim + self.cfg.hidden_dim)
        else:
            dims = float(3 * self.cfg.hidden_dim + 3 * self.cfg.out_dim)
        return dense_edges * dims

    def effective_message_cost(self) -> torch.Tensor:
        effective_edges = self.effective_edge_count()
        if self.last_param_mask is None:
            hidden_keep = self.edge_logits.new_tensor(float(self.cfg.hidden_dim))
        else:
            hidden_keep = self.last_param_mask.sum()
        if self.cfg.backbone == "gcn":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + hidden_keep
        elif self.cfg.backbone == "deepgcn":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + float(self.cfg.num_gnn_layers - 1) * hidden_keep
        elif self.cfg.backbone == "sage":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + hidden_keep
        else:
            dims = 3.0 * hidden_keep + float(3 * self.cfg.out_dim)
        return effective_edges * dims

    def memory_state_items(self) -> torch.Tensor:
        graph_items = float(self.cfg.memory_rank * self.cfg.memory_rank)
        param_items = float(self.cfg.hidden_dim * self.cfg.memory_rank * self.cfg.memory_rank)
        recall_items = float(self.cfg.num_edges + self.cfg.hidden_dim)
        steering_items = float(self.cfg.memory_rank * self.cfg.memory_rank)
        return self.edge_logits.new_tensor(graph_items + param_items + recall_items + steering_items)

    def resource_regularization(self) -> torch.Tensor:
        message_ratio = self.effective_message_cost() / self.dense_message_cost().clamp_min(1.0)
        param_ratio = self.effective_parameter_count() / self.dense_parameter_count().clamp_min(1.0)
        target = self.edge_logits.new_tensor(float(self.cfg.budget_target))
        return 0.5 * ((message_ratio - target).abs() + (param_ratio - target).abs())

    @torch.no_grad()
    def resource_stats(self) -> dict[str, float]:
        dense_message = self.dense_message_cost().detach().float()
        effective_message = self.effective_message_cost().detach().float()
        dense_params = self.dense_parameter_count().detach().float()
        effective_params = self.effective_parameter_count().detach().float()
        memory_items = self.memory_state_items().detach().float()
        return {
            "dense_message_cost": float(dense_message.item()),
            "effective_message_cost": float(effective_message.item()),
            "message_cost_ratio": float((effective_message / dense_message.clamp_min(1.0)).item()),
            "message_cost_reduction": float((1.0 - effective_message / dense_message.clamp_min(1.0)).item()),
            "dense_parameter_count": float(dense_params.item()),
            "effective_parameter_count": float(effective_params.item()),
            "parameter_cost_ratio": float((effective_params / dense_params.clamp_min(1.0)).item()),
            "parameter_cost_reduction": float((1.0 - effective_params / dense_params.clamp_min(1.0)).item()),
            "memory_state_items": float(memory_items.item()),
            "memory_overhead_vs_dense_params": float((memory_items / dense_params.clamp_min(1.0)).item()),
        }

    def edge_context(self, param_keep: torch.Tensor) -> torch.Tensor:
        row, col = self.base_edge_index
        x_proj = self.x_ref @ self.feature_proj
        degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(x_proj.device)
        degree = torch.log1p(degree / degree.mean().clamp_min(1.0)).unsqueeze(-1)
        param_ctx = param_keep.expand(row.numel(), 1)
        return torch.cat([x_proj[row], x_proj[col], degree[row], degree[col], param_ctx], dim=-1)

    def param_context(self, graph_keep: torch.Tensor) -> torch.Tensor:
        w1_norm = self.lin1.weight.t().norm(dim=0, keepdim=True).t()
        w2_norm = self.lin2.weight.norm(dim=0, keepdim=True).t()
        if self.cfg.backbone == "sage":
            if self.sage_lin1_neigh is None or self.sage_lin2_neigh is None:
                raise RuntimeError("GraphSAGE layers were not initialized.")
            w1_norm = torch.stack(
                [w1_norm, self.sage_lin1_neigh.weight.t().norm(dim=0, keepdim=True).t()],
                dim=0,
            ).mean(dim=0)
            w2_norm = torch.stack(
                [w2_norm, self.sage_lin2_neigh.weight.norm(dim=0, keepdim=True).t()],
                dim=0,
            ).mean(dim=0)
        elif self.cfg.backbone == "gat":
            if self.gat_attn1_src is None or self.gat_attn1_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            attn_norm = torch.stack([self.gat_attn1_src.abs(), self.gat_attn1_dst.abs()], dim=0).mean(dim=0).unsqueeze(-1)
            w1_norm = 0.5 * (w1_norm + attn_norm)
        elif self.cfg.backbone == "deepgcn" and len(self.deep_hidden_lins) > 0:
            hidden_norms = [layer.weight.norm(dim=0, keepdim=True).t() for layer in self.deep_hidden_lins]
            w2_norm = torch.stack([w2_norm, *hidden_norms], dim=0).mean(dim=0)
        channel_id = torch.linspace(0, 1, self.cfg.hidden_dim, device=w1_norm.device).unsqueeze(-1)
        sparsity = torch.sigmoid(self.param_logits).detach().unsqueeze(-1)
        graph_ctx = graph_keep.expand(self.cfg.hidden_dim, 1)
        bias = torch.ones_like(graph_ctx)
        return torch.cat([w1_norm, w2_norm, channel_id, sparsity, graph_ctx, bias], dim=-1)

    def normalized_param_signal(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.detach().float().mean().to(signal.dtype)
        std = centered.detach().float().std(unbiased=False)
        if float(std.item()) <= 1e-6:
            return torch.zeros_like(signal)
        normalized = centered / std.to(signal.dtype)
        return normalized.clamp(-self.cfg.param_correction_clip, self.cfg.param_correction_clip)

    def param_score_scale(self) -> torch.Tensor:
        current = self.param_logits.detach().float().std(unbiased=False)
        current = current.clamp(self.cfg.param_score_scale_min, self.cfg.param_score_scale_max)
        if self.training:
            decay = float(self.cfg.param_score_scale_decay)
            self.param_logit_scale_ema.mul_(decay).add_((1.0 - decay) * current.to(self.param_logit_scale_ema.device))
        return self.param_logit_scale_ema.to(device=self.param_logits.device, dtype=self.param_logits.dtype)

    def masks(self, temperature: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        base_graph_keep = torch.sigmoid(self.edge_logits).mean().detach()
        base_param_keep = torch.sigmoid(self.param_logits).mean().detach()
        param_scale = self.param_score_scale()

        if self.cfg.use_memory:
            graph_cross_ctx = base_param_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            param_cross_ctx = base_graph_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            graph_ctx = self.edge_context(graph_cross_ctx)
            param_ctx = self.param_context(param_cross_ctx)
            graph_corr, _, _, _ = self.graph_memory.read(graph_ctx)
            raw_param_corr, _, _, _ = self.param_memory.read(param_ctx)
            unit_param_corr = self.normalized_param_signal(raw_param_corr)
            param_corr = param_scale * unit_param_corr
        else:
            graph_corr = torch.zeros_like(self.edge_logits)
            raw_param_corr = torch.zeros_like(self.param_logits)
            unit_param_corr = torch.zeros_like(self.param_logits)
            param_corr = torch.zeros_like(self.param_logits)

        graph_score = self.edge_logits
        param_score = self.param_logits
        raw_param_recall_corr = self.param_memory.recall_correction(param_score)
        unit_param_recall_corr = self.normalized_param_signal(raw_param_recall_corr)
        param_recall_corr = param_scale * unit_param_recall_corr

        if self.cfg.use_memory:
            graph_score = graph_score + self.cfg.graph_gamma * graph_corr
            param_score = param_score + self.cfg.param_gamma * param_corr
            if self.cfg.use_graph_pruning and self.cfg.event_gamma != 0.0:
                graph_score = graph_score + self.cfg.event_gamma * self.graph_memory.event_correction(graph_score)
            if self.cfg.use_graph_pruning and self.cfg.recall_gamma != 0.0:
                graph_score = graph_score + self.cfg.recall_gamma * self.graph_memory.recall_correction(graph_score)
            if self.cfg.use_param_pruning and self.cfg.recall_gamma != 0.0:
                param_score = param_score + self.cfg.recall_gamma * param_recall_corr
        graph_keep_target = self.cfg.graph_target_keep
        param_keep_target = self.cfg.param_target_keep

        if self.cfg.use_graph_pruning and graph_keep_target < 0.999:
            graph_mask = budgeted_sigmoid(
                graph_score,
                graph_keep_target,
                temperature,
                self.cfg.hard_masks and not self.training,
            )
        else:
            graph_mask = torch.ones_like(graph_score)
        if self.cfg.use_param_pruning and param_keep_target < 0.999:
            param_mask = budgeted_sigmoid(
                param_score,
                param_keep_target,
                temperature,
                self.cfg.hard_masks and not self.training,
            )
        else:
            param_mask = torch.ones_like(param_score)

        self.last_graph_score = graph_score
        self.last_param_score = param_score
        self.last_graph_mask = graph_mask
        self.last_param_mask = param_mask
        stats = {
            "graph_keep": float(graph_mask.detach().mean().item()),
            "param_keep": float(param_mask.detach().mean().item()),
            "param_logits_mean": float(self.param_logits.detach().float().mean().item()),
            "param_logits_std": float(self.param_logits.detach().float().std(unbiased=False).item()),
            "param_memory_correction_mean": float(param_corr.detach().float().mean().item()),
            "param_memory_correction_std": float(param_corr.detach().float().std(unbiased=False).item()),
            "param_memory_unit_correction_mean": float(unit_param_corr.detach().float().mean().item()),
            "param_memory_unit_correction_std": float(unit_param_corr.detach().float().std(unbiased=False).item()),
            "param_memory_raw_correction_mean": float(raw_param_corr.detach().float().mean().item()),
            "param_memory_raw_correction_std": float(raw_param_corr.detach().float().std(unbiased=False).item()),
            "recall_correction_mean": float(param_recall_corr.detach().float().mean().item()),
            "recall_correction_std": float(param_recall_corr.detach().float().std(unbiased=False).item()),
            "recall_unit_correction_mean": float(unit_param_recall_corr.detach().float().mean().item()),
            "recall_unit_correction_std": float(unit_param_recall_corr.detach().float().std(unbiased=False).item()),
            "param_score_scale": float(param_scale.detach().float().item()),
            "param_memory_score_delta_std": float((self.cfg.param_gamma * param_corr).detach().float().std(unbiased=False).item()),
            "recall_score_delta_std": float((self.cfg.recall_gamma * param_recall_corr).detach().float().std(unbiased=False).item()),
        }
        return graph_mask, param_mask, stats

    def steering_context(
        self,
        hidden: torch.Tensor,
        graph_mask: torch.Tensor,
        param_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_detached = hidden.detach().float()
        graph_mask_detached = graph_mask.detach().float()
        param_mask_detached = param_mask.detach().float()
        values = [
            graph_mask_detached.mean(),
            param_mask_detached.mean(),
            hidden.new_tensor(self.cfg.graph_target_keep).float(),
            hidden.new_tensor(self.cfg.param_target_keep).float(),
            graph_mask_detached.std(unbiased=False),
            param_mask_detached.std(unbiased=False),
            hidden_detached.norm(dim=-1).mean(),
            hidden_detached.std(unbiased=False),
            self.graph_memory.state.detach().float().norm().to(hidden.device),
            self.param_memory.state.detach().float().norm().to(hidden.device),
        ]
        context = torch.stack([v.to(device=hidden.device, dtype=hidden.dtype) for v in values]).unsqueeze(0)
        if context.size(-1) != self.cfg.steer_context_dim:
            raise ValueError("steering_context size must match cfg.steer_context_dim.")
        return context

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        graph_mask, param_mask, stats = self.masks(temperature)
        num_nodes = x.size(0)
        if self.cfg.backbone in {"gcn", "deepgcn"}:
            self_loops = torch.arange(num_nodes, device=x.device)
            self_loop_index = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([self.base_edge_index, self_loop_index], dim=1)
            edge_weight = torch.cat([graph_mask, torch.ones(num_nodes, device=x.device)])
            norm_weight = symmetric_norm(edge_index, edge_weight, num_nodes)
            h = sparse_gcn_mm(edge_index, norm_weight, x, num_nodes)
            h = self.lin1(h)
        elif self.cfg.backbone == "sage":
            edge_index = self.base_edge_index
            edge_weight = graph_mask
            if self.sage_lin1_neigh is None or self.sage_lin2_neigh is None:
                raise RuntimeError("GraphSAGE layers were not initialized.")
            neigh = sparse_mean_mm(edge_index, edge_weight, x, num_nodes)
            h = self.lin1(x) + self.sage_lin1_neigh(neigh)
        elif self.cfg.backbone == "gat":
            self_loops = torch.arange(num_nodes, device=x.device)
            self_loop_index = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([self.base_edge_index, self_loop_index], dim=1)
            edge_weight = torch.cat([graph_mask, torch.ones(num_nodes, device=x.device)])
            if self.gat_attn1_src is None or self.gat_attn1_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            h_linear = self.lin1(x)
            h = sparse_gat_mm(edge_index, edge_weight, h_linear, self.gat_attn1_src, self.gat_attn1_dst, num_nodes)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone!r}.")

        self.last_steering_context = None
        self.last_steered_hidden = None
        if self.cfg.use_memory and self.cfg.use_steering_memory and self.cfg.steer_gamma != 0.0:
            steering_context = self.steering_context(h, graph_mask, param_mask)
            delta_h, steering_stats = self.steering_memory.read(steering_context)
            h = h + self.cfg.steer_gamma * delta_h.unsqueeze(0).to(dtype=h.dtype, device=h.device)
            self.last_steering_context = steering_context.detach()
            if self.training and h.requires_grad:
                h.retain_grad()
                self.last_steered_hidden = h
            stats.update({f"steering_{key}": value for key, value in steering_stats.items()})
        h = F.relu(h) * param_mask
        h = F.dropout(h, p=0.5, training=self.training)
        if self.cfg.backbone == "gcn":
            h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
            out = self.lin2(h)
        elif self.cfg.backbone == "deepgcn":
            for layer in self.deep_hidden_lins:
                h_next = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
                h_next = layer(h_next)
                h = F.relu(h + h_next) * param_mask
                h = F.dropout(h, p=0.5, training=self.training)
            h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
            out = self.lin2(h)
        elif self.cfg.backbone == "sage":
            neigh = sparse_mean_mm(edge_index, edge_weight, h, num_nodes)
            out = self.lin2(h) + self.sage_lin2_neigh(neigh)
        else:
            if self.gat_attn2_src is None or self.gat_attn2_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            out_linear = self.lin2(h)
            out = sparse_gat_mm(edge_index, edge_weight, out_linear, self.gat_attn2_src, self.gat_attn2_dst, num_nodes)
        return out, stats

    def regularization(self) -> torch.Tensor:
        reg = self.edge_logits.new_tensor(0.0)
        if self.cfg.use_graph_pruning and self.last_graph_mask is not None:
            reg = reg + (self.last_graph_mask.mean() - self.cfg.graph_target_keep).abs()
        if self.cfg.use_param_pruning and self.last_param_mask is not None:
            reg = reg + (self.last_param_mask.mean() - self.cfg.param_target_keep).abs()
        return reg

    @torch.no_grad()
    def churn(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.last_graph_mask is not None:
            graph_mask = self.last_graph_mask.detach().float()
            if self.prev_graph_mask is None:
                out["graph_churn"] = 0.0
            else:
                out["graph_churn"] = float((graph_mask - self.prev_graph_mask).abs().mean().item())
            self.prev_graph_mask = graph_mask
        if self.last_param_mask is not None:
            param_mask = self.last_param_mask.detach().float()
            if self.prev_param_mask is None:
                out["param_churn"] = 0.0
            else:
                out["param_churn"] = float((param_mask - self.prev_param_mask).abs().mean().item())
            self.prev_param_mask = param_mask
        return out

    @torch.no_grad()
    def graph_event_tensors(self, graph_utility: torch.Tensor) -> dict[str, torch.Tensor]:
        row, col = self.base_edge_index
        graph_utility = graph_utility.detach().float().to(row.device)
        degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(row.device)
        log_degree = torch.log1p(degree)
        feature_norm = self.x_ref.norm(dim=1).float().to(row.device)

        node_utility_sum = torch.zeros(self.cfg.num_nodes, device=row.device)
        node_utility_count = torch.zeros(self.cfg.num_nodes, device=row.device)
        node_utility_sum.scatter_add_(0, row, graph_utility)
        node_utility_sum.scatter_add_(0, col, graph_utility)
        ones = torch.ones_like(graph_utility)
        node_utility_count.scatter_add_(0, row, ones)
        node_utility_count.scatter_add_(0, col, ones)
        node_utility = node_utility_sum / node_utility_count.clamp_min(1.0)

        def minmax(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)

        degree_score = minmax(log_degree)
        feature_score = minmax(feature_norm)
        node_utility_score = minmax(node_utility)
        node_importance = (degree_score + feature_score + node_utility_score) / 3.0
        utility_score = minmax(graph_utility)
        edge_importance = 0.5 * utility_score + 0.25 * (node_importance[row] + node_importance[col])

        return {
            "row": row,
            "col": col,
            "degree": degree,
            "feature_norm": feature_norm,
            "graph_utility": graph_utility,
            "node_importance": node_importance,
            "edge_importance": edge_importance,
        }

    @torch.no_grad()
    def write_graph_event_memory(self, graph_utility: torch.Tensor) -> dict[str, float]:
        if self.last_graph_mask is None or not self.cfg.use_graph_pruning:
            return {}
        graph_mask = self.last_graph_mask.detach().float()
        if self.event_prev_graph_mask is None:
            self.event_prev_graph_mask = graph_mask.clone()
            return {
                "graph_event_updates": 0.0,
                "graph_event_bias_norm": float(self.graph_memory.event_bias.norm().item()),
            }

        mask_drop = (self.event_prev_graph_mask.to(graph_mask.device) - graph_mask).clamp_min(0.0)
        self.event_prev_graph_mask = graph_mask.clone()
        if not bool((mask_drop > 0).any().item()):
            return {
                "graph_event_updates": 0.0,
                "graph_event_bias_norm": float(self.graph_memory.event_bias.norm().item()),
            }

        event = self.graph_event_tensors(graph_utility)
        edge_importance = event["edge_importance"]
        centered_importance = edge_importance - edge_importance.mean()
        scaled_importance = centered_importance / centered_importance.std(unbiased=False).clamp_min(1e-6)
        event_delta = mask_drop * scaled_importance.clamp(-3.0, 3.0)
        event_stats = self.graph_memory.write_events(event_delta, self.cfg.event_top_k)
        return {f"graph_event_{key}": value for key, value in event_stats.items()}

    @torch.no_grad()
    def recall_delta_from_drop(self, previous_mask: torch.Tensor, current_mask: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
        mask_drop = (previous_mask.to(current_mask.device) - current_mask).clamp_min(0.0)
        if not bool((mask_drop > 0).any().item()):
            return torch.zeros_like(current_mask)
        utility = utility.detach().float().to(current_mask.device)
        scaled_utility = (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)
        important_after_drop = scaled_utility.clamp_min(0.0).clamp_max(3.0)
        return mask_drop * important_after_drop

    @torch.no_grad()
    def write_graph_recall_memory(self, graph_utility: torch.Tensor) -> dict[str, float]:
        if self.last_graph_mask is None or not self.cfg.use_graph_pruning:
            return {}
        graph_mask = self.last_graph_mask.detach().float()
        if self.recall_prev_graph_mask is None:
            self.recall_prev_graph_mask = graph_mask.clone()
            return {
                "graph_recall_updates": 0.0,
                "graph_recall_bias_norm": float(self.graph_memory.recall_bias.norm().item()),
            }
        recall_delta = self.recall_delta_from_drop(self.recall_prev_graph_mask, graph_mask, graph_utility)
        self.recall_prev_graph_mask = graph_mask.clone()
        recall_stats = self.graph_memory.write_recall(recall_delta, self.cfg.recall_top_k)
        return {f"graph_recall_{key}": value for key, value in recall_stats.items()}

    @torch.no_grad()
    def write_param_recall_memory(self, param_utility: torch.Tensor) -> dict[str, float]:
        if self.last_param_mask is None or not self.cfg.use_param_pruning:
            return {}
        param_mask = self.last_param_mask.detach().float()
        if self.recall_prev_param_mask is None:
            self.recall_prev_param_mask = param_mask.clone()
            return {
                "param_recall_updates": 0.0,
                "param_recall_bias_norm": float(self.param_memory.recall_bias.norm().item()),
            }
        recall_delta = self.recall_delta_from_drop(self.recall_prev_param_mask, param_mask, param_utility)
        self.recall_prev_param_mask = param_mask.clone()
        recall_stats = self.param_memory.write_recall(recall_delta, self.cfg.recall_top_k)
        return {f"param_recall_{key}": value for key, value in recall_stats.items()}

    @torch.no_grad()
    def write_steering_memory(self) -> dict[str, float]:
        if (
            not self.cfg.use_steering_memory
            or self.last_steering_context is None
            or self.last_steered_hidden is None
            or self.last_steered_hidden.grad is None
        ):
            return {}
        grad = self.last_steered_hidden.grad.detach().float()
        target_delta = -grad.mean(dim=0)
        target_delta = target_delta / target_delta.norm().clamp_min(1e-6)
        steering_stats = self.steering_memory.write(self.last_steering_context, target_delta)
        return {f"steering_memory_{key}": value for key, value in steering_stats.items()}

    @torch.no_grad()
    def graph_trace_snapshot(
        self,
        dataset: str,
        variant: str,
        seed: int,
        epoch: int,
        top_k: int,
    ) -> list[dict[str, float | int | str]]:
        if self.last_graph_mask is None or self.last_graph_score is None or top_k <= 0:
            return []
        graph_mask = self.last_graph_mask.detach().float()
        if self.trace_prev_graph_mask is None:
            self.trace_prev_graph_mask = graph_mask.clone()
            return []

        mask_delta = (self.trace_prev_graph_mask.to(graph_mask.device) - graph_mask).clamp_min(0.0)
        self.trace_prev_graph_mask = graph_mask.clone()
        positive = mask_delta > 0
        if not bool(positive.any().item()):
            return []

        k = min(int(top_k), int(positive.sum().item()))
        top_values, edge_ids = torch.topk(mask_delta.masked_fill(~positive, -1.0), k=k)
        keep = top_values > 0
        if not bool(keep.any().item()):
            return []
        edge_ids = edge_ids[keep]

        graph_utility = (
            self.last_graph_utility.detach().float()
            if self.last_graph_utility is not None
            else torch.zeros_like(graph_mask)
        )
        event = self.graph_event_tensors(graph_utility)
        row = event["row"]
        col = event["col"]
        degree = event["degree"]
        feature_norm = event["feature_norm"]
        node_importance = event["node_importance"]
        edge_importance_tensor = event["edge_importance"]

        current_graph_keep = float(graph_mask.mean().item())
        current_param_keep = (
            float(self.last_param_mask.detach().float().mean().item())
            if self.last_param_mask is not None
            else float(self.cfg.param_target_keep)
        )

        rows = []
        for edge_id_tensor in edge_ids.detach().cpu():
            edge_id = int(edge_id_tensor.item())
            src = int(row[edge_id].item())
            dst = int(col[edge_id].item())
            src_importance = float(node_importance[src].item())
            dst_importance = float(node_importance[dst].item())
            edge_importance = float(edge_importance_tensor[edge_id].item())
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "edge_id": edge_id,
                    "src_node": src,
                    "dst_node": dst,
                    "prev_mask": float((graph_mask[edge_id] + mask_delta[edge_id]).item()),
                    "current_mask": float(graph_mask[edge_id].item()),
                    "mask_delta": float(mask_delta[edge_id].item()),
                    "graph_score": float(self.last_graph_score.detach().float()[edge_id].item()),
                    "graph_utility": float(graph_utility[edge_id].item()),
                    "src_degree": float(degree[src].item()),
                    "dst_degree": float(degree[dst].item()),
                    "src_feature_norm": float(feature_norm[src].item()),
                    "dst_feature_norm": float(feature_norm[dst].item()),
                    "src_node_importance": src_importance,
                    "dst_node_importance": dst_importance,
                    "edge_importance": edge_importance,
                    "graph_keep": current_graph_keep,
                    "param_keep": current_param_keep,
                }
            )
        return rows

    @torch.no_grad()
    def write_memories(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        graph_utility = None
        param_utility = None
        if self.edge_logits.grad is not None and self.last_graph_mask is not None:
            graph_utility = (self.edge_logits.grad.detach() * self.last_graph_mask.detach()).abs()
            self.last_graph_utility = graph_utility
        if self.param_logits.grad is not None and self.last_param_mask is not None:
            param_utility = (self.param_logits.grad.detach() * self.last_param_mask.detach()).abs()
            self.last_param_utility = param_utility
        if not self.cfg.use_memory:
            return stats
        if self.cfg.memory_write_mode == "none":
            stats["memory_write_skipped"] = 1.0
            stats["memory_write_mode"] = 0.0
            return stats
        if graph_utility is not None:
            graph_cross_ctx = (
                self.last_param_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            )
            graph_ctx = self.edge_context(graph_cross_ctx)
            graph_stats = self.graph_memory.write(graph_ctx, graph_utility, mode=self.cfg.memory_write_mode)
            stats.update({f"graph_memory_{key}": value for key, value in graph_stats.items()})
            stats.update(self.write_graph_event_memory(graph_utility))
            stats.update(self.write_graph_recall_memory(graph_utility))
        if param_utility is not None:
            param_cross_ctx = (
                self.last_graph_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            )
            param_ctx = self.param_context(param_cross_ctx)
            param_stats = self.param_memory.write(param_ctx, param_utility, mode=self.cfg.memory_write_mode)
            stats.update({f"param_memory_{key}": value for key, value in param_stats.items()})
            stats.update(self.write_param_recall_memory(param_utility))
        stats.update(self.write_steering_memory())
        return stats
