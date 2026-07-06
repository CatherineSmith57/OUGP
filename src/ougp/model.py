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


@dataclass(frozen=True)
class OUGPConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_nodes: int
    num_edges: int
    edge_context_dim: int = 15
    param_context_dim: int = 6
    memory_rank: int = 8
    feature_context_dim: int = 6
    graph_target_keep: float = 0.70
    param_target_keep: float = 0.70
    graph_gamma: float = 0.35
    param_gamma: float = 0.35
    cross_gamma: float = 0.20
    write_beta: float = 0.12
    write_lambda: float = 0.98
    event_gamma: float = 0.0
    event_beta: float = 0.10
    event_decay: float = 0.95
    event_top_k: int = 2000
    hard_masks: bool = False
    use_graph_pruning: bool = True
    use_param_pruning: bool = True
    use_memory: bool = True
    use_cross: bool = True
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
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.event_beta = event_beta
        self.event_decay = event_decay
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.read_head = nn.Linear(rank, 1, bias=False)
        self.utility_head = nn.Linear(rank, 1, bias=False)
        self.register_buffer("state", torch.zeros(rank, rank))
        self.register_buffer("event_bias", torch.zeros(max(0, event_items)))

    def reset_state(self) -> None:
        self.state.zero_()
        self.event_bias.zero_()

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
    def write(self, context: torch.Tensor, utility: torch.Tensor) -> dict[str, float]:
        if context.numel() == 0:
            return {"utility_mean": 0.0, "residual_mean": 0.0, "state_norm": float(self.state.norm().item())}
        _, k, v = self.project_qkv(context.detach())
        pred_vec = k @ self.state.t()
        pred = self.utility_head(pred_vec).squeeze(-1)
        utility = utility.detach().float()
        utility = (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)
        residual = utility - pred
        target_v = v * residual.tanh().unsqueeze(-1)
        current = k @ self.state.t()
        delta_vec = target_v - current
        delta_state = torch.einsum("br,bk->rk", delta_vec, k) / max(1, context.size(0))
        self.state.mul_(self.write_lambda).add_(self.write_beta * delta_state)
        self.state.clamp_(-5.0, 5.0)
        return {
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


class OUGPGCN(nn.Module):
    """Two-layer GCN with online graph and channel pruning memories."""

    def __init__(self, cfg: OUGPConfig, edge_index: torch.Tensor, x: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("base_edge_index", edge_index)
        self.register_buffer("x_ref", x)
        self.lin1 = nn.Linear(cfg.in_dim, cfg.hidden_dim, bias=False)
        self.lin2 = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        self.edge_logits = nn.Parameter(torch.full((cfg.num_edges,), 2.0))
        self.param_logits = nn.Parameter(torch.full((cfg.hidden_dim,), 2.0))

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
        )
        self.param_memory = OnlinePruningMemory(
            cfg.param_context_dim,
            cfg.memory_rank,
            cfg.write_beta,
            cfg.write_lambda,
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

    def reset_memory(self) -> None:
        self.graph_memory.reset_state()
        self.param_memory.reset_state()

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
        channel_id = torch.linspace(0, 1, self.cfg.hidden_dim, device=w1_norm.device).unsqueeze(-1)
        sparsity = torch.sigmoid(self.param_logits).detach().unsqueeze(-1)
        graph_ctx = graph_keep.expand(self.cfg.hidden_dim, 1)
        bias = torch.ones_like(graph_ctx)
        return torch.cat([w1_norm, w2_norm, channel_id, sparsity, graph_ctx, bias], dim=-1)

    def masks(self, temperature: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        base_graph_keep = torch.sigmoid(self.edge_logits).mean().detach()
        base_param_keep = torch.sigmoid(self.param_logits).mean().detach()

        if self.cfg.use_memory:
            graph_cross_ctx = base_param_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            param_cross_ctx = base_graph_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            graph_ctx = self.edge_context(graph_cross_ctx)
            param_ctx = self.param_context(param_cross_ctx)
            graph_corr, _, _, _ = self.graph_memory.read(graph_ctx)
            param_corr, _, _, _ = self.param_memory.read(param_ctx)
        else:
            graph_corr = torch.zeros_like(self.edge_logits)
            param_corr = torch.zeros_like(self.param_logits)

        graph_score = self.edge_logits
        param_score = self.param_logits

        if self.cfg.use_memory:
            graph_score = graph_score + self.cfg.graph_gamma * graph_corr
            param_score = param_score + self.cfg.param_gamma * param_corr
            if self.cfg.use_graph_pruning and self.cfg.event_gamma != 0.0:
                graph_score = graph_score + self.cfg.event_gamma * self.graph_memory.event_correction(graph_score)
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
        }
        return graph_mask, param_mask, stats

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        graph_mask, param_mask, stats = self.masks(temperature)
        num_nodes = x.size(0)
        self_loops = torch.arange(num_nodes, device=x.device)
        self_loop_index = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([self.base_edge_index, self_loop_index], dim=1)
        edge_weight = torch.cat([graph_mask, torch.ones(num_nodes, device=x.device)])
        norm_weight = symmetric_norm(edge_index, edge_weight, num_nodes)

        h = sparse_gcn_mm(edge_index, norm_weight, x, num_nodes)
        h = self.lin1(h)
        h = F.relu(h) * param_mask
        h = F.dropout(h, p=0.5, training=self.training)
        h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
        out = self.lin2(h)
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
        if graph_utility is not None:
            graph_cross_ctx = (
                self.last_param_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            )
            graph_ctx = self.edge_context(graph_cross_ctx)
            graph_stats = self.graph_memory.write(graph_ctx, graph_utility)
            stats.update({f"graph_memory_{key}": value for key, value in graph_stats.items()})
            stats.update(self.write_graph_event_memory(graph_utility))
        if param_utility is not None:
            param_cross_ctx = (
                self.last_graph_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            )
            param_ctx = self.param_context(param_cross_ctx)
            param_stats = self.param_memory.write(param_ctx, param_utility)
            stats.update({f"param_memory_{key}": value for key, value in param_stats.items()})
        return stats
