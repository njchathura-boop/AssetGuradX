"""Small edge/type-aware GATv2-style encoder plus a shared temporal GRU.

Implemented in PyTorch so the CPU pilot needs no optional graph extension
wheels. Attention is an internal model weight, not a causal loss coefficient.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Normalizer:
    """Fit numeric statistics only on valid observations in the training period."""
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, data, start, end):
        a = data.arrays
        selected = (a["times"] >= start) & (a["times"] < end)
        if not selected.any():
            raise ValueError("no training observations for preprocessing")
        values, mask = a["node_values"][selected], a["node_mask"][selected]
        count = mask.sum(axis=(0, 1))
        mean = (values * mask).sum(axis=(0, 1), dtype=np.float64) / np.maximum(count, 1)
        variance = (((values - mean) ** 2) * mask).sum(axis=(0, 1)) / np.maximum(count, 1)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1
        e_mean, e_std = [], []
        for relation in range(len(data.metadata["relations"])):
            edge_ids = a["edge_type"] == relation
            ev = a["edge_values"][selected][:, edge_ids, 0]
            em = a["edge_mask"][selected][:, edge_ids]
            observed = ev[em].astype(np.float64)
            e_mean.append(float(observed.mean()) if len(observed) else 0)
            deviation = float(observed.std()) if len(observed) else 1
            e_std.append(deviation if deviation >= 1e-6 else 1)
        return cls({"node_mean": mean.tolist(), "node_std": std.tolist(),
                    "edge_mean": e_mean, "edge_std": e_std, "fitted_start": start, "fitted_end_exclusive": end})

    def transform(self, data):
        a, s = data.arrays, self.state
        mask = a["node_mask"]
        normalized = ((a["node_values"] - np.asarray(s["node_mean"])) / np.asarray(s["node_std"])) * mask
        node = np.concatenate((normalized, mask, a["node_age"]), axis=-1).astype(np.float32)
        relation = a["edge_type"]
        value = ((a["edge_values"][:, :, 0] - np.asarray(s["edge_mean"])[relation])
                 / np.asarray(s["edge_std"])[relation]) * a["edge_mask"]
        edge = np.stack((value, a["edge_values"][:, :, 1], a["edge_mask"]), axis=-1).astype(np.float32)
        if not np.isfinite(node).all() or not np.isfinite(edge).all():
            raise ValueError("non-finite model inputs after preprocessing")
        return node, edge


class RelationalAttention(nn.Module):
    def __init__(self, hidden, heads, relation_count, dropout=0.1, *, attention=True):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.hidden, self.heads, self.width = hidden, heads, hidden // heads
        self.attention = attention
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.target = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Linear(3, hidden, bias=False)
        self.relation = nn.Embedding(relation_count, hidden)
        self.score = nn.Parameter(torch.empty(heads, self.width))
        nn.init.xavier_uniform_(self.score)
        self.output = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, edge_attr, edge_mask, *, return_attention=False):
        # x: [batch*time, nodes, hidden]. Each hour has its own edge mask/values.
        batch, nodes, _ = x.shape
        source, target = edge_index
        edges = len(source)
        source_features = self.source(x[:, source]).reshape(batch, edges, self.heads, self.width)
        target_features = self.target(x[:, target]).reshape(batch, edges, self.heads, self.width)
        context = (self.edge(edge_attr) + self.relation(edge_type)[None]).reshape(batch, edges, self.heads, self.width)
        scores = (F.leaky_relu(source_features + target_features + context, 0.2) * self.score).sum(-1)
        if not self.attention:
            scores = torch.zeros_like(scores)
        scores = scores.masked_fill(~edge_mask[..., None], -1e9)
        groups = target[None, :, None].expand(batch, edges, self.heads)
        maximum = torch.full((batch, nodes, self.heads), -1e9, dtype=x.dtype, device=x.device)
        maximum.scatter_reduce_(1, groups, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maximum.gather(1, groups)) * edge_mask[..., None]
        denominator = torch.zeros_like(maximum).scatter_add_(1, groups, weights)
        weights = weights / denominator.gather(1, groups).clamp_min(1e-12)
        messages = (source_features + context) * weights[..., None]
        output = torch.zeros((batch, nodes, self.heads, self.width), dtype=x.dtype, device=x.device)
        output.scatter_add_(1, groups[..., None].expand_as(messages), messages)
        encoded = F.gelu(self.norm(x + self.dropout(self.output(output.flatten(2)))))
        return (encoded, weights) if return_attention else encoded


class TemporalGAT(nn.Module):
    def __init__(self, *, node_features, node_kinds, relation_count, hidden=32, heads=4,
                 layers=2, dropout=0.1, architecture="gat"):
        super().__init__()
        if architecture not in {"gat", "mean", "gru"} or layers not in {1, 2} or not 0 <= dropout < 1:
            raise ValueError("invalid forecasting architecture")
        self.settings = dict(node_features=node_features, node_kinds=node_kinds,
                             relation_count=relation_count, hidden=hidden, heads=heads,
                             layers=layers, dropout=dropout, architecture=architecture)
        self.features = nn.Linear(node_features, hidden)
        self.kind = nn.Embedding(node_kinds, hidden)
        self.graph_layers = nn.ModuleList([
            RelationalAttention(hidden, heads, relation_count, dropout, attention=architecture == "gat")
            for _ in range(layers if architecture != "gru" else 0)
        ])
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        # Five stablecoins, two LSTs and the sole BTC wrapper share one encoder.
        # Heads stay small; no learned expert router or independent asset nets.
        self.family_heads = nn.ModuleList([nn.Linear(hidden, 9) for _ in range(3)])

    def forward(self, node, edge, edge_mask, edge_index, edge_type, node_type, target_index, family_index):
        batch, steps, nodes, _ = node.shape
        x = F.gelu(self.features(node) + self.kind(node_type)[None, None])
        x = x.reshape(batch * steps, nodes, -1)
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_type, edge.reshape(batch * steps, edge.shape[2], 3),
                      edge_mask.reshape(batch * steps, edge_mask.shape[2]))
        x = x.reshape(batch, steps, nodes, -1)[:, :, target_index]
        x = x.permute(0, 2, 1, 3).reshape(batch * len(target_index), steps, -1)
        _, last = self.temporal(x)
        last = last[0].reshape(batch, len(target_index), -1)
        all_heads = torch.stack([head(last) for head in self.family_heads], dim=2)
        selected = all_heads[:, torch.arange(len(target_index), device=node.device), family_index]
        selected = selected.reshape(batch, len(target_index), 3, 3)
        return selected[..., 0], torch.cumsum(F.softplus(selected[..., 1:]), dim=-2) / 100


def onset_probabilities(hazard_logits, temperature=1.0):
    """Cumulative interval hazards guarantee p6 <= p12 <= p24."""
    return -torch.expm1(torch.cumsum(F.logsigmoid(-hazard_logits / temperature), dim=-1))
