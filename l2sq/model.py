"""Frozen VGGT encoder + trainable superquadric decoder (instructions/pipeline.JPG, sq_decoder_brief.md §4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, rot6d_to_matrix

TOKEN_DIM = 2048  # VGGT aggregator: frame- and global-attention outputs concatenated (2 x 1024)


class FrozenVGGT(nn.Module):
    """VGGT-1B aggregator only (no heads), eval mode, no gradients, bf16 autocast.

    Returns tokens (B, N, T, 2048) of one aggregator layer; T = 5 + 1369 at 518x518
    (1 camera + 4 register tokens, then patch tokens from index patch_start_idx = 5).
    """

    def __init__(self, layer: int = 23):
        super().__init__()
        from recon.vggt_utils import load_vggt

        vggt = load_vggt()
        self.aggregator = vggt.aggregator
        del vggt
        self.aggregator.eval().requires_grad_(False)
        if layer not in self.aggregator.cached_layer_indices:
            raise ValueError(f"layer {layer} not cached; available {sorted(self.aggregator.cached_layer_indices)}")
        self.layer = layer

    def train(self, mode: bool = True):  # stays in eval mode whatever the parent does
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            tokens_list, _ = self.aggregator(images)
        return tokens_list[self.layer].float()


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class SQDecoder(nn.Module):
    """LayerNorm + Linear(2048 -> D) once, then L post-norm decoder layers
    (self-attn -> cross-attn to the shared tokens -> FFN, each with Add & Norm), then a head.

    Head per query (15 raw values): translation 3, rotation 6D 6, scale 3, exponents 2, existence 1.
    deform=True adds 11: tapering 2 (tanh), bending amount 3 (sigmoid, axes z, x, y) and bending
    direction 3 x 2D vectors (angle = atan2), as in SuperFlex.
    """

    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 1, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True, deform: bool = False):
        super().__init__()
        self.deform = deform
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        # marks the reference view (frame 0); no per-view index embeddings -> permutation-invariant over views 2..N
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        self.queries = nn.Parameter(sinusoidal(n_queries, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=False)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 26 if deform else 15)

    def forward(self, tokens: torch.Tensor) -> dict:
        B, N, T, _ = tokens.shape
        mem = self.in_proj(self.in_norm(tokens))                     # (B,N,T,D)
        if self.ref_embedding is not None:
            mem = torch.cat([mem[:, :1] + self.ref_embedding, mem[:, 1:]], dim=1)
        mem = mem.flatten(1, 2)                                       # (B,N*T,D)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, mem)
        raw = self.head(q)                                            # (B,P,15 or 26)
        out = {
            "t": raw[..., 0:3],
            "R": rot6d_to_matrix(raw[..., 3:9]),
            "scale": F.softplus(raw[..., 9:12]) + SCALE_MIN,
            "eps": EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(raw[..., 12:14]),
            "exist_logit": raw[..., 14],
        }
        if self.deform:
            vec = F.normalize(raw[..., 20:26].unflatten(-1, (3, 2)), dim=-1)   # (B,P,3,2) = (sin, cos)
            out.update(taper=torch.tanh(raw[..., 15:17]), bend=torch.sigmoid(raw[..., 17:20]),
                       alpha_vec=vec, alpha=torch.atan2(vec[..., 0], vec[..., 1]))
        return out
