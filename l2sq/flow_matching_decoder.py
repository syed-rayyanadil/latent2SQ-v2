"""Flow-matching superquadric denoiser (standalone; TimestepEmbedder & Euler loop adapted from nova3r, Apache-2.0)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from l2sq.model import TOKEN_DIM
from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, rot6d_to_matrix

SQ_DIM = 15  # t(3), rot6d(6), log_scale(3), eps_raw(2), exist(1)
GROUP_SLICES = {"t": slice(0, 3), "R": slice(3, 9), "scale": slice(9, 12), "eps": slice(12, 14), "exist": slice(14, 15)}
# Shift/scale per dim to map unconstrained x_1 roughly to [-1, 1] matching N(0, 1) noise.
_SHIFT = [0.0] * 3 + [0.0] * 6 + [-2.5] * 3 + [0.0] * 2 + [0.5]
_SCALE = [1.0] * 3 + [1.0] * 6 + [2.0] * 3 + [2.0] * 2 + [0.5]


def _shift_scale(like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(_SHIFT, device=like.device, dtype=like.dtype),
            torch.tensor(_SCALE, device=like.device, dtype=like.dtype))


def pack_sq(tgt: dict, normalize: bool = True) -> torch.Tensor:
    """Pack target dict -> unconstrained (B, P, 15) flow target x_1 (optionally group-normalized)."""
    R = tgt["R"]
    rot6d = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
    log_scale = torch.log(tgt["scale"].clamp_min(SCALE_MIN))
    eps_u = ((tgt["eps"] - EPS_MIN) / (EPS_MAX - EPS_MIN)).clamp(1e-4, 1.0 - 1e-4)
    raw = torch.cat([tgt["t"], rot6d, log_scale, torch.logit(eps_u), tgt["exist"].unsqueeze(-1)], dim=-1)
    if not normalize:
        return raw
    shift, scale = _shift_scale(raw)
    return (raw - shift) / scale


def decode_sq(x: torch.Tensor, normalized: bool = True) -> dict:
    """Convert flow sample (B, P, 15) back into valid SQ dict matching SQDecoder's output keys."""
    if normalized:
        shift, scale = _shift_scale(x)
        x = x * scale + shift
    return {
        "t": x[..., 0:3],
        "R": rot6d_to_matrix(x[..., 3:9]),
        "scale": torch.exp(x[..., 9:12].clamp(-10.0, 4.0)).clamp_min(SCALE_MIN),
        "eps": EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(x[..., 12:14]),
        "exist_logit": torch.logit(x[..., 14].clamp(1e-4, 1.0 - 1e-4)),  # >0 iff exist > 0.5
    }


def flow_group_stats(x_raw: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, tuple[float, float]]:
    """Compute (min, max) per parameter group on unconstrained x_1."""
    m = (mask > 0.5) if mask is not None else None
    out = {}
    for name, sl in GROUP_SLICES.items():
        vals = x_raw[..., sl] if (m is None or name == "exist") else x_raw[..., sl][m]
        out[name] = (float(vals.min()), float(vals.max()))
    return out


# Adapted from nova3r (Apache-2.0): flowm_decoder_point_joint_v2.py (modulate & TimestepEmbedder).
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar embedding + 2-layer SiLU MLP (adapted from nova3r, Apache-2.0)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.reshape(-1, 1).float() * 1000.0) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class FlowSQBlock(nn.Module):
    """SQ self-attn (adaLN) -> cross-attn to VGGT tokens -> FFN (adaLN)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        x = x + g1.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.norm_kv(mem)
        x = x + self.cross_attn(self.norm_q(x), kv, kv, need_weights=False)[0]
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class FlowSQDenoiser(nn.Module):
    """Predicts flow velocity v(x_t, t, tokens) of shape (B, P, 15) for CondOT flow matching."""

    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True):
        super().__init__()
        self.deform = False
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        self.x_proj = nn.Linear(SQ_DIM, d_model)
        self.slot_emb = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.t_embedder = TimestepEmbedder(d_model)
        self.view_embedder = TimestepEmbedder(d_model)
        self.blocks = nn.ModuleList([FlowSQBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, SQ_DIM)
        self.x_skip = nn.Linear(SQ_DIM, SQ_DIM)
        self.t_scale = nn.Sequential(nn.SiLU(), nn.Linear(d_model, SQ_DIM))

    def encode_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        mem = self.in_proj(self.in_norm(tokens))
        if self.ref_embedding is not None:
            mem = torch.cat([mem[:, :1] + self.ref_embedding, mem[:, 1:]], dim=1)
        return mem.flatten(1, 2), tokens.shape[1]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor | None = None,
                mem: torch.Tensor | None = None, n_views: int | None = None) -> torch.Tensor:
        if mem is None:
            mem, n_views = self.encode_tokens(tokens)
        cond = self.t_embedder(t)
        if n_views is not None:
            cond = cond + self.view_embedder(torch.full_like(t, float(n_views) / 8.0))
        x = self.x_proj(x_t) + self.slot_emb[:, :x_t.shape[1]]
        for blk in self.blocks:
            x = blk(x, mem, cond)
        return self.out_proj(self.out_norm(x)) + self.t_scale(cond).unsqueeze(1) * self.x_skip(x_t)


def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """CondOT velocity MSE: active slots for SQ params, all P slots for existence."""
    x_1 = pack_sq(tgt)
    x_0 = torch.randn(x_1.shape, device=x_1.device, dtype=x_1.dtype, generator=generator)
    t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype, generator=generator)
    x_t = t[:, None, None] * x_1 + (1.0 - t[:, None, None]) * x_0
    v_pred = denoiser(x_t, t, tokens=tokens)
    err2 = (v_pred - (x_1 - x_0)) ** 2
    m = tgt["exist"]
    n = m.sum().clamp_min(1.0)
    parts = {k: (err2[..., sl].mean(-1) * m).sum() / n for k, sl in GROUP_SLICES.items() if k != "exist"}
    parts["exist"] = err2[..., 14].mean()
    total = sum(parts.values())
    return total, {k: v.item() for k, v in parts.items()}


# Adapted from nova3r (Apache-2.0): BatchModelWrapper + Euler ODESolver loop (model_wrapper.py, ode_solver.py).
@torch.no_grad()
def sample_euler(denoiser: FlowSQDenoiser, tokens: torch.Tensor, n_queries: int = 16,
                 n_steps: int = 20, generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate CondOT ODE from t=0 (N(0,1)) to t=1 using n_steps Euler steps."""
    mem, n_views = denoiser.encode_tokens(tokens)
    x = torch.randn((tokens.shape[0], n_queries, SQ_DIM), device=tokens.device, dtype=tokens.dtype, generator=generator)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((tokens.shape[0],), k * dt, device=tokens.device, dtype=tokens.dtype)
        x = x + dt * denoiser(x, t, mem=mem, n_views=n_views)
    return x
