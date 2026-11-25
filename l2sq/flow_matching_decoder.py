"""Flow-matching superquadric denoiser (standalone; TimestepEmbedder & Euler loop adapted from nova3r, Apache-2.0)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from l2sq.model import TOKEN_DIM, sinusoidal
from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, chamfer, grid_faces, rot6d_to_matrix, sample_surface

SQ_DIM = 15  # rigid: t(3), rot6d(6), log_scale(3), eps_raw(2), exist(1)
DEFORM_DIM = 26  # + taper(2), bend_logit(3), alpha_cs(6)
GROUP_SLICES = {"t": slice(0, 3), "R": slice(3, 9), "scale": slice(9, 12), "eps": slice(12, 14), "exist": slice(14, 15)}
DEFORM_GROUP_SLICES = {
    **GROUP_SLICES,
    "taper": slice(15, 17),
    "bend": slice(17, 20),
    "alpha": slice(20, 26),
}
# Shift/scale per dim to map unconstrained x_1 roughly to [-1, 1] matching N(0, 1) noise.
_SHIFT = [0.0] * 3 + [0.0] * 6 + [-2.5] * 3 + [0.0] * 2 + [0.5]
_SCALE = [1.0] * 3 + [1.0] * 6 + [2.0] * 3 + [2.0] * 2 + [0.5]
# Deformable dims: taper in [-1, 1] (0, 1); bend logit in [-9.2, ~0] (-4.5, 4.5); (cos, sin) in [-1, 1] (0, 1).
_DEFORM_SHIFT = [0.0] * 2 + [-4.5] * 3 + [0.0] * 6
_DEFORM_SCALE = [1.0] * 2 + [4.5] * 3 + [1.0] * 6


def _default_shift_scale(dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if dim == DEFORM_DIM:
        sh, sc = _SHIFT + _DEFORM_SHIFT, _SCALE + _DEFORM_SCALE
    else:
        sh, sc = _SHIFT, _SCALE
    return torch.tensor(sh, device=device, dtype=dtype), torch.tensor(sc, device=device, dtype=dtype)


def _shift_scale(like: torch.Tensor, shift: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if shift is not None and scale is not None:
        return shift.to(like), scale.to(like).clamp_min(1e-6)
    return _default_shift_scale(like.shape[-1], like.device, like.dtype)


def pack_sq(tgt: dict, normalize: bool = True,
            t_mean: torch.Tensor | None = None, t_std: torch.Tensor | None = None,
            shift: torch.Tensor | None = None, scale: torch.Tensor | None = None,
            deform: bool | None = None) -> torch.Tensor:
    """Pack target dict -> unconstrained (B, P, 15 or 26) flow target x_1 (optionally group-normalized)."""
    use_deform = ("taper" in tgt) if deform is None else deform
    R = tgt["R"]
    t_val = tgt["t"]
    if normalize and t_mean is not None and t_std is not None:
        t_val = (t_val - t_mean.to(t_val)) / t_std.to(t_val).clamp_min(1e-6)
    rot6d = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
    log_scale = torch.log(tgt["scale"].clamp_min(SCALE_MIN))
    eps_u = ((tgt["eps"] - EPS_MIN) / (EPS_MAX - EPS_MIN)).clamp(1e-4, 1.0 - 1e-4)
    parts = [t_val, rot6d, log_scale, torch.logit(eps_u), tgt["exist"].unsqueeze(-1)]
    if use_deform:
        taper = tgt["taper"].clamp(-1.0, 1.0)
        bend_logit = torch.logit(tgt["bend"].clamp(1e-4, 1.0 - 1e-4))
        alpha = tgt["alpha"]
        alpha_cs = torch.stack([alpha.cos(), alpha.sin()], dim=-1).flatten(-2)
        parts.extend([taper, bend_logit, alpha_cs])
    raw = torch.cat(parts, dim=-1)
    if not normalize:
        return raw
    sh, sc = _shift_scale(raw, shift, scale)
    return (raw - sh) / sc


def decode_sq(x: torch.Tensor, normalized: bool = True, smooth_exist: bool = False,
              t_mean: torch.Tensor | None = None, t_std: torch.Tensor | None = None,
              shift: torch.Tensor | None = None, scale: torch.Tensor | None = None,
              deform: bool | None = None) -> dict:
    """Convert flow sample (B, P, 15 or 26) back into valid SQ dict matching SQDecoder's output keys."""
    use_deform = (x.shape[-1] == DEFORM_DIM) if deform is None else deform
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    if normalized:
        sh, sc = _shift_scale(x, shift, scale)
        x = x * sc + sh
    t_out = x[..., 0:3]
    if normalized and t_mean is not None and t_std is not None:
        t_out = t_out * t_std.to(x).clamp_min(1e-6) + t_mean.to(x)
    # In flow space, GT existence is 0 or 1 (threshold 0.5). For differentiable BCE in mix loss,
    # map (exist - 0.5) * 8.0 to an unbounded logit centered at 0.5 so gradients never vanish.
    exist_logit = (x[..., 14] - 0.5) * 8.0 if smooth_exist else torch.logit(x[..., 14].clamp(1e-4, 1.0 - 1e-4))
    out = {
        "t": t_out,
        "R": rot6d_to_matrix(x[..., 3:9].clamp(-1e4, 1e4)),
        "scale": torch.exp(x[..., 9:12].clamp(-10.0, 4.0)).clamp_min(SCALE_MIN),
        "eps": EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(x[..., 12:14]),
        "exist_logit": exist_logit,
    }
    if use_deform:
        raw_cs = x[..., 20:26].unflatten(-1, (3, 2)).clamp(-1e4, 1e4)
        vec_cs = F.normalize(raw_cs, dim=-1, eps=1e-6)                       # (..., 3, 2) = (cos, sin)
        cs_ok = (raw_cs ** 2).sum(dim=-1, keepdim=True) >= 1e-12
        cs_safe = torch.where(cs_ok, vec_cs, vec_cs.new_tensor([1.0, 0.0]).expand_as(vec_cs))
        alpha = torch.atan2(cs_safe[..., 1], cs_safe[..., 0])
        alpha_vec = torch.stack([vec_cs[..., 1], vec_cs[..., 0]], dim=-1)    # (sin, cos) matching SQDecoder
        out.update(
            taper=x[..., 15:17].clamp(-1.0, 1.0),
            bend=torch.sigmoid(x[..., 17:20].clamp(-10.0, 10.0)),
            alpha_vec=alpha_vec,
            alpha=alpha,
        )
    return out


def flow_group_stats(x_raw: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, tuple[float, float]]:
    """Compute (min, max) per parameter group on unconstrained x_1."""
    m = (mask > 0.5) if mask is not None else None
    slices = DEFORM_GROUP_SLICES if x_raw.shape[-1] == DEFORM_DIM else GROUP_SLICES
    out = {}
    for name, sl in slices.items():
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
    """Predicts flow velocity v(x_t, t, tokens) of shape (B, P, 15 or 26) for CondOT flow matching."""

    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True,
                 slot_init: str = "small", fourier_freqs: int = 0, deform: bool = False):
        super().__init__()
        if fourier_freqs < 0:
            raise ValueError(f"fourier_freqs must be >= 0, got {fourier_freqs}")
        self.deform = deform
        self.sq_dim = DEFORM_DIM if deform else SQ_DIM
        self.fourier_freqs = fourier_freqs
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        in_dim = self.sq_dim * (1 + 2 * fourier_freqs)
        self.x_proj = nn.Linear(in_dim, d_model)
        if slot_init == "small":
            self.slot_emb = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        elif slot_init == "sinus":
            self.slot_emb = nn.Parameter(sinusoidal(n_queries, d_model).unsqueeze(0))
        else:
            raise ValueError(f"unknown slot_init {slot_init!r}, expected 'small' or 'sinus'")
        self.t_embedder = TimestepEmbedder(d_model)
        self.view_embedder = TimestepEmbedder(d_model)
        self.blocks = nn.ModuleList([FlowSQBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.sq_dim)
        self.x_skip = nn.Linear(self.sq_dim, self.sq_dim)
        self.t_scale = nn.Sequential(nn.SiLU(), nn.Linear(d_model, self.sq_dim))
        self.register_buffer("t_mean", torch.zeros(3))
        self.register_buffer("t_std", torch.ones(3))
        sh, sc = _default_shift_scale(self.sq_dim, torch.device("cpu"), torch.float32)
        self.register_buffer("flow_shift", sh)
        self.register_buffer("flow_scale", sc)

    def set_t_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        with torch.no_grad():
            self.t_mean.copy_(mean.to(device=self.t_mean.device, dtype=self.t_mean.dtype))
            self.t_std.copy_(std.to(device=self.t_std.device, dtype=self.t_std.dtype).clamp_min(1e-6))

    def set_flow_norm(self, shift: torch.Tensor, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.flow_shift.copy_(shift.to(device=self.flow_shift.device, dtype=self.flow_shift.dtype))
            self.flow_scale.copy_(scale.to(device=self.flow_scale.device, dtype=self.flow_scale.dtype).clamp_min(1e-6))

    def load_state_dict(self, state_dict, *args, **kwargs):
        sd = dict(state_dict)
        if "t_mean" not in sd and "t_std" not in sd:
            sd["t_mean"], sd["t_std"] = self.t_mean, self.t_std
        if "flow_shift" not in sd and "flow_scale" not in sd:
            sd["flow_shift"], sd["flow_scale"] = self.flow_shift, self.flow_scale
        return super().load_state_dict(sd, *args, **kwargs)

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
        if self.fourier_freqs > 0:
            feats = [x_t]
            for k in range(self.fourier_freqs):
                ang = x_t * (math.pi * (2.0 ** k))
                feats.append(torch.sin(ang))
                feats.append(torch.cos(ang))
            x_in = torch.cat(feats, dim=-1)
        else:
            x_in = x_t
        x = self.x_proj(x_in) + self.slot_emb[:, :x_t.shape[1]]
        for blk in self.blocks:
            x = blk(x, mem, cond)
        return self.out_proj(self.out_norm(x)) + self.t_scale(cond).unsqueeze(1) * self.x_skip(x_t)


def _sanitize_sq_for_sampling(sq: dict, deform: bool) -> dict:
    """Clamp/clean SQ tensors to finite sane ranges for surface sampling without NaN backward leaks."""
    scale = torch.nan_to_num(sq["scale"], nan=SCALE_MIN, posinf=50.0, neginf=1e-4).clamp(min=1e-4, max=50.0)
    eps = torch.nan_to_num(sq["eps"], nan=1.0, posinf=EPS_MAX, neginf=EPS_MIN).clamp(min=EPS_MIN, max=EPS_MAX)
    t = torch.nan_to_num(sq["t"], nan=0.0, posinf=50.0, neginf=-50.0).clamp(min=-50.0, max=50.0)
    R_clean = torch.nan_to_num(sq["R"], nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    eye = torch.eye(3, device=R_clean.device, dtype=R_clean.dtype).expand_as(R_clean)
    r_ok = torch.isfinite(sq["R"]).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    R = torch.where(r_ok, R_clean, eye)
    out = {"scale": scale, "eps": eps, "R": R, "t": t}
    if deform:
        taper = torch.nan_to_num(sq["taper"], nan=0.0, posinf=0.99, neginf=-0.99).clamp(-0.99, 0.99)
        bend = torch.nan_to_num(sq["bend"], nan=0.0, posinf=1.0 - 1e-4, neginf=0.0).clamp(0.0, 1.0 - 1e-4)
        alpha = torch.nan_to_num(sq["alpha"], nan=0.0, posinf=math.pi, neginf=-math.pi).clamp(-1e4, 1e4)
        out.update(taper=taper, bend=bend, alpha=alpha)
    return out


def _safe_sample_surface_uniform(
    scale: torch.Tensor, eps: torch.Tensor, R: torch.Tensor, t: torch.Tensor, n: int,
    taper: torch.Tensor | None = None, bend: torch.Tensor | None = None, alpha: torch.Tensor | None = None,
    n_eta: int = 24, n_omega: int = 48, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Area-uniform differentiable surface sampling with sanitized multinomial weights (no CUDA assert)."""
    grid = sample_surface(scale, eps, R, t, n_eta, n_omega, taper, bend, alpha)        # (..., K, 3)
    lead, K = grid.shape[:-2], grid.shape[-2]
    g = grid.reshape(-1, K, 3)                                                          # (Q, K, 3)
    faces = grid_faces(n_eta, n_omega, g.device)                                        # (F, 3)
    tri = g[:, faces]                                                                   # (Q, F, 3, 3)
    with torch.no_grad():
        area = torch.linalg.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]).norm(dim=-1)
        weights = torch.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
        invalid = ~torch.isfinite(weights).all(dim=-1) | (weights.sum(dim=-1) <= 0)
        if invalid.any():
            weights = torch.where(invalid[:, None], torch.ones_like(weights), weights)
        idx = torch.multinomial(weights, n, replacement=True, generator=generator)     # (Q, n)
        u, v = torch.rand(2, *idx.shape, 1, device=g.device, generator=generator)
        flip = (u + v) > 1
        u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    t3 = torch.gather(tri, 1, idx[..., None, None].expand(-1, -1, 3, 3))                # (Q, n, 3, 3)
    pts = t3[:, :, 0] + u * (t3[:, :, 1] - t3[:, :, 0]) + v * (t3[:, :, 2] - t3[:, :, 0])
    return pts.reshape(*lead, n, 3)


def _geo_surface_loss(
    pred_sq: dict, tgt: dict, wt: torch.Tensor, m: torch.Tensor, n: torch.Tensor,
    rep, deform: bool, geo_loss: str, repeat: int,
    n_geo_points: int = 128, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """t-weighted surface geometry loss on one-step clean estimates x1_hat ('points' or 'chamfer')."""
    if geo_loss == "points":
        if not (m > 0.5).any():
            return pred_sq["t"].sum() * 0.0
        p_clean = _sanitize_sq_for_sampling(pred_sq, deform)
        # Fixed deterministic 8x16 = 128-point (eta, omega) grid per SQ; preserves slot & point correspondence.
        deform_p = (p_clean["taper"], p_clean["bend"], p_clean["alpha"]) if deform else (None, None, None)
        p_pts = sample_surface(p_clean["scale"], p_clean["eps"], p_clean["R"], p_clean["t"],
                               8, 16, *deform_p)                                            # (B,P,128,3)
        with torch.no_grad():
            t_clean = _sanitize_sq_for_sampling({
                "scale": rep(tgt["scale"]), "eps": rep(tgt["eps"]),
                "R": rep(tgt["R"]), "t": rep(tgt["t"]),
                **({k: rep(tgt[k]) for k in ("taper", "bend", "alpha")} if deform else {}),
            }, deform)
            deform_g = (t_clean["taper"], t_clean["bend"], t_clean["alpha"]) if deform else (None, None, None)
            g_pts = sample_surface(t_clean["scale"], t_clean["eps"], t_clean["R"], t_clean["t"],
                                   8, 16, *deform_g)                                        # (B,P,128,3)
        d_slot = ((p_pts - g_pts) ** 2).sum(dim=-1).mean(dim=-1)                           # (B,P)
        return ((d_slot * wt) * m).sum() / n

    if geo_loss == "chamfer":
        active = m > 0.5
        if not active.any():
            return pred_sq["t"].sum() * 0.0
        p_clean = _sanitize_sq_for_sampling(pred_sq, deform)
        deform_p = (p_clean["taper"][active], p_clean["bend"][active], p_clean["alpha"][active]) if deform else (None, None, None)
        p_flat = _safe_sample_surface_uniform(
            p_clean["scale"][active], p_clean["eps"][active], p_clean["R"][active], p_clean["t"][active],
            n_geo_points, *deform_p, generator=generator,
        )                                                                                   # (N_act, S, 3)
        counts = active.sum(dim=1).tolist()
        p_per_b = torch.split(p_flat, counts, dim=0)
        has_surf = "surf" in tgt
        if has_surf:
            g_surf = rep(tgt["surf"])
        else:
            with torch.no_grad():
                t_clean = _sanitize_sq_for_sampling({
                    "scale": rep(tgt["scale"]), "eps": rep(tgt["eps"]),
                    "R": rep(tgt["R"]), "t": rep(tgt["t"]),
                    **({k: rep(tgt[k]) for k in ("taper", "bend", "alpha")} if deform else {}),
                }, deform)
                deform_g = (t_clean["taper"][active], t_clean["bend"][active], t_clean["alpha"][active]) if deform else (None, None, None)
                g_flat = _safe_sample_surface_uniform(
                    t_clean["scale"][active], t_clean["eps"][active], t_clean["R"][active], t_clean["t"][active],
                    n_geo_points, *deform_g, generator=generator,
                )
                g_per_b = torch.split(g_flat, counts, dim=0)
        s_vec = rep(tgt["s"]) if "s" in tgt else None
        losses = []
        for b, cnt in enumerate(counts):
            if cnt == 0:
                continue
            p_b = p_per_b[b].reshape(-1, 3)
            g_b = g_surf[b] if has_surf else g_per_b[b].reshape(-1, 3)
            sb = s_vec[b] if s_vec is not None else 1.0
            losses.append(wt[b, 0] * chamfer(p_b, g_b) * sb)
        return torch.stack(losses).mean() if losses else pred_sq["t"].sum() * 0.0

    raise ValueError(f"unknown geo_loss {geo_loss!r}")


def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None, repeat: int = 1,
              mode: str = "velocity", w_x1: float = 1.0,
              vel_loss: str = "mse", geo_loss: str = "none",
              w_geo: float = 1.0, n_geo_points: int = 128) -> tuple[torch.Tensor, dict]:
    """CondOT velocity loss ('mse' or Euclidean 'l2'), optionally mixed with t-weighted decoded x1_hat MSE + BCE."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if mode not in ("velocity", "mix"):
        raise ValueError(f"unknown flow loss mode {mode!r}")
    if vel_loss not in ("mse", "l2"):
        raise ValueError(f"unknown vel_loss {vel_loss!r}")
    if geo_loss not in ("none", "points", "chamfer"):
        raise ValueError(f"unknown geo_loss {geo_loss!r}")
    mem, n_views = denoiser.encode_tokens(tokens)
    x_1 = pack_sq(tgt, t_mean=denoiser.t_mean, t_std=denoiser.t_std,
                  shift=denoiser.flow_shift, scale=denoiser.flow_scale, deform=denoiser.deform)
    m = tgt["exist"]
    if repeat > 1:
        mem = mem.repeat_interleave(repeat, dim=0)
        x_1 = x_1.repeat_interleave(repeat, dim=0)
        m = m.repeat_interleave(repeat, dim=0)
    x_0 = torch.randn(x_1.shape, device=x_1.device, dtype=x_1.dtype, generator=generator)
    t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype, generator=generator)
    x_t = t[:, None, None] * x_1 + (1.0 - t[:, None, None]) * x_0
    v_pred = denoiser(x_t, t, mem=mem, n_views=n_views)
    err2 = (v_pred - (x_1 - x_0)) ** 2
    n = m.sum().clamp_min(1.0)
    slices = DEFORM_GROUP_SLICES if denoiser.deform else GROUP_SLICES
    if vel_loss == "l2":
        parts = {k: (torch.sqrt(err2[..., sl].sum(dim=-1) + 1e-8) * m).sum() / n
                 for k, sl in slices.items() if k != "exist"}
        parts["exist"] = torch.sqrt(err2[..., 14:15].sum(dim=-1) + 1e-8).mean()
    else:
        parts = {k: (err2[..., sl].mean(-1) * m).sum() / n for k, sl in slices.items() if k != "exist"}
        parts["exist"] = err2[..., 14].mean()
    v_loss = sum(parts.values())
    if mode == "velocity":
        return v_loss, {k: v.item() for k, v in parts.items()}

    # Weight x1 parameter loss by t so noisy samples near t=0 (where x1_hat is unreliable) do not dominate.
    x1_hat = x_t + (1.0 - t[:, None, None]) * v_pred
    pred_sq = decode_sq(x1_hat, normalized=True, smooth_exist=True,
                        t_mean=denoiser.t_mean, t_std=denoiser.t_std,
                        shift=denoiser.flow_shift, scale=denoiser.flow_scale, deform=denoiser.deform)
    wt = t[:, None]
    rep = (lambda a: a.repeat_interleave(repeat, dim=0)) if repeat > 1 else (lambda a: a)
    x1_masked = lambda a, b: ((((a - b) ** 2).flatten(2).mean(-1) * wt) * m).sum() / n
    x1_loss = (
        x1_masked(pred_sq["t"], rep(tgt["t"]))
        + x1_masked(pred_sq["R"], rep(tgt["R"]))
        + x1_masked(pred_sq["scale"].log(), rep(tgt["scale"]).log())
        + x1_masked(pred_sq["eps"], rep(tgt["eps"]))
        + (F.binary_cross_entropy_with_logits(pred_sq["exist_logit"], m, reduction="none") * wt).mean()
    )
    extra_x1: dict[str, float] = {}
    if denoiser.deform:
        bend_tgt = rep(tgt["bend"])
        bend_logit_tgt = torch.logit(bend_tgt.clamp(1e-4, 1.0 - 1e-4))
        bend_logit_pred = torch.logit(pred_sq["bend"].clamp(1e-4, 1.0 - 1e-4))
        tvec = torch.stack([rep(tgt["alpha"]).sin(), rep(tgt["alpha"]).cos()], -1)
        x1_taper = x1_masked(pred_sq["taper"], rep(tgt["taper"]))
        x1_bend = x1_masked(bend_logit_pred, bend_logit_tgt)
        d2_alpha = ((pred_sq["alpha_vec"] - tvec) ** 2).sum(-1) * bend_tgt
        x1_alpha = ((d2_alpha.mean(-1) * wt) * m).sum() / n
        x1_loss = x1_loss + x1_taper + x1_bend + x1_alpha
        extra_x1 = {"x1_taper": x1_taper.item(), "x1_bend": x1_bend.item(), "x1_alpha": x1_alpha.item()}
    total = v_loss + w_x1 * x1_loss
    if geo_loss != "none":
        geo = _geo_surface_loss(pred_sq, tgt, wt, m, n, rep, denoiser.deform, geo_loss, repeat,
                                n_geo_points=n_geo_points, generator=generator)
        total = total + w_geo * geo
        extra_x1["geo"] = geo.item()
    out = {k: v.item() for k, v in parts.items()}
    out.update(v_loss=v_loss.item(), x1_loss=x1_loss.item(), **extra_x1)
    return total, out


# Adapted from nova3r (Apache-2.0): BatchModelWrapper + Euler ODESolver loop (model_wrapper.py, ode_solver.py).
@torch.no_grad()
def sample_euler(denoiser: FlowSQDenoiser, tokens: torch.Tensor, n_queries: int = 16,
                 n_steps: int = 20, generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate CondOT ODE from t=0 (N(0,1)) to t=1 using n_steps Euler steps."""
    mem, n_views = denoiser.encode_tokens(tokens)
    x = torch.randn((tokens.shape[0], n_queries, denoiser.sq_dim), device=tokens.device, dtype=tokens.dtype, generator=generator)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((tokens.shape[0],), k * dt, device=tokens.device, dtype=tokens.dtype)
        x = x + dt * denoiser(x, t, mem=mem, n_views=n_views)
    return x
