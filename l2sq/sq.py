"""Superquadric utilities: rotations, frame changes, surface sampling, meshes, Chamfer.

A rigid superquadric (SuperFlex convention) has
    scale   (3,)   semi-axes s_x, s_y, s_z
    eps     (2,)   shape exponents (eps1 for eta / the z profile, eps2 for omega / the xy profile)
    R       (3,3)  rotation, local -> frame
    t       (3,)   translation
and surface points x = R @ x_local + t with
    x_local = (s_x * fc(eta, e1) * fc(omega, e2),
               s_y * fc(eta, e1) * fs(omega, e2),
               s_z * fs(eta, e1)),   eta in [-pi/2, pi/2], omega in [-pi, pi]
where fc(a, e) = sign(cos a) |cos a|^e and fs(a, e) = sign(sin a) |sin a|^e.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

EPS_MIN, EPS_MAX = 0.1, 1.9   # SuperFlex shape-exponent range (0.1 + 1.8 * sigmoid)
SCALE_MIN = 1e-3


# --------------------------------------------------------------------------- rotations

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D rotation representation (Zhou et al. 2019) -> rotation matrix. (..., 6) -> (..., 3, 3).

    The two 3-vectors are Gram-Schmidt orthonormalized and become the first two COLUMNS.
    """
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle of R1^T R2 in degrees. (..., 3, 3) -> (...)."""
    tr = (R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)))


# --------------------------------------------------------------------------- frame change

def to_view_frame(R: np.ndarray, t: np.ndarray, scale: np.ndarray, w2c: np.ndarray, s: float):
    """World-frame superquadrics -> VGGT target frame of one sample.

    Camera v0 frame via its w2c = [R0 | t0], then divided by the sample's normalization s:
        R' = R0 R,  t' = (R0 t + t0) / s,  scale' = scale / s   (exponents unchanged)
    Shapes: R (M,3,3), t (M,3), scale (M,3), w2c (3,4).
    """
    R0, t0 = w2c[:, :3], w2c[:, 3]
    return np.einsum("ij,mjk->mik", R0, R), (t @ R0.T + t0) / s, scale / s


# --------------------------------------------------------------------------- surface sampling

def _signed_pow(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().clamp_min(1e-6) ** e


def sq_grid(n_eta: int, n_omega: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameter grid (eta, omega), each flattened to (n_eta * n_omega,), eta-major."""
    eta = torch.linspace(-math.pi / 2, math.pi / 2, n_eta, device=device)
    omega = torch.linspace(-math.pi, math.pi, n_omega, device=device)
    e, o = torch.meshgrid(eta, omega, indexing="ij")
    return e.reshape(-1), o.reshape(-1)


def sample_surface(scale, eps, R, t, n_eta: int = 16, n_omega: int = 32) -> torch.Tensor:
    """Differentiable surface points. scale (...,3), eps (...,2), R (...,3,3), t (...,3) -> (..., K, 3)."""
    eta, omega = sq_grid(n_eta, n_omega, scale.device)
    e1, e2 = eps[..., 0:1], eps[..., 1:2]                       # (..., 1)
    ce, se = _signed_pow(torch.cos(eta), e1), _signed_pow(torch.sin(eta), e1)
    co, so = _signed_pow(torch.cos(omega), e2), _signed_pow(torch.sin(omega), e2)
    local = torch.stack([scale[..., 0:1] * ce * co, scale[..., 1:2] * ce * so, scale[..., 2:3] * se], -1)
    return local @ R.transpose(-1, -2) + t.unsqueeze(-2)


def sq_mesh(scale, eps, R, t, colors=None, n_eta: int = 24, n_omega: int = 48):
    """trimesh mesh of a set of superquadrics (numpy or torch inputs, leading dim = primitives)."""
    import trimesh

    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    pts = sample_surface(to_t(scale), to_t(eps), to_t(R), to_t(t), n_eta, n_omega).numpy()  # (M, K, 3)
    i, j = np.meshgrid(np.arange(n_eta - 1), np.arange(n_omega - 1), indexing="ij")
    a, b = i * n_omega + j, i * n_omega + j + 1
    c, d = (i + 1) * n_omega + j, (i + 1) * n_omega + j + 1
    quad = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([c, b, d], -1).reshape(-1, 3)])
    verts, faces, fcols = [], [], []
    palette = colors if colors is not None else _palette(len(pts))
    for m, p in enumerate(pts):
        faces.append(quad + m * p.shape[0])
        verts.append(p)
        fcols.append(np.tile(palette[m % len(palette)], (len(quad), 1)))
    return trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces),
                           face_colors=np.concatenate(fcols), process=False)


def _palette(n: int) -> np.ndarray:
    import colorsys
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(k / max(n, 1), 0.65, 0.95)] + [255]
                     for k in range(n)], dtype=np.uint8)


# --------------------------------------------------------------------------- metrics

def chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric L2 Chamfer (mean of the two mean nearest-neighbour distances). a (K,3), b (L,3)."""
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())
