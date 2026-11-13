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

Deformable superquadrics (SuperFlex) also deform x_local before R, t:
    tapering (2,)  kx, ky in [-1, 1]:  x *= 1 + kx z / s_z,  y *= 1 + ky z / s_z
    bending        per axis in the order (z, x, y): curvature kb >= 0 (1/length) and direction
                   angle alpha; applied around y, then x, then z (SuperFlex stores them interleaved
                   as bending (6,) = [kb_z, a_z, kb_x, a_x, kb_y, a_y]).
Here bending is stored as the dimensionless amount bend = kb * min_radius(scale, taper) in [0, 1]
(SuperFlex's own parameterization, kb = sigmoid(raw) / min_radius), which does not change when
the superquadric is rescaled -> no extra handling in to_view_frame.
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


def min_radius(scale, taper):
    """Smallest allowed bending radius per axis, in the bending order (z, x, y) (SuperFlex
    SuperQOptimization._compute_min_radius). Works for numpy and torch. (...,3), (...,2) -> (...,3)."""
    lib = torch if isinstance(scale, torch.Tensor) else np
    sx = scale[..., 0] * (1 + abs(taper[..., 0]))
    sy = scale[..., 1] * (1 + abs(taper[..., 1]))
    sz = scale[..., 2]
    thick = [lib.maximum(sy, sz), lib.maximum(sx, sz), lib.maximum(sx, sy)]      # x, y, z
    loop = [sx / math.pi, sy / math.pi, sz / math.pi]
    rx, ry, rz = (lib.maximum(a, b) for a, b in zip(thick, loop))
    return lib.stack([rz, rx, ry], -1)


def _bend_axis(x, y, z, kb, alpha, axis: str):
    """SuperFlex bending around one axis (loss/utils.py parametric_to_points_extended)."""
    u, v, w = {"z": (x, y, z), "x": (y, z, x), "y": (z, x, y)}[axis]
    beta = torch.atan2(v, u)
    r = torch.sqrt(u ** 2 + v ** 2 + 1e-12) * torch.cos(alpha - beta)
    on = kb.abs() >= 1e-3
    kb = torch.where(on, kb, torch.full_like(kb, 1e-3))
    inv = 1.0 / kb
    gamma = w * kb
    rho = inv - r
    d = inv - rho * torch.cos(gamma) - r
    u2, v2, w2 = u + d * torch.cos(alpha), v + d * torch.sin(alpha), rho * torch.sin(gamma)
    u, v, w = torch.where(on, u2, u), torch.where(on, v2, v), torch.where(on, w2, w)
    return {"z": (u, v, w), "x": (w, u, v), "y": (v, w, u)}[axis]


def sample_surface(scale, eps, R, t, n_eta: int = 16, n_omega: int = 32,
                   taper=None, bend=None, alpha=None) -> torch.Tensor:
    """Differentiable surface points. scale (...,3), eps (...,2), R (...,3,3), t (...,3) -> (..., K, 3).
    Optional deformation: taper (...,2), bend (...,3) dimensionless, alpha (...,3), both in (z, x, y)."""
    eta, omega = sq_grid(n_eta, n_omega, scale.device)
    e1, e2 = eps[..., 0:1], eps[..., 1:2]                       # (..., 1)
    ce, se = _signed_pow(torch.cos(eta), e1), _signed_pow(torch.sin(eta), e1)
    co, so = _signed_pow(torch.cos(omega), e2), _signed_pow(torch.sin(omega), e2)
    x, y, z = scale[..., 0:1] * ce * co, scale[..., 1:2] * ce * so, scale[..., 2:3] * se
    if taper is not None:
        zn = z / scale[..., 2:3]
        x, y = x * (1 + taper[..., 0:1] * zn), y * (1 + taper[..., 1:2] * zn)
    if bend is not None:
        kb = bend / min_radius(scale, taper if taper is not None else torch.zeros_like(scale[..., :2]))
        for ch, axis in [(2, "y"), (1, "x"), (0, "z")]:
            x, y, z = _bend_axis(x, y, z, kb[..., ch:ch + 1], alpha[..., ch:ch + 1], axis)
    local = torch.stack([x, y, z], -1)
    return local @ R.transpose(-1, -2) + t.unsqueeze(-2)


def sq_mesh(scale, eps, R, t, colors=None, n_eta: int = 24, n_omega: int = 48, taper=None, bend=None, alpha=None):
    """trimesh mesh of a set of superquadrics (numpy or torch inputs, leading dim = primitives)."""
    import trimesh

    to_t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=torch.float32)
    pts = sample_surface(to_t(scale), to_t(eps), to_t(R), to_t(t), n_eta, n_omega,
                         to_t(taper), to_t(bend), to_t(alpha)).numpy()  # (M, K, 3)
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


if __name__ == "__main__":
    # Check: our surface of a SuperFlex fit vs SuperFlex's own mesh of it (.ply next to the .npz).
    #   python -m l2sq.sq dataset/chair/gt_sq_deform.npz      -> Chamfer should be ~1e-3 or below
    import sys
    from pathlib import Path

    import trimesh

    from l2sq.data import GTSuperquadrics

    path = Path(sys.argv[1])
    gt = GTSuperquadrics(path, deform=True)
    ours = sq_mesh(gt.scale, gt.eps, gt.R, gt.t, n_eta=48, n_omega=96, taper=gt.taper, bend=gt.bend, alpha=gt.alpha)
    ref = trimesh.load(str(path.with_suffix(".ply")), process=False, force="mesh")
    a = torch.as_tensor(trimesh.sample.sample_surface(ours, 20000, seed=0)[0], dtype=torch.float32)
    b = torch.as_tensor(trimesh.sample.sample_surface(ref, 20000, seed=0)[0], dtype=torch.float32)
    print(f"{len(gt)} superquadrics, max bend {gt.bend.max():.3f}, max |taper| {np.abs(gt.taper).max():.3f}; "
          f"Chamfer(ours, SuperFlex mesh) = {float(chamfer(a, b)):.5f}")
