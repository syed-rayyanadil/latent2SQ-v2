"""SuperFlex's geometric loss (Tavernini et al. 2026, Sec. 4.1 + App. B.2), for our decoder.

    L = w_iou L_IoU + w_sdf L_SDF + w_overlap L_overlap + w_sparsity L_sparsity + w_exist L_exist

No pseudo-GT parameters are used: only GT surface points (for L_SDF) and volume points with
inside/outside labels (for L_IoU), both in the sample frame. So the slot order of the queries
does not matter and the decoder may pick its own decomposition.

Ported from superflex/loss/loss.py (IoULoss) and superflex/optimization.py, with one change:
SuperFlex weights L_SDF with a learned point-to-primitive assignment from its point encoder;
our decoder sees images, not points, so we use the union's distance (soft minimum over the
primitives, weighted by existence) as SuperFlex's test-time optimization does, and derive the
usage for L_sparsity / L_exist from the same soft assignment.

Everything is computed after normalizing each sample like SuperFlex's input (centered on the
GT surface points, divided by c = 2 max |coordinate|, i.e. the object fits a unit cube), so
its hyperparameters (tau_IoU = 1e-3, truncation 0.05, ...) keep their meaning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from l2sq.sq import min_radius

TAU_IOU = 1e-3        # soft occupancy temperature (Eq. 6)
TAU_UNION = 1e-2      # soft-min temperature over primitives (optimization.py)
TAU_OVERLAP = 1e-3
EXIST_POINTS = 24     # a primitive "exists" if it is the closest one for > 24 of 4096 surface points
SCALE_FLOOR = 0.01    # IoULoss clamps scales to >= 0.01 (normalized units) for stable distances


def _safe_pow(x, y, eps=1e-3):
    return x.abs().clamp(min=eps, max=5e2).pow(y).clamp(min=eps, max=5e2)


def _safe_mul(x, y, eps=1e-4):
    x = torch.sign(x) * x.abs().clamp(eps, 1e6)
    y = torch.sign(y) * y.abs().clamp(eps, 1e6)
    return x * y + eps


def _inverse_bend(x, y, z, kb, alpha, axis: str):
    """Inverse of l2sq.sq._bend_axis (superflex/loss/utils.py inverse_bending_axis); identity for kb < 1e-3."""
    u, v, w = {"z": (x, y, z), "x": (y, z, x), "y": (z, x, y)}[axis]
    on = kb >= 1e-3
    inv = 1.0 / kb.clamp_min(1e-3)
    beta = torch.atan2(v, u)
    R = torch.sqrt(u ** 2 + v ** 2 + 1e-12) * torch.cos(alpha - beta)
    gamma = torch.atan2(w, inv - R)
    r = inv - torch.sqrt(w ** 2 + (inv - R) ** 2)
    u2, v2, w2 = u - (R - r) * torch.cos(alpha), v - (R - r) * torch.sin(alpha), inv * gamma
    u, v, w = torch.where(on, u2, u), torch.where(on, v2, v), torch.where(on, w2, w)
    return {"z": (u, v, w), "x": (w, u, v), "y": (v, w, u)}[axis]


def radial_distance(points: torch.Tensor, sq: dict) -> torch.Tensor:
    """Signed radial distance (paper Eq. 3, superflex sdf_batch with use_floor) of points (B,M,3)
    to every superquadric in sq (scale (B,P,3), eps, R, t, optional taper / bend / alpha) -> (B,P,M)."""
    scale = sq["scale"].clamp_min(SCALE_FLOOR)
    X = torch.einsum("bpji,bpmj->bpmi", sq["R"], points[:, None] - sq["t"][:, :, None])   # R^T (x - t)
    eps = 1e-6
    X = torch.where(X > 0, 1.0, -1.0) * X.abs().clamp_min(eps)
    x, y, z = X.unbind(-1)                                                                  # (B,P,M)
    r0 = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    sx, sy, sz = (scale[..., i:i + 1] for i in range(3))
    if "taper" in sq:
        taper = sq["taper"]
        kb = sq["bend"] / min_radius(scale, taper)                                          # (B,P,3) z,x,y
        for ch, axis in [(0, "z"), (1, "x"), (2, "y")]:                                     # reverse of forward
            x, y, z = _inverse_bend(x, y, z, kb[..., ch:ch + 1], sq["alpha"][..., ch:ch + 1], axis)
        fx = _safe_mul(taper[..., 0:1] / sz, z) + 1
        fy = _safe_mul(taper[..., 1:2] / sz, z) + 1
        fx = torch.where(fx > 0, 1.0, -1.0) * fx.abs().clamp_min(eps)
        fy = torch.where(fy > 0, 1.0, -1.0) * fy.abs().clamp_min(eps)
        x, y = x / fx, y / fy
    e1, e2 = sq["eps"][..., 0:1], sq["eps"][..., 1:2]
    t1 = _safe_pow(_safe_pow(x / sx, 2), 1 / e2)
    t2 = _safe_pow(_safe_pow(y / sy, 2), 1 / e2)
    t3 = _safe_pow(_safe_pow(z / sz, 2), 1 / e1)
    f = _safe_pow(_safe_pow(t1 + t2, e2 / e1) + t3, -e1 / 2)
    d = _safe_mul(r0, 1 - f)
    floor = r0 - scale.norm(dim=-1, keepdim=True)          # grows linearly outside the bounding sphere
    return torch.maximum(d, floor)


def normalize(pred: dict, tgt: dict) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center on the GT surface points and divide by c = 2 max|coord| (SuperFlex normalize_points).
    Returns normalized superquadrics, surface points, volume points, and c (B,)."""
    surf = tgt["surf"]
    mu = surf.mean(1, keepdim=True)                                     # (B,1,3)
    c = 2 * (surf - mu).abs().amax(dim=(1, 2))                          # (B,)
    cc = c[:, None, None]
    sq = {"scale": pred["scale"] / cc, "t": (pred["t"] - mu) / cc, "R": pred["R"], "eps": pred["eps"]}
    if "taper" in pred:
        sq.update(taper=pred["taper"], bend=pred["bend"], alpha=pred["alpha"])   # scale-free
    return sq, (surf - mu) / cc, (tgt["vol"] - mu) / cc, c


def superflex_loss(pred: dict, tgt: dict, w: dict, truncation: float = 0.05,
                   neg_slope: float = 0.1) -> tuple[torch.Tensor, dict]:
    sq, surf, vol, _ = normalize(pred, tgt)
    exist = torch.sigmoid(pred["exist_logit"])                                     # (B,P)
    log_e = torch.log(exist.clamp_min(1e-6))[..., None]                            # (B,P,1)

    # surface: soft minimum of the existence-weighted distances = distance to the union
    d_s = radial_distance(surf, sq)                                                # (B,P,Ms)
    logits = -d_s / TAU_UNION + log_e
    d_union = -TAU_UNION * torch.logsumexp(logits, dim=1)                          # (B,Ms)
    psi = truncation * torch.tanh(d_union / truncation)
    l_sdf = 16 * F.leaky_relu(psi, neg_slope).abs().mean()                         # App. B.2, omega = neg_slope

    # volume: probabilistic union of soft occupancies (Eqs. 6-8)
    d_v = radial_distance(vol, sq)                                                 # (B,P,Mv)
    occ = tgt["occ"]
    p = (exist[..., None] * torch.sigmoid(-d_v / TAU_IOU)).clamp(0.0, 0.9999)
    pred_occ = 1.0 - torch.exp(torch.log(1.0 - p).sum(1))                          # (B,Mv)
    inter = (pred_occ * occ).sum(1)
    union = (pred_occ + occ - pred_occ * occ).sum(1)
    iou = (inter + 1e-6) / (union.clamp_min(1.0) + 1e-6)
    l_iou = -torch.log(iou).mean()

    # overlap (Eq. 11): volume claimed by more than one existing primitive
    ind = torch.sigmoid(-(d_v + truncation) / TAU_OVERLAP) * (exist[..., None] > 0.5)
    l_overlap = F.relu(ind.sum(1) - 1).mean()

    # usage from the soft assignment of the surface points -> sparsity (0.5-norm) and existence targets
    assign = torch.softmax(logits, dim=1)                                          # (B,P,Ms)
    usage = assign.mean(-1)                                                        # (B,P)
    l_sparsity = (usage + 0.01).sqrt().mean(1).pow(2).mean()
    counts = F.one_hot(d_s.abs().argmin(1), d_s.shape[1]).sum(1).float()           # (B,P) closest-primitive counts
    exist_tgt = (counts > EXIST_POINTS * d_s.shape[-1] / 4096).float()
    l_exist = F.binary_cross_entropy_with_logits(pred["exist_logit"], exist_tgt)

    parts = {"iou": l_iou, "sdf": l_sdf, "overlap": l_overlap, "sparsity": l_sparsity, "exist_sf": l_exist}
    total = sum(w[k] * v for k, v in parts.items())
    logs = {k: v.item() for k, v in parts.items()}
    logs["iou_value"] = iou.mean().item()
    return total, logs


@torch.no_grad()
def occupancy_iou(pred: dict, tgt: dict, keep: torch.Tensor) -> torch.Tensor:
    """Hard IoU of the union of the kept superquadrics with the GT occupancy, per sample (B,)."""
    sq, _, vol, _ = normalize(pred, tgt)
    inside = (radial_distance(vol, sq) < 0) & keep[..., None]
    pred_occ = inside.any(1).float()
    occ = tgt["occ"]
    return (pred_occ * occ).sum(1) / ((pred_occ + occ - pred_occ * occ).sum(1).clamp_min(1.0))
