# Change Report: Fix CUDA Multinomial Assert in `--flow-geo chamfer`

## Root Cause
When training with `--flow-geo chamfer`, `_geo_surface_loss` in `l2sq/flow_matching_decoder.py` called `l2sq.sq.sample_surface_uniform` on the noisy one-step estimate `x1_hat` across all `P` slots (including inactive and unstabilized slots). Early in training, noisy `x1_hat` values can produce zero, `NaN`, or `Inf` triangle areas, causing `torch.multinomial(area + 1e-12, ...)` in `sample_surface_uniform` to fail on CUDA with:
`binarySearchForMultinomial: Assertion cumdist[size - 1] > 0 failed`.

## What Changed & Where

1. **`l2sq/flow_matching_decoder.py`**
   - **Line 12**: Replaced `sample_surface_uniform` import with `grid_faces`.
   - **Lines 248–263 (`_sanitize_sq_for_sampling`)**: Added helper that replaces non-finite entries in `scale`, `eps`, `t`, `R` (and `taper`, `bend`, `alpha` when `deform=True`) via `torch.where(torch.isfinite(...), ...)` and clamps them to sane finite ranges (`scale` in `[1e-4, 50.0]`, `eps` in `[EPS_MIN, EPS_MAX]`, `t` in `[-50.0, 50.0]`, `taper` in `[-0.99, 0.99]`, `bend` in `[1e-4, 1 - 1e-4]`), preserving gradients on finite parameters.
   - **Lines 266–289 (`_safe_sample_surface_uniform`)**: Added area-uniform differentiable surface sampler for flow chamfer loss that sanitizes face triangle area weights before `torch.multinomial` (`torch.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)`), with a fallback to uniform `1.0` weights if any row still has non-finite entries or `sum <= 0`.
   - **Lines 310–347 (`_geo_surface_loss`, `chamfer` branch)**:
     - Samples only active teacher slots (`active = m > 0.5`).
     - Returns `0.0` immediately when no active slots exist in the batch (`if not active.any()`), and skips individual batch items with `cnt == 0`.
     - Applies `_sanitize_sq_for_sampling` and `_safe_sample_surface_uniform` to predicted SQs (and target SQs when `tgt["surf"]` is absent).
     - Leaves `"points"` mode (lines 298–308) and all default `--flow-geo none` behavior completely untouched.

2. **`l2sq/test_flow_matching_decoder.py`**
   - **Lines 614–640 (inside `test_flow_geo_loss`)**: Added CPU regression checks for `geo_loss="chamfer"`:
     - Tests an `x1_hat` containing a zero-scale SQ (slot 0) and a NaN/Inf/zero-rot6d SQ (slot 1) alongside a valid active slot (slot 2), asserting finite loss, finite gradients, and non-zero gradient on the valid slot.
     - Tests partial active masks and an all-zero active mask (`m_none`), asserting finite / `0.0` loss.

## Old vs. New Code

### `l2sq/flow_matching_decoder.py`

Old (import & `_geo_surface_loss` `chamfer` branch):
```python
from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, chamfer, rot6d_to_matrix, sample_surface, sample_surface_uniform
```
```python
    if geo_loss == "chamfer":
        deform_p = (pred_sq["taper"], pred_sq["bend"], pred_sq["alpha"]) if deform else (None, None, None)
        p_all = sample_surface_uniform(pred_sq["scale"], pred_sq["eps"], pred_sq["R"], pred_sq["t"],
                                       n_geo_points, *deform_p, generator=generator)        # (B,P,S,3)
        has_surf = "surf" in tgt
        if has_surf:
            g_surf = rep(tgt["surf"])
        else:
            with torch.no_grad():
                deform_g = (rep(tgt["taper"]), rep(tgt["bend"]), rep(tgt["alpha"])) if deform else (None, None, None)
                g_all = sample_surface_uniform(rep(tgt["scale"]), rep(tgt["eps"]), rep(tgt["R"]), rep(tgt["t"]),
                                               n_geo_points, *deform_g, generator=generator)
        s_vec = rep(tgt["s"]) if "s" in tgt else None
        losses = []
        for b in range(p_all.shape[0]):
            mb = m[b] > 0.5
            if not mb.any():
                continue
            p_b = p_all[b, mb].reshape(-1, 3)
            g_b = g_surf[b] if has_surf else g_all[b, mb].reshape(-1, 3)
            sb = s_vec[b] if s_vec is not None else 1.0
            losses.append(wt[b, 0] * chamfer(p_b, g_b) * sb)
        return torch.stack(losses).mean() if losses else p_all.sum() * 0.0
```

New (lines 12, 248–289, 310–347):
```python
from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, chamfer, grid_faces, rot6d_to_matrix, sample_surface
```
```python
def _sanitize_sq_for_sampling(sq: dict, deform: bool) -> dict:
    """Clamp/clean SQ tensors to finite sane ranges for surface sampling while preserving finite gradients."""
    scale = torch.where(torch.isfinite(sq["scale"]), sq["scale"], torch.full_like(sq["scale"], SCALE_MIN))
    scale = scale.clamp(min=1e-4, max=50.0)
    eps = torch.where(torch.isfinite(sq["eps"]), sq["eps"], torch.ones_like(sq["eps"])).clamp(min=EPS_MIN, max=EPS_MAX)
    t = torch.where(torch.isfinite(sq["t"]), sq["t"], torch.zeros_like(sq["t"])).clamp(min=-50.0, max=50.0)
    eye = torch.eye(3, device=sq["R"].device, dtype=sq["R"].dtype).expand_as(sq["R"])
    r_ok = torch.isfinite(sq["R"]).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    R = torch.where(r_ok, sq["R"], eye)
    out = {"scale": scale, "eps": eps, "R": R, "t": t}
    if deform:
        taper = torch.where(torch.isfinite(sq["taper"]), sq["taper"], torch.zeros_like(sq["taper"])).clamp(-0.99, 0.99)
        bend = torch.where(torch.isfinite(sq["bend"]), sq["bend"], torch.full_like(sq["bend"], 1e-4)).clamp(1e-4, 1.0 - 1e-4)
        alpha = torch.where(torch.isfinite(sq["alpha"]), sq["alpha"], torch.zeros_like(sq["alpha"]))
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
```
```python
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
```

## Test Output
Ran:
```bash
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder
```

Result (`Command exited with code 0`):
```text
=== Overfit 1 fixed batch (500 steps) ===
  step   1: loss=9.1109  exist=2.2268  t=2.1510  R=1.5065  scale=1.5302  eps=1.6965
  ...
  step 500: loss=0.0539  exist=0.0084  t=0.0107  R=0.0206  scale=0.0047  eps=0.0094
  eval@t=0.5: loss=0.0576  exist=0.0149
...
=== train.py --decoder flow CPU wiring test (1 train step + eval) ===
...
empty-pred eval check: n_empty=2, chamfer=nan

ALL CPU FLOW TESTS PASSED
```
