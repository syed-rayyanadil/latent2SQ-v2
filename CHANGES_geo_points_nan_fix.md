# Change Report: Fix NaN Gradients in `--flow-geo points` & Add `--grad-clip` Guard

## Root Cause
1. **Subnormal `atan2` singularity in `_bend_axis` (`l2sq/sq.py`)**: On the polar rings of the `8 x 16` parameter grid (`eta = +/- pi/2`), `cos(eta) ~ 0` drives `(x, y)` to subnormal float32 scale after `(y, x)` bending (`u^2 + v^2 ~ 2.8e-45`). Computing `beta = torch.atan2(v, u)` and `r = torch.sqrt(u**2 + v**2 + 1e-12) * torch.cos(alpha - beta)` caused `Atan2Backward0` (`u / (u^2 + v^2)`) to overflow and produce `NaN` gradients on `scale`, `eps`, `taper`, `bend`, and `alpha` even when the forward loss `geo` and all parameters were finite. Then `clip_grad_norm_` scaled every parameter gradient by `NaN`, poisoning model weights on `opt.step()` (`loss = nan` on all subsequent steps).
2. **Unsanitized `points` path & zero-vector normalization / `torch.where` NaN leaks**:
   - `_geo_surface_loss`'s `"points"` branch did not sanitize `pred_sq` or teacher `tgt` (`no_grad`) before `sample_surface`, so extreme/non-finite predicted or padded teacher SQs could yield `NaN`/`Inf` points (`NaN * 0.0 = NaN` when masked by `m`).
   - `_sanitize_sq_for_sampling` used `torch.where(torch.isfinite(x), x, fallback)` directly around expressions that may hold `NaN`, and `decode_sq` / `rot6d_to_matrix` did not sanitize non-finite/extreme inputs or guard small/zero vectors before `F.normalize` (`eps=1e-12`) and `atan2`, allowing `NaN` or `1e12`-scale gradients to leak through backward.
   - `sample_surface` divided by raw `scale[..., 2:3]` (taper) and raw `min_radius(...)` (bend) without a minimum epsilon clamp.

## What Changed & Where

1. **`l2sq/sq.py`**
   - **Lines 44–45 (`rot6d_to_matrix`)**: Raised `F.normalize(..., eps=1e-6)` (bitwise identical for vectors with norm `>= 1e-6`, prevents `1e12` derivative spikes on zero/collinear vectors).
   - **Lines 99–100 (`_bend_axis`)**: Replaced `beta = torch.atan2(v, u); r = torch.sqrt(u ** 2 + v ** 2 + 1e-12) * torch.cos(alpha - beta)` with the exact Cartesian dot-product identity `r = u * torch.cos(alpha) + v * torch.sin(alpha)`, eliminating both the `atan2(0, 0)`/subnormal pole singularity and `sqrt` at `0`.
   - **Lines 122–126 (`sample_surface`)**: Clamped taper z-divisor `scale[..., 2:3].clamp_min(1e-6)` and bend `min_radius(...).clamp_min(1e-6)` (bitwise identical for normal scales `>= 1e-4`, avoids `0 / 0 = NaN` if `scale == 0`).

2. **`l2sq/flow_matching_decoder.py`**
   - **Lines 79–102 (`decode_sq`)**: Applied `torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)` at entry (bitwise identical for finite values in `[-1e4, 1e4]`), clamped `rot6d` / raw bend direction vectors to `[-1e4, 1e4]`, set `F.normalize(..., eps=1e-6)` for bend directions, and substituted safe `[1.0, 0.0]` before `atan2` when the raw 2D vector norm `< 1e-6`.
   - **Lines 252–267 (`_sanitize_sq_for_sampling`)**: Switched to `torch.nan_to_num(...).clamp(...)` across all fields (`scale`, `eps`, `t`, `R`, and `taper`/`bend`/`alpha`) so `NaN` backward gradients cannot leak through `torch.where` around `NaN` tensors; clamped `bend` to `[0.0, 1.0 - 1e-4]` so zero-bend teacher SQs remain exact.
   - **Lines 302–320 (`_geo_surface_loss`, `"points"` branch)**: Skips cleanly (`0.0`) when no active teacher slots exist, and sanitizes both predicted SQs (`p_clean = _sanitize_sq_for_sampling(pred_sq, deform)`) and teacher targets inside `torch.no_grad()` (`t_clean = _sanitize_sq_for_sampling(..., deform)`).

3. **`l2sq/train.py`**
   - **Lines 289–290**: Added `--grad-clip` (`float`, default `0.0` = off), automatically recorded in `args.json`.
   - **Lines 509–518**: When `--grad-clip > 0`, clips gradients with `torch.nn.utils.clip_grad_norm_(dec.parameters(), a.grad_clip)` and skips `opt.step()` with a warning to `stderr` if `loss` or `grad_norm` is non-finite. When `0.0` (default), behavior is identical to before (`clip_grad_norm_(dec.parameters(), 1.0)` + `opt.step()`).

4. **`l2sq/test_flow_matching_decoder.py`**
   - **Lines 213–246 (`test_train_py_flow_wiring`)**: Added CPU checks that `--grad-clip` defaults to `0.0` in `args.json`, produces identical weights whether omitted or `0`, records `--grad-clip 1.0`, and skips `opt.step()` with a warning while keeping weights finite when loss is non-finite.
   - **Lines 678–763 (`test_flow_geo_loss`, section 5)**: Added CPU regression tests in `deform=True` mode covering zero-scale SQs, `eps` at `EPS_MIN`/`EPS_MAX` clamp limits, non-finite SQs (in both `_geo_surface_loss` and `flow_loss` via extreme/NaN-injected `x1_hat`), non-finite padded teacher slots, and 10-step `AdamW` training stability at `w_geo=20.0`.

## Old vs. New Code

### `l2sq/sq.py`
Old:
```python
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
...
    beta = torch.atan2(v, u)
    r = torch.sqrt(u ** 2 + v ** 2 + 1e-12) * torch.cos(alpha - beta)
...
    if taper is not None:
        zn = z / scale[..., 2:3]
        x, y = x * (1 + taper[..., 0:1] * zn), y * (1 + taper[..., 1:2] * zn)
    if bend is not None:
        kb = bend / min_radius(scale, taper if taper is not None else torch.zeros_like(scale[..., :2]))
```
New:
```python
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
...
    # Equivalent to sqrt(u^2 + v^2) * cos(alpha - atan2(v, u)) without atan2/sqrt singularity at u=v=0.
    r = u * torch.cos(alpha) + v * torch.sin(alpha)
...
    if taper is not None:
        zn = z / scale[..., 2:3].clamp_min(1e-6)
        x, y = x * (1 + taper[..., 0:1] * zn), y * (1 + taper[..., 1:2] * zn)
    if bend is not None:
        mr = min_radius(scale, taper if taper is not None else torch.zeros_like(scale[..., :2])).clamp_min(1e-6)
        kb = bend / mr
```

### `l2sq/flow_matching_decoder.py`
Old (`points` branch & `_sanitize_sq_for_sampling`):
```python
def _sanitize_sq_for_sampling(sq: dict, deform: bool) -> dict:
    scale = torch.where(torch.isfinite(sq["scale"]), sq["scale"], torch.full_like(sq["scale"], SCALE_MIN))
    scale = scale.clamp(min=1e-4, max=50.0)
    ...
    if geo_loss == "points":
        deform_p = (pred_sq["taper"], pred_sq["bend"], pred_sq["alpha"]) if deform else (None, None, None)
        p_pts = sample_surface(pred_sq["scale"], pred_sq["eps"], pred_sq["R"], pred_sq["t"],
                               8, 16, *deform_p)
        with torch.no_grad():
            deform_g = (rep(tgt["taper"]), rep(tgt["bend"]), rep(tgt["alpha"])) if deform else (None, None, None)
            g_pts = sample_surface(rep(tgt["scale"]), rep(tgt["eps"]), rep(tgt["R"]), rep(tgt["t"]),
                                   8, 16, *deform_g)
        d_slot = ((p_pts - g_pts) ** 2).sum(dim=-1).mean(dim=-1)
        return ((d_slot * wt) * m).sum() / n
```
New (also pre-cleaning `x` and small bend vectors in `decode_sq`):
```python
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
```
```python
    if geo_loss == "points":
        if not (m > 0.5).any():
            return pred_sq["t"].sum() * 0.0
        p_clean = _sanitize_sq_for_sampling(pred_sq, deform)
        deform_p = (p_clean["taper"], p_clean["bend"], p_clean["alpha"]) if deform else (None, None, None)
        p_pts = sample_surface(p_clean["scale"], p_clean["eps"], p_clean["R"], p_clean["t"],
                               8, 16, *deform_p)
        with torch.no_grad():
            t_clean = _sanitize_sq_for_sampling({
                "scale": rep(tgt["scale"]), "eps": rep(tgt["eps"]),
                "R": rep(tgt["R"]), "t": rep(tgt["t"]),
                **({k: rep(tgt[k]) for k in ("taper", "bend", "alpha")} if deform else {}),
            }, deform)
            deform_g = (t_clean["taper"], t_clean["bend"], t_clean["alpha"]) if deform else (None, None, None)
            g_pts = sample_surface(t_clean["scale"], t_clean["eps"], t_clean["R"], t_clean["t"],
                                   8, 16, *deform_g)
        d_slot = ((p_pts - g_pts) ** 2).sum(dim=-1).mean(dim=-1)
        return ((d_slot * wt) * m).sum() / n
```

### `l2sq/train.py`
Old:
```python
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()
```
New:
```python
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), a.grad_clip)
            if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                print(f"[warn] step {step}: non-finite loss ({loss.item()}) or grad norm ({float(grad_norm)}), "
                      f"skipping optimizer step", file=sys.stderr)
            else:
                opt.step()
        else:
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
        sched.step()
```

## Test Output
Ran:
```bash
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder
PYTHONPATH=. python3 -m l2sq.test_eval_occlusion
```

Results (`Command exited with code 0` for both):
```text
=== Overfit 1 fixed batch (500 steps) ===
...
empty-pred eval check: n_empty=2, chamfer=nan
...
saved .../run_gc_default
...
saved .../run_gc_zero

ALL CPU FLOW TESTS PASSED
```
```text
ALL OCCLUSION EVAL SMOKE TESTS PASSED
```
