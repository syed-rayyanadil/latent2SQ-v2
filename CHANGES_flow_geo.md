# Change Report: Switchable Shape-Based Loss (`--flow-geo`) for the Flow Decoder

## 1. What Changed

Added a switchable, default-off surface geometry loss (`--flow-geo {none,points,chamfer}`, `--w-geo 1.0`) to `flow_loss` in `mode="mix"` so pose (translation, rotation), scale, shape, and deformation errors on thin/elongated superquadrics are penalized directly on the one-step clean prediction `x1_hat = x_t + (1 - t) * v_pred`. When `--flow-geo none` (default), all behavior, `parts` keys, saved defaults, and tests stay unchanged.

### Reused Existing Geometry Functions
All geometry computation reuses existing functions from `l2sq/sq.py` and `l2sq/data.py` (`GeomTargets` / `tgt["surf"]`):
- `rot6d_to_matrix` (`l2sq/sq.py:38-47`): converts predicted 6D rotation inside `decode_sq` into differentiable `3 x 3` rotation matrices.
- `sample_surface` (`l2sq/sq.py:112-129`): used by `geo_loss == "points"` on the deterministic `(eta, omega)` grid (`sq_grid`), supporting both rigid and deformable (`taper`, `bend`, `alpha`) superquadrics.
- `sample_surface_uniform` (`l2sq/sq.py:140-161`): used by `geo_loss == "chamfer"` to uniformly sample surfaces of active predicted SQs (and active teacher SQs if `tgt["surf"]` is absent in synthetic batches).
- `chamfer` (`l2sq/sq.py:193-196`): symmetric L2 Chamfer distance between active predicted SQ surface points and `tgt["surf"]` (from `GeomTargets` via `collate`), scaled to world units via `tgt["s"]`.

### Point Counts & `t`-Weighting
- **`points` mode**: uses a deterministic **`8 x 16 = 128` points per SQ** `(eta, omega)` grid (`sample_surface(..., 8, 16)`). Computes per-point squared L2 distance `((p_pts - g_pts) ** 2).sum(dim=-1).mean(dim=-1)` between corresponding `(eta, omega)` points on the predicted SQ and same-slot teacher SQ (teacher points computed under `torch.no_grad()`), masked by active teacher slots `m = tgt["exist"]`, weighted by `wt = t[:, None]` (exact same formula as `x1_masked`), and normalized by `n = m.sum().clamp_min(1.0)`.
- **`chamfer` mode**: samples **`n_geo_points = 128` points per active SQ** via `sample_surface_uniform`, flattens each sample's active slots `m[b] > 0.5`, computes `chamfer(p_b, g_b) * sb` (world units via `tgt["s"][b]`, against `tgt["surf"][b]` when present or sampled teacher SQs otherwise), weights each sample `b` by `wt[b, 0] = t[b]`, and averages across the batch.
- **Total loss**: `total = v_loss + w_x1 * x1_loss + w_geo * geo`, with `"geo"` added to the returned `parts` dict (and printed in `train.py` stderr log lines and TensorBoard `train_loss/geo`).

### Checkpoint Loading Compatibility
- `l2sq/train.py` saves `flow_geo` and `w_geo` into `args.json` and `decoder.pt["args"]` via `vars(a)`.
- `evaluate()` in `l2sq/train.py` uses `cfg.get("flow_geo", "none")` and `cfg.get("w_geo", 1.0)`, so legacy `cfg`/`args.json` dicts without these keys default safely to `"none"` / `1.0`.
- `l2sq/eval_occlusion.py` and `l2sq/diagnose_flow.py` only call `sample_euler` and `decode_sq` (not `flow_loss`), and `FlowSQDenoiser` architecture is unchanged by `--flow-geo`, so both old and new checkpoints load without modification.

---

## 2. Where Changes Were Made (Old vs New Code)

### 2.1 `l2sq/flow_matching_decoder.py`
- **Line 12**: imported `chamfer`, `sample_surface`, `sample_surface_uniform` alongside `rot6d_to_matrix` from `l2sq.sq`.
- **Lines 248–290**: added `_geo_surface_loss(...)`.
- **Lines 293–368**: added `geo_loss: str = "none"`, `w_geo: float = 1.0`, `n_geo_points: int = 128` to `flow_loss(...)`, validating `geo_loss` and adding `w_geo * geo` + `"geo"` in `out` only when `geo_loss != "none"` in `mode == "mix"`.

**Old (`l2sq/flow_matching_decoder.py`):**
```python
def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None, repeat: int = 1,
              mode: str = "velocity", w_x1: float = 1.0,
              vel_loss: str = "mse") -> tuple[torch.Tensor, dict]:
...
    total = v_loss + w_x1 * x1_loss
    out = {k: v.item() for k, v in parts.items()}
    out.update(v_loss=v_loss.item(), x1_loss=x1_loss.item(), **extra_x1)
    return total, out
```

**New (`l2sq/flow_matching_decoder.py:293-368`):**
```python
def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None, repeat: int = 1,
              mode: str = "velocity", w_x1: float = 1.0,
              vel_loss: str = "mse", geo_loss: str = "none",
              w_geo: float = 1.0, n_geo_points: int = 128) -> tuple[torch.Tensor, dict]:
...
    total = v_loss + w_x1 * x1_loss
    if geo_loss != "none":
        geo = _geo_surface_loss(pred_sq, tgt, wt, m, n, rep, denoiser.deform, geo_loss, repeat,
                                n_geo_points=n_geo_points, generator=generator)
        total = total + w_geo * geo
        extra_x1["geo"] = geo.item()
    out = {k: v.item() for k, v in parts.items()}
    out.update(v_loss=v_loss.item(), x1_loss=x1_loss.item(), **extra_x1)
    return total, out
```

### 2.2 `l2sq/train.py`
- **Lines 164–172 (`evaluate`)**: passes `flow_geo`/`w_geo` when set (with `.get(...)` defaults so `"none"` keeps `mode="velocity"` unchanged).
- **Lines 284–286**: added `--flow-geo` (`choices=["none", "points", "chamfer"], default="none"`) and `--w-geo` (`type=float, default=1.0`).
- **Lines 363–370**: added `flow_loss`, `w_x1`, `flow_geo`, `w_geo` to `cfg` and checked `--flow-geo` requires `--flow-loss mix`.
- **Lines 494–496**: passed `geo_loss=a.flow_geo, w_geo=a.w_geo` to `flow_loss(...)`.

**Old (`l2sq/train.py`):**
```python
            if cfg.get("decoder", "parallel") == "flow":
                pred = decode_sq(sample_euler(dec, tokens, n_queries=P, n_steps=cfg.get("flow_steps", 20), generator=gen),
                                 t_mean=dec.t_mean, t_std=dec.t_std, shift=dec.flow_shift, scale=dec.flow_scale,
                                 deform=dec.deform)
                loss, _ = flow_loss(dec, tokens, tgt, generator=gen, vel_loss=cfg.get("flow_vel_loss", "mse"))
...
        if a.decoder == "flow":
            loss, parts = flow_loss(dec, tokens, tgt, repeat=a.flow_repeat, mode=a.flow_loss,
                                    w_x1=a.w_x1, vel_loss=a.flow_vel_loss)
```

**New (`l2sq/train.py:164-172, 284-286, 494-496`):**
```python
            if cfg.get("decoder", "parallel") == "flow":
                pred = decode_sq(sample_euler(dec, tokens, n_queries=P, n_steps=cfg.get("flow_steps", 20), generator=gen),
                                 t_mean=dec.t_mean, t_std=dec.t_std, shift=dec.flow_shift, scale=dec.flow_scale,
                                 deform=dec.deform)
                geo_mode = cfg.get("flow_geo", "none")
                eval_mode = cfg.get("flow_loss", "velocity") if geo_mode != "none" else "velocity"
                loss, _ = flow_loss(dec, tokens, tgt, generator=gen, mode=eval_mode,
                                    w_x1=cfg.get("w_x1", 1.0), vel_loss=cfg.get("flow_vel_loss", "mse"),
                                    geo_loss=geo_mode, w_geo=cfg.get("w_geo", 1.0))
...
    p.add_argument("--flow-geo", choices=["none", "points", "chamfer"], default="none",
                   help="t-weighted shape loss on x1_hat in --flow-loss mix (default none)")
    p.add_argument("--w-geo", type=float, default=1.0, help="weight of --flow-geo loss")
...
        if a.decoder == "flow":
            loss, parts = flow_loss(dec, tokens, tgt, repeat=a.flow_repeat, mode=a.flow_loss,
                                    w_x1=a.w_x1, vel_loss=a.flow_vel_loss,
                                    geo_loss=a.flow_geo, w_geo=a.w_geo)
```

### 2.3 `l2sq/test_flow_matching_decoder.py`
- Added `test_flow_geo_loss()` verifying:
  - `_geo_surface_loss(..., geo_loss="points")` equals `0.0` when `pred == tgt`, and strictly increases (`> 1e-4`) when one thin rod is rotated 20 degrees or shifted by `0.1`.
  - Gradients flow cleanly through `decode_sq` (`rot6d`, `t`) and `sample_surface` / `sample_surface_uniform`.
  - Both `"points"` and `"chamfer"` produce finite losses, finite gradients, and `"geo"` in `parts` for both rigid (`15`-dim) and deformable (`26`-dim) decoders combined with `--flow-slot-sinus`, `--flow-fourier`, `--flow-vel-loss l2`, and `--flow-deform`.
  - Default `geo_loss="none"` omits `"geo"` from `parts`.
- Extended `test_train_py_flow_wiring()` with `--flow-geo points --w-geo 0.5`, verifying `args.json` and log output (`geo ...`).

---

## 3. CPU Test Output

Command:
```bash
cd /Users/syedrayyanadil/Desktop/SuperQuadratics-TUM-DI-LAB/latent2SQ-v2
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder && PYTHONPATH=. python3 -m l2sq.test_eval_occlusion
```

Output summary (exit code `0`):
```text
=== Overfit 1 fixed batch (500 steps) ===
  step 500: loss=0.0539  exist=0.0084  t=0.0107  R=0.0206  scale=0.0047  eps=0.0094
  eval@t=0.5: loss=0.0576  exist=0.0149
=== Single-batch overfit with K=4 (200 steps) ===
  step 200: loss=0.3511  exist=0.0439
=== Single-batch overfit with --flow-loss mix (200 steps) ===
  step 200: total=0.9394  v_loss=0.8955  x1_loss=0.0439
=== Single-batch overfit with --flow-t-norm on (200 steps) ===
  step 200: total=0.9520  v_loss=0.9153  x1_loss=0.0367
=== Single-batch overfit with --flow-norm-all (200 steps) ===
  step 200: total=1.0335  v_loss=1.0175  x1_loss=0.0160

=== train.py --decoder flow CPU wiring test (1 train step + eval) ===
...
step     1  loss 18.49  t 2.423  R 3.563  scale 2.383  eps 2.269  taper 1.507  bend 1.887  alpha 2.972  exist 1.091  v_loss 18.09  x1_loss 0.3827  x1_taper 0.0003704  x1_bend 0.0003928  x1_alpha 5.162e-05  geo 0.03312  lr 0.00e+00  0.02s/step  vggt 0.003s/step  dec 0.011s/step  peak 0.0 GB
[eval train step 1] loss 17.1264  rot 99.6deg  t 0.9264  scale 96.5%  chamfer 0.3344  IoU 0.051  chamfer-gt 0.3674  exist-acc 0.641  IoU-target 0.154  #SQ 8.2/7  empty 0
[eval test  step 1] loss 17.4489  rot 110.1deg  t 0.9650  scale 138.3%  chamfer 0.3088  IoU 0.016  chamfer-gt 0.3316  exist-acc 0.547  IoU-target 0.154  #SQ 10.2/7  empty 0
...
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmphp4ivzz1/run_flow_deform
empty-pred eval check: n_empty=2, chamfer=nan

ALL CPU FLOW TESTS PASSED
...
ALL OCCLUSION EVAL SMOKE TESTS PASSED
```
