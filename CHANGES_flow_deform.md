# Change Report: Deformable Superquadric Support for the Flow Decoder (`--flow-deform`)

## 1. What Changed

Added switchable deformable superquadric (`26`-dim) support to `FlowSQDenoiser`, `pack_sq`, `decode_sq`, `flow_loss`, `sample_euler`, `l2sq/train.py`, `l2sq/eval_occlusion.py`, and `l2sq/diagnose_flow.py`, off by default (`--flow-deform`). When off, all 15-dim behavior, checkpoints, saved args, and tests remain identical.

### Teacher File to Pass
- When running with `--decoder flow --flow-deform`, pass the deformable SuperFlex pseudo-GT file **`gt_sq_deform.npz`** to `--gt-sq`:
  ```bash
  python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_deform.npz \
      --decoder flow --flow-deform --flow-loss mix --flow-t-norm --out runs/flow_deform
  ```
- `--deform` continues to apply only to `--decoder parallel`; passing `--decoder flow --deform` exits with instructions to use `--flow-deform`.

### Parameterization & Scaling Choices for the 11 New Deformable Dims
Raw packed 26-dim vector layout (`DEFORM_DIM = 26`):
- Dims `0..14`: unchanged 15 rigid dims (`t` 3, `rot6d` 6, `log_scale` 3, `eps_logit` 2, `exist` 1).
- Dims `15..16` (`taper`, 2): clamped to `[-1.0, 1.0]` directly; default group normalization `(shift=0.0, scale=1.0)` since tapering coefficients lie in `[-1, 1]` centered at `0`. Decoded as `x[..., 15:17].clamp(-1.0, 1.0)`.
- Dims `17..19` (`bend`, 3): dimensionless bending `kb * min_radius` in `[0, 1]` clamped to `[1e-4, 1 - 1e-4]` and packed in logit space `torch.logit(...)`; default normalization `(shift=-4.5, scale=4.5)` maps `[logit(1e-4) ≈ -9.21, 0.0]` roughly onto `[-1.05, +1.0]`, matching `N(0, 1)` scale. Decoded via `torch.sigmoid(x[..., 17:20].clamp(-10.0, 10.0))`.
- Dims `20..25` (`alpha`, 6): teacher bending direction angles `(z, x, y)` packed as 3 raw `(cos(alpha), sin(alpha))` pairs (`shift=0.0, scale=1.0`, matching `rot6d` unit-direction scaling). Decoded by L2-normalizing each 2D `(cos, sin)` pair (`F.normalize(x[..., 20:26].unflatten(-1, (3, 2)), dim=-1)`), computing `alpha = torch.atan2(sin, cos)` and storing `alpha_vec = (sin, cos)` to match `SQDecoder`'s exact representation in `l2sq/model.py` and `l2sq/train.py`.
- `--flow-norm-all` / `--flow-t-norm`: `--flow-t-norm` normalizes `t` (`0..2`) identically for 15 and 26 dims; `--flow-norm-all` now keeps `exist` (`dim 14`) at `(shift=0.5, scale=0.5)` and standardizes all remaining continuous dims (`0..13` and `15..25`) with `std >= 0.05`.

---

## 2. Where Changes Were Made (Old vs New Code)

### 2.1 `l2sq/flow_matching_decoder.py`
- **Lines 14–116**: Added `DEFORM_DIM = 26`, `DEFORM_GROUP_SLICES`, `_DEFORM_SHIFT`, `_DEFORM_SCALE`, `_default_shift_scale()`, and updated `_shift_scale()`, `pack_sq()`, `decode_sq()`, and `flow_group_stats()` to handle both 15 and 26 dims.
- **Lines 166–200 (`FlowSQDenoiser.__init__`)**: Added `deform: bool = False`, set `self.deform = deform`, `self.sq_dim = DEFORM_DIM if deform else SQ_DIM`, and sized `x_proj` (`self.sq_dim * (1 + 2 * fourier_freqs)`), `out_proj`, `x_skip`, `t_scale`, and `flow_shift`/`flow_scale` buffers from `self.sq_dim`.
- **Lines 248–315 (`flow_loss`)**:
  - Velocity loss uses `DEFORM_GROUP_SLICES` when `denoiser.deform` is True (`parts` includes `taper`, `bend`, `alpha`).
  - `mix` loss `x1` part adds `x1_taper` (`MSE` on `taper`), `x1_bend` (`MSE` on `logit(bend)`), and `x1_alpha` (bend-weighted `MSE` between `alpha_vec` and `(sin, cos)` target, matching `mse_loss` in `l2sq/train.py:72-73`), and logs `x1_taper`, `x1_bend`, `x1_alpha` in `parts`.
- **Line 324 (`sample_euler`)**: Changed initial noise shape from hardcoded `SQ_DIM` to `denoiser.sq_dim`.

**Old `FlowSQDenoiser.__init__` / `sample_euler` (`l2sq/flow_matching_decoder.py`):**
```python
        self.deform = False
        self.fourier_freqs = fourier_freqs
        ...
        in_dim = SQ_DIM * (1 + 2 * fourier_freqs)
        self.x_proj = nn.Linear(in_dim, d_model)
        ...
        self.out_proj = nn.Linear(d_model, SQ_DIM)
        self.x_skip = nn.Linear(SQ_DIM, SQ_DIM)
        self.t_scale = nn.Sequential(nn.SiLU(), nn.Linear(d_model, SQ_DIM))
        self.register_buffer("flow_shift", torch.tensor(_SHIFT, dtype=torch.float32))
        self.register_buffer("flow_scale", torch.tensor(_SCALE, dtype=torch.float32))
...
    x = torch.randn((tokens.shape[0], n_queries, SQ_DIM), device=tokens.device, dtype=tokens.dtype, generator=generator)
```

**New `FlowSQDenoiser.__init__` / `sample_euler` (`l2sq/flow_matching_decoder.py:169-200, 324`):**
```python
    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True,
                 slot_init: str = "small", fourier_freqs: int = 0, deform: bool = False):
        ...
        self.deform = deform
        self.sq_dim = DEFORM_DIM if deform else SQ_DIM
        self.fourier_freqs = fourier_freqs
        ...
        in_dim = self.sq_dim * (1 + 2 * fourier_freqs)
        self.x_proj = nn.Linear(in_dim, d_model)
        ...
        self.out_proj = nn.Linear(d_model, self.sq_dim)
        self.x_skip = nn.Linear(self.sq_dim, self.sq_dim)
        self.t_scale = nn.Sequential(nn.SiLU(), nn.Linear(d_model, self.sq_dim))
        ...
        sh, sc = _default_shift_scale(self.sq_dim, torch.device("cpu"), torch.float32)
        self.register_buffer("flow_shift", sh)
        self.register_buffer("flow_scale", sc)
...
    x = torch.randn((tokens.shape[0], n_queries, denoiser.sq_dim), device=tokens.device, dtype=tokens.dtype, generator=generator)
```

### 2.2 `l2sq/train.py`
- **Lines 164–168 (`evaluate`)**: Passes `deform=dec.deform` to `decode_sq` (and `evaluate` already computes `taper_err`, `bend_err`, uniform deformed `even_points`, and deformed `occupancy_iou` whenever `dec.deform` is True).
- **Lines 214–218 (`export_examples`)**: Passes `deform=dec.deform` to `decode_sq`, feeding deformed params into `sq_args` / `sq_mesh` for `.ply` export and setting `"deform": dec.deform` in `sample.json`.
- **Lines 276–277**: Added `--flow-deform` (`action="store_true"`, default `False`, automatically saved in `args.json` and `decoder.pt["args"]`).
- **Lines 358–364**: Loads `GTSuperquadrics(a.gt_sq, deform=use_deform)` with `use_deform = a.flow_deform if a.decoder == "flow" else a.deform`.
- **Lines 378–402**: Instantiates `FlowSQDenoiser(..., deform=a.flow_deform)` and updates `--flow-norm-all` to standardize dims `0..13` and `15..25` while preserving `exist` (`14`) at `(0.5, 0.5)`.

**Old `--flow-norm-all` slice (`l2sq/train.py`):**
```python
            sh = torch.cat([mean_all[:14], torch.tensor([0.5])])
            sc = torch.cat([std_all[:14], torch.tensor([0.5])])
```

**New `--flow-norm-all` slice (`l2sq/train.py:393-402`):**
```python
            sh = torch.cat([mean_all[:14], torch.tensor([0.5]), mean_all[15:]])
            sc = torch.cat([std_all[:14], torch.tensor([0.5]), std_all[15:]])
```

### 2.3 Checkpoint Rebuilding (`l2sq/eval_occlusion.py`, `l2sq/diagnose_flow.py`)
- **`l2sq/eval_occlusion.py` lines 46–51, 199–204, 403**:
  - `load_decoder`: passes `deform=args.get("flow_deform", False)` to `FlowSQDenoiser(...)`. Old checkpoints lacking `"flow_deform"` default to `False` (15 dims).
  - `_eval_batches`: passes `deform=dec_flow.deform` to `decode_sq(...)`.
  - `main`: loads `GTSuperquadrics(gt_sq_path, deform=dec_parallel.deform or dec_flow.deform)`.
- **`l2sq/diagnose_flow.py` lines 97–108, 190–204**:
  - `main`: passes `deform=args.get("flow_deform", False)` to `FlowSQDenoiser(...)` and `deform=dec.deform` to `GTSuperquadrics`.
  - Section 5 (`Raw flow-space ranges`): passes `deform=dec.deform` to `pack_sq` and prints `taper`, `bend`, `alpha` ranges via `DEFORM_GROUP_SLICES` when `dec.deform` is True.

### 2.4 `l2sq/test_flow_matching_decoder.py`
- Added `test_flow_deform()` (lines 479–540) testing 26-dim `pack_sq`/`decode_sq` round trip (`angle -> (cos, sin) -> atan2`), `FlowSQDenoiser(..., deform=True, slot_init="sinus", fourier_freqs=4)` forward/sampling/checkpoint reload, and `flow_loss(..., mode="mix", vel_loss="mse"|"l2")` finite losses/gradients and reported parts.
- Extended `test_train_py_flow_wiring()` (lines 177–194) to run a full CPU train+eval+export step with `--decoder flow --flow-deform --flow-norm-all --flow-loss mix --flow-vel-loss l2 --flow-slot-sinus --flow-fourier 2`, verifying `args.json`, `eval_final.json` (`taper_err`, `bend_err`), and 26-dim buffers in `decoder.pt`.

---

## 3. CPU Test Output

Command:
```bash
cd /Users/syedrayyanadil/Desktop/SuperQuadratics-TUM-DI-LAB/latent2SQ-v2
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder && PYTHONPATH=. python3 -m l2sq.test_eval_occlusion
```

Relevant `test_flow_matching_decoder` output (exit code `0`, all tests passed):
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
...
=== train.py --decoder flow CPU wiring test (1 train step + eval) ===
...
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0), 'taper': (0.25, 0.25), 'bend': (-1.7346010208129883, -1.7346010208129883), 'alpha': (0.4794255495071411, 0.8775825500488281)}
flow norm_all (64 train samples, std>=0.05): {'t': ([0.042, -0.128, 0.462], [0.147, 0.197, 0.245]), 'rot6d': ([0.248, -0.187, 0.198, -0.343, -0.001, 0.37], [0.49, 0.653, 0.444, 0.432, 0.574, 0.479]), 'log_scale': ([-2.868, -2.933, -2.714], [0.834, 0.582, 0.67]), 'eps': ([-1.109, -0.02], [0.459, 0.863]), 'exist': ([0.5], [0.5]), 'taper': ([0.25, 0.25], [0.05, 0.05]), 'bend': ([-1.735, -1.735, -1.735], [0.05, 0.05, 0.05]), 'alpha': ([0.878, 0.479, 0.878, 0.479, 0.878, 0.479], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05])}
decoder parameters: 0.27 M
step     1  loss 18.48  t 2.423  R 3.563  scale 2.383  eps 2.269  taper 1.507  bend 1.887  alpha 2.972  exist 1.091  v_loss 18.09  x1_loss 0.3827  x1_taper 0.0003704  x1_bend 0.0003928  x1_alpha 5.162e-05  lr 0.00e+00  0.11s/step  vggt 0.011s/step  dec 0.037s/step  peak 0.0 GB
[eval train step 1] loss 16.7536  rot 99.6deg  t 0.9264  scale 96.5%  chamfer 0.3344  IoU 0.087  chamfer-gt 0.3619  exist-acc 0.641  IoU-target 0.000  #SQ 8.2/7  empty 0
[eval test  step 1] loss 17.0400  rot 110.1deg  t 0.9650  scale 138.3%  chamfer 0.3088  IoU 0.018  chamfer-gt 0.3350  exist-acc 0.547  IoU-target 0.000  #SQ 10.2/7  empty 0
...
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmprcfo7ufp/run_flow_deform
empty-pred eval check: n_empty=2, chamfer=nan

ALL CPU FLOW TESTS PASSED
...
ALL OCCLUSION EVAL SMOKE TESTS PASSED
```
