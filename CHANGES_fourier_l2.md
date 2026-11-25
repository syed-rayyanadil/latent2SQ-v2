# Change Report: Fourier Input Embedding & Euclidean Velocity Loss

## 1. What Changed

1. **Fourier input embedding (`fourier_freqs` / `--flow-fourier`, default `0`)**: Added `fourier_freqs: int = 0` to `FlowSQDenoiser.__init__`. When `fourier_freqs > 0`, `FlowSQDenoiser.forward` embeds `x_t` before `self.x_proj` as `[x, sin(x * pi * 2^k), cos(x * pi * 2^k) for k in 0..fourier_freqs-1]` along the last dimension (`15 * (1 + 2 * fourier_freqs)` inputs to `self.x_proj`), leaving everything else in `forward` (`x_skip(x_t)`, attention blocks, output projections) unchanged.
2. **Euclidean velocity loss (`vel_loss` / `--flow-vel-loss`, default `"mse"`)**: Added `vel_loss: str = "mse"` to `flow_loss`. With `vel_loss == "l2"`, each group's velocity error is computed per SQ as `sqrt(sum of squared errors + 1e-8)` over that group's dimensions (`t`, `R`, `scale`, `eps` masked over active GT slots; `exist` averaged over all slots), keeping the same `parts` dictionary keys and leaving the `x1` loss in `mix` mode unchanged.
3. **Checkpoint recording & rebuilding**: Recorded `flow_fourier`, `flow_vel_loss`, and `flow_slot_sinus` in `args.json` / `ckpt["args"]` via `train.py` and updated `l2sq/eval_occlusion.py` and `l2sq/diagnose_flow.py` to rebuild `FlowSQDenoiser` with `slot_init` and `fourier_freqs` using `.get(...)` defaults (`False` / `0`) so old checkpoints load unchanged. (`l2sq/predict.py` only instantiates `SQDecoder` and does not load `FlowSQDenoiser`.)
4. **CPU tests**: Added `test_fourier_and_l2_vel_loss()` in `l2sq/test_flow_matching_decoder.py` and ran the full CPU test suite.

---

## 2. Where Changes Were Made (Old vs New Code)

### 2.1 `l2sq/flow_matching_decoder.py`

- **Lines 129–141 (`FlowSQDenoiser.__init__`)**:

**Old:**
```python
    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True,
                 slot_init: str = "small"):
        super().__init__()
        self.deform = False
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        self.x_proj = nn.Linear(SQ_DIM, d_model)
```

**New:**
```python
    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True,
                 slot_init: str = "small", fourier_freqs: int = 0):
        super().__init__()
        if fourier_freqs < 0:
            raise ValueError(f"fourier_freqs must be >= 0, got {fourier_freqs}")
        self.deform = False
        self.fourier_freqs = fourier_freqs
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        in_dim = SQ_DIM * (1 + 2 * fourier_freqs)
        self.x_proj = nn.Linear(in_dim, d_model)
```

- **Lines 191–203 (`FlowSQDenoiser.forward`)**:

**Old:**
```python
        x = self.x_proj(x_t) + self.slot_emb[:, :x_t.shape[1]]
        for blk in self.blocks:
            x = blk(x, mem, cond)
        return self.out_proj(self.out_norm(x)) + self.t_scale(cond).unsqueeze(1) * self.x_skip(x_t)
```

**New:**
```python
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
```

- **Lines 206–238 (`flow_loss`)**:

**Old:**
```python
def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None, repeat: int = 1,
              mode: str = "velocity", w_x1: float = 1.0) -> tuple[torch.Tensor, dict]:
    """CondOT velocity MSE, optionally mixed with t-weighted decoded x1_hat parameter MSE + BCE."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if mode not in ("velocity", "mix"):
        raise ValueError(f"unknown flow loss mode {mode!r}")
...
    err2 = (v_pred - (x_1 - x_0)) ** 2
    n = m.sum().clamp_min(1.0)
    parts = {k: (err2[..., sl].mean(-1) * m).sum() / n for k, sl in GROUP_SLICES.items() if k != "exist"}
    parts["exist"] = err2[..., 14].mean()
    v_loss = sum(parts.values())
```

**New:**
```python
def flow_loss(denoiser: FlowSQDenoiser, tokens: torch.Tensor, tgt: dict,
              generator: torch.Generator | None = None, repeat: int = 1,
              mode: str = "velocity", w_x1: float = 1.0,
              vel_loss: str = "mse") -> tuple[torch.Tensor, dict]:
    """CondOT velocity loss ('mse' or Euclidean 'l2'), optionally mixed with t-weighted decoded x1_hat MSE + BCE."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if mode not in ("velocity", "mix"):
        raise ValueError(f"unknown flow loss mode {mode!r}")
    if vel_loss not in ("mse", "l2"):
        raise ValueError(f"unknown vel_loss {vel_loss!r}")
...
    err2 = (v_pred - (x_1 - x_0)) ** 2
    n = m.sum().clamp_min(1.0)
    if vel_loss == "l2":
        parts = {k: (torch.sqrt(err2[..., sl].sum(dim=-1) + 1e-8) * m).sum() / n
                 for k, sl in GROUP_SLICES.items() if k != "exist"}
        parts["exist"] = torch.sqrt(err2[..., 14:15].sum(dim=-1) + 1e-8).mean()
    else:
        parts = {k: (err2[..., sl].mean(-1) * m).sum() / n for k, sl in GROUP_SLICES.items() if k != "exist"}
        parts["exist"] = err2[..., 14].mean()
    v_loss = sum(parts.values())
```

---

### 2.2 `l2sq/train.py`

- **Lines 164–167 (`evaluate`)**:
  - Old: `loss, _ = flow_loss(dec, tokens, tgt, generator=gen)`
  - New: `loss, _ = flow_loss(dec, tokens, tgt, generator=gen, vel_loss=cfg.get("flow_vel_loss", "mse"))`
- **Lines 272–275 (`parse_args`)**:
  - Added `--flow-fourier` (`type=int, default=0`) and `--flow-vel-loss` (`choices=["mse", "l2"], default="mse"`).
- **Lines 352–353 (`cfg`)**:
  - Added `"flow_vel_loss": a.flow_vel_loss` (while `(out / "args.json").write_text(json.dumps(vars(a), indent=2))` at line 351 already records `flow_slot_sinus`, `flow_fourier`, and `flow_vel_loss`).
- **Lines 374–376 (`FlowSQDenoiser` init)**:
  - Old:
    ```python
    dec = FlowSQDenoiser(a.d_model, a.queries, a.layers, a.heads, a.ffn,
                         slot_init="sinus" if a.flow_slot_sinus else "small").to(device)
    ```
  - New:
    ```python
    dec = FlowSQDenoiser(a.d_model, a.queries, a.layers, a.heads, a.ffn,
                         slot_init="sinus" if a.flow_slot_sinus else "small",
                         fourier_freqs=a.flow_fourier).to(device)
    ```
- **Lines 474–476 (training loop)**:
  - Old: `loss, parts = flow_loss(dec, tokens, tgt, repeat=a.flow_repeat, mode=a.flow_loss, w_x1=a.w_x1)`
  - New: `loss, parts = flow_loss(dec, tokens, tgt, repeat=a.flow_repeat, mode=a.flow_loss, w_x1=a.w_x1, vel_loss=a.flow_vel_loss)`

---

### 2.3 Checkpoint Rebuilding (`l2sq/eval_occlusion.py`, `l2sq/diagnose_flow.py`)

Both scripts read `ckpt["args"]` when instantiating `FlowSQDenoiser`:
- `slot_init="sinus" if args.get("flow_slot_sinus", False) else "small"`
- `fourier_freqs=args.get("flow_fourier", 0)`

Because `.get(..., False)` and `.get(..., 0)` default to `"small"` and `0`, old checkpoints without `flow_slot_sinus` or `flow_fourier` in `args` still build `x_proj` with `15` input features and `slot_emb` matching the saved state dict, while new checkpoints rebuild the exact matching `x_proj` input size `15 * (1 + 2 * flow_fourier)` before calling `load_state_dict(ckpt["decoder"])`.

- **`l2sq/eval_occlusion.py` lines 45–50 (`load_decoder`)**:
  ```python
    elif kind == "flow":
        dec = FlowSQDenoiser(
            args["d_model"], args["queries"], args["layers"], args["heads"], args["ffn"],
            slot_init="sinus" if args.get("flow_slot_sinus", False) else "small",
            fourier_freqs=args.get("flow_fourier", 0),
        )
  ```
- **`l2sq/diagnose_flow.py` lines 97–101 (`main`)**:
  ```python
    dec = FlowSQDenoiser(
        args["d_model"], P, args["layers"], args["heads"], args["ffn"],
        slot_init="sinus" if args.get("flow_slot_sinus", False) else "small",
        fourier_freqs=args.get("flow_fourier", 0),
    ).to(a.device).eval()
  ```

---

### 2.4 `l2sq/test_flow_matching_decoder.py`

Added `test_fourier_and_l2_vel_loss()` (lines 404–450) and invoked it in `__main__` (line 461):
- Verifies `FlowSQDenoiser(..., slot_init="sinus", fourier_freqs=4)` has `x_proj.in_features == 135` (`15 * (1 + 2 * 4)`) and returns finite output of shape `(B, 16, 15)`.
- Verifies `load_decoder` in `l2sq/eval_occlusion.py` reloads both new checkpoints (with `flow_slot_sinus=True`, `flow_fourier=4`, `flow_vel_loss="l2"`) and legacy checkpoints (missing those keys).
- Verifies `flow_loss(..., vel_loss="l2")` (for both `mode="velocity"` and `mode="mix"`) returns finite loss, finite gradients, the exact same `parts` group keys, and a different loss value from `vel_loss="mse"`.

---

## 3. CPU Test Output

Command:
```bash
cd /Users/syedrayyanadil/Desktop/SuperQuadratics-TUM-DI-LAB/latent2SQ-v2
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder
```

Output (exit code `0`):
```text
=== Overfit 1 fixed batch (500 steps) ===
  step   1: loss=9.1109  exist=2.2268  t=2.1510  R=1.5065  scale=1.5302  eps=1.6965
  step 100: loss=3.0964  exist=0.3121  t=0.6201  R=0.9100  scale=0.4938  eps=0.7604
  step 200: loss=5.1138  exist=1.5099  t=1.0987  R=1.0841  scale=0.6896  eps=0.7315
  step 300: loss=0.6886  exist=0.2083  t=0.0660  R=0.1778  scale=0.0837  eps=0.1529
  step 400: loss=0.2093  exist=0.0768  t=0.0193  R=0.0424  scale=0.0245  eps=0.0463
  step 500: loss=0.0539  exist=0.0084  t=0.0107  R=0.0206  scale=0.0047  eps=0.0094
  eval@t=0.5: loss=0.0576  exist=0.0149

=== Single-batch overfit with K=4 (200 steps) ===
  step   1: loss=8.3332  exist=1.9191
  step  50: loss=2.6052  exist=0.4623
  step 100: loss=1.1578  exist=0.1339
  step 150: loss=1.5679  exist=0.2796
  step 200: loss=0.3511  exist=0.0439

=== Single-batch overfit with --flow-loss mix (200 steps) ===
  step   1: total=9.3843  v_loss=8.4117  x1_loss=0.9726
  step  50: total=3.5579  v_loss=3.3020  x1_loss=0.2559
  step 100: total=1.3151  v_loss=1.0926  x1_loss=0.2225
  step 150: total=0.7135  v_loss=0.6606  x1_loss=0.0529
  step 200: total=0.9394  v_loss=0.8955  x1_loss=0.0439

=== Single-batch overfit with --flow-t-norm on (200 steps) ===
  step   1: total=10.1270  v_loss=9.2225  x1_loss=0.9045
  step  50: total=4.4845  v_loss=4.2294  x1_loss=0.2551
  step 100: total=2.1622  v_loss=1.9311  x1_loss=0.2312
  step 150: total=0.9957  v_loss=0.9442  x1_loss=0.0516
  step 200: total=0.9520  v_loss=0.9153  x1_loss=0.0367

=== Single-batch overfit with --flow-norm-all (200 steps) ===
  step   1: total=12.5213  v_loss=12.2311  x1_loss=0.2902
  step  50: total=6.3013  v_loss=6.1319  x1_loss=0.1694
  step 100: total=4.2830  v_loss=4.2514  x1_loss=0.0316
  step 150: total=1.1157  v_loss=1.0920  x1_loss=0.0236
  step 200: total=1.0335  v_loss=1.0175  x1_loss=0.0160

=== train.py --decoder flow CPU wiring test (1 train step + eval) ===
loading views and pseudo-GT ...
GT geometry: 256 surface points, 256 volume points (9.8% inside); loss: mse
train views 8, test views 8, GT superquadrics 7, P=16
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0)}
decoder parameters: 0.26 M
step     1  loss 8.379  t 1.46  R 1.854  scale 1.642  eps 1.716  exist 1.707  lr 0.00e+00  0.03s/step  vggt 0.002s/step  dec 0.006s/step  peak 0.0 GB
[eval train step 1] loss 9.4286  rot 123.4deg  t 3.7401  scale 1272.9%  chamfer 3.7531  IoU 0.000  chamfer-gt 3.4224  exist-acc 0.594  IoU-target 0.167  #SQ 5.0/7  empty 0
[eval test  step 1] loss 9.5831  rot 124.8deg  t 3.7471  scale 1253.5%  chamfer 3.3061  IoU 0.000  chamfer-gt 3.0776  exist-acc 0.625  IoU-target 0.167  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 126.7deg  t 3.0524  scale 2403.0%  chamfer 5.4413  IoU 0.000  chamfer-gt 5.3199  #SQ 6.0
  train N=2: rot 126.1deg  t 3.2707  scale 970.2%  chamfer 2.3123  IoU 0.000  chamfer-gt 2.1440  #SQ 6.0
  train N=4: rot 115.0deg  t 4.0808  scale 1189.6%  chamfer 3.3418  IoU 0.000  chamfer-gt 3.0245  #SQ 6.0
  train N=8: rot 125.8deg  t 4.5566  scale 528.7%  chamfer 3.9170  IoU 0.000  chamfer-gt 3.2012  #SQ 2.0
  test  N=1: rot 129.4deg  t 3.3561  scale 1338.3%  chamfer 4.6319  IoU 0.000  chamfer-gt 4.3909  #SQ 6.0
  test  N=2: rot 124.1deg  t 3.1386  scale 2028.4%  chamfer 2.2636  IoU 0.000  chamfer-gt 2.1805  #SQ 6.0
  test  N=4: rot 117.4deg  t 4.0286  scale 1064.0%  chamfer 3.2193  IoU 0.000  chamfer-gt 2.8868  #SQ 6.0
  test  N=8: rot 128.5deg  t 4.4651  scale 583.5%  chamfer 3.1095  IoU 0.000  chamfer-gt 2.8524  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpgomliiev/run
loading views and pseudo-GT ...
GT geometry: 256 surface points, 256 volume points (13.7% inside); loss: mse
train views 8, test views 8, GT superquadrics 7, P=16
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0)}
flow t_norm (64 train samples): mean=[0.0420462004840374, -0.12782062590122223, 0.46166229248046875]  std=[0.1473735272884369, 0.19727975130081177, 0.24463875591754913]
decoder parameters: 0.26 M
step     1  loss 9.695  t 2.618  R 1.822  scale 1.762  eps 1.859  exist 1.634  lr 0.00e+00  0.02s/step  vggt 0.002s/step  dec 0.006s/step  peak 0.0 GB
[eval train step 1] loss 9.7484  rot 123.4deg  t 0.9013  scale 1272.9%  chamfer 1.7483  IoU 0.192  chamfer-gt 1.7410  exist-acc 0.594  IoU-target 0.000  #SQ 5.0/7  empty 0
[eval test  step 1] loss 9.8089  rot 124.9deg  t 0.9068  scale 1253.6%  chamfer 1.4828  IoU 0.135  chamfer-gt 1.5615  exist-acc 0.625  IoU-target 0.000  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 126.7deg  t 0.7982  scale 2403.1%  chamfer 4.1975  IoU 0.158  chamfer-gt 4.2179  #SQ 6.0
  train N=2: rot 126.1deg  t 0.7739  scale 970.2%  chamfer 0.4144  IoU 0.286  chamfer-gt 0.4931  #SQ 6.0
  train N=4: rot 115.0deg  t 0.9995  scale 1189.6%  chamfer 1.8181  IoU 0.158  chamfer-gt 1.9139  #SQ 6.0
  train N=8: rot 125.8deg  t 1.0334  scale 528.7%  chamfer 0.5632  IoU 0.167  chamfer-gt 0.3389  #SQ 2.0
  test  N=1: rot 129.4deg  t 0.8445  scale 1338.3%  chamfer 3.3821  IoU 0.143  chamfer-gt 3.4110  #SQ 6.0
  test  N=2: rot 124.1deg  t 0.7694  scale 2028.4%  chamfer 0.4515  IoU 0.231  chamfer-gt 0.5670  #SQ 6.0
  test  N=4: rot 117.4deg  t 0.9966  scale 1064.0%  chamfer 1.7325  IoU 0.167  chamfer-gt 1.8091  #SQ 6.0
  test  N=8: rot 128.5deg  t 1.0166  scale 583.5%  chamfer 0.3650  IoU 0.000  chamfer-gt 0.4590  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpgomliiev/run_tnorm
loading views and pseudo-GT ...
GT geometry: 256 surface points, 256 volume points (10.2% inside); loss: mse
train views 8, test views 8, GT superquadrics 7, P=16
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0)}
flow norm_all (64 train samples, std>=0.05): {'t': ([0.042, -0.128, 0.462], [0.147, 0.197, 0.245]), 'rot6d': ([0.248, -0.187, 0.198, -0.343, -0.001, 0.37], [0.49, 0.653, 0.444, 0.432, 0.574, 0.479]), 'log_scale': ([-2.868, -2.933, -2.714], [0.834, 0.582, 0.67]), 'eps': ([-1.109, -0.02], [0.459, 0.863]), 'exist': ([0.5], [0.5])}
decoder parameters: 0.26 M
step     1  loss 11.71  t 2.678  R 2.381  scale 2.334  eps 2.626  exist 1.69  lr 0.00e+00  0.02s/step  vggt 0.002s/step  dec 0.013s/step  peak 0.0 GB
[eval train step 1] loss 11.3734  rot 115.0deg  t 0.9013  scale 131.9%  chamfer 0.3955  IoU 0.047  chamfer-gt 0.4069  exist-acc 0.594  IoU-target 0.111  #SQ 5.0/7  empty 0
[eval test  step 1] loss 11.3670  rot 121.0deg  t 0.9068  scale 139.0%  chamfer 0.3743  IoU 0.094  chamfer-gt 0.4122  exist-acc 0.625  IoU-target 0.111  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 113.8deg  t 0.7983  scale 186.4%  chamfer 0.3905  IoU 0.077  chamfer-gt 0.4097  #SQ 6.0
  train N=2: rot 116.7deg  t 0.7740  scale 110.5%  chamfer 0.3554  IoU 0.111  chamfer-gt 0.3986  #SQ 6.0
  train N=4: rot 111.4deg  t 0.9995  scale 121.8%  chamfer 0.2478  IoU 0.000  chamfer-gt 0.4659  #SQ 6.0
  train N=8: rot 118.0deg  t 1.0334  scale 109.0%  chamfer 0.5883  IoU 0.000  chamfer-gt 0.3535  #SQ 2.0
  test  N=1: rot 132.7deg  t 0.8445  scale 170.7%  chamfer 0.4252  IoU 0.250  chamfer-gt 0.3537  #SQ 6.0
  test  N=2: rot 117.7deg  t 0.7694  scale 153.5%  chamfer 0.3524  IoU 0.125  chamfer-gt 0.3951  #SQ 6.0
  test  N=4: rot 115.6deg  t 0.9966  scale 118.7%  chamfer 0.2577  IoU 0.000  chamfer-gt 0.4644  #SQ 6.0
  test  N=8: rot 117.9deg  t 1.0166  scale 113.1%  chamfer 0.4618  IoU 0.000  chamfer-gt 0.4355  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpgomliiev/run_norm_all
empty-pred eval check: n_empty=2, chamfer=nan

ALL CPU FLOW TESTS PASSED
```
