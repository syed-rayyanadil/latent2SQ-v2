# Change Report: `slot_init` for `FlowSQDenoiser`

## 1. What Changed

Added a switchable sinusoidal slot-embedding initialization (`slot_init: str = "small"`) to `FlowSQDenoiser`, exposed it in `l2sq/train.py` via `--flow-slot-sinus` (default off, preserving existing `N(0, 0.02^2)` initialization and all other behavior unchanged), and added a CPU unit test in `l2sq/test_flow_matching_decoder.py`.

---

## 2. Where Changes Were Made

### 2.1 `l2sq/flow_matching_decoder.py`
- **Line 11**: imported `sinusoidal` alongside `TOKEN_DIM` from `l2sq.model`.
- **Lines 129–143**: added `slot_init: str = "small"` parameter to `FlowSQDenoiser.__init__` and selected either `torch.randn(1, n_queries, d_model) * 0.02` (`"small"`, default) or `sinusoidal(n_queries, d_model).unsqueeze(0)` (`"sinus"`).

**Old code:**
```python
from l2sq.model import TOKEN_DIM
...
    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True):
        super().__init__()
        self.deform = False
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        self.x_proj = nn.Linear(SQ_DIM, d_model)
        self.slot_emb = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
```

**New code:**
```python
from l2sq.model import TOKEN_DIM, sinusoidal
...
    def __init__(self, d_model: int = 256, n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0, ref_embedding: bool = True,
                 slot_init: str = "small"):
        super().__init__()
        self.deform = False
        self.in_norm = nn.LayerNorm(TOKEN_DIM)
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.ref_embedding = nn.Parameter(torch.zeros(d_model)) if ref_embedding else None
        self.x_proj = nn.Linear(SQ_DIM, d_model)
        if slot_init == "small":
            self.slot_emb = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        elif slot_init == "sinus":
            self.slot_emb = nn.Parameter(sinusoidal(n_queries, d_model).unsqueeze(0))
        else:
            raise ValueError(f"unknown slot_init {slot_init!r}, expected 'small' or 'sinus'")
```

---

### 2.2 `l2sq/train.py`
- **Lines 270–271**: added `--flow-slot-sinus` (`action="store_true"`, default `False`), which is automatically recorded in `args.json` via `(out / "args.json").write_text(json.dumps(vars(a), indent=2))` at line 347.
- **Lines 370–371**: passed `slot_init="sinus" if a.flow_slot_sinus else "small"` to `FlowSQDenoiser(...)`.

**Old code:**
```python
    p.add_argument("--flow-norm-all", action="store_true",
                   help="per-dim (x - mean) / std over all SQ groups (t, rot6d, log-scale, eps logit) in flow space")
...
        dec = FlowSQDenoiser(a.d_model, a.queries, a.layers, a.heads, a.ffn).to(device)
```

**New code:**
```python
    p.add_argument("--flow-norm-all", action="store_true",
                   help="per-dim (x - mean) / std over all SQ groups (t, rot6d, log-scale, eps logit) in flow space")
    p.add_argument("--flow-slot-sinus", action="store_true",
                   help="initialize FlowSQDenoiser.slot_emb with sinusoidal embeddings instead of N(0, 0.02^2)")
...
        dec = FlowSQDenoiser(a.d_model, a.queries, a.layers, a.heads, a.ffn,
                             slot_init="sinus" if a.flow_slot_sinus else "small").to(device)
```

---

### 2.3 `l2sq/test_flow_matching_decoder.py`
- **Lines 390–409**: added `test_slot_init_sinus()` verifying `denoiser.slot_emb` matches `sinusoidal(16, 256)` and forward pass output shape is `(B, 16, 15)` with finite values, and called it from `__main__`.

**Old code:**
```python
if __name__ == "__main__":
    test_overfit_one_batch()
    test_flow_repeat()
    test_flow_loss_mix()
    test_flow_t_norm()
    test_flow_norm_all()
    test_train_py_flow_wiring()
    print("\nALL CPU FLOW TESTS PASSED")
```

**New code:**
```python
def test_slot_init_sinus() -> None:
    from l2sq.model import sinusoidal

    B = 2
    denoiser = FlowSQDenoiser(d_model=256, n_queries=16, n_layers=1, n_heads=8, ffn_dim=256, slot_init="sinus")
    expected = sinusoidal(16, 256)
    assert denoiser.slot_emb.shape == (1, 16, 256)
    assert torch.equal(denoiser.slot_emb.squeeze(0), expected)
    out = denoiser(torch.randn(B, 16, 15), torch.rand(B), tokens=torch.randn(B, 2, 8, TOKEN_DIM))
    assert out.shape == (B, 16, 15) and torch.isfinite(out).all()


if __name__ == "__main__":
    test_overfit_one_batch()
    test_flow_repeat()
    test_flow_loss_mix()
    test_flow_t_norm()
    test_flow_norm_all()
    test_slot_init_sinus()
    test_train_py_flow_wiring()
    print("\nALL CPU FLOW TESTS PASSED")
```

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
step     1  loss 8.379  t 1.46  R 1.854  scale 1.642  eps 1.716  exist 1.707  lr 0.00e+00  0.10s/step  vggt 0.003s/step  dec 0.040s/step  peak 0.0 GB
[eval train step 1] loss 9.4286  rot 123.4deg  t 3.7401  scale 1272.9%  chamfer 3.7531  IoU 0.000  chamfer-gt 3.4934  exist-acc 0.594  IoU-target 0.000  #SQ 5.0/7  empty 0
[eval test  step 1] loss 9.5831  rot 124.8deg  t 3.7471  scale 1253.5%  chamfer 3.3061  IoU 0.000  chamfer-gt 3.1060  exist-acc 0.625  IoU-target 0.000  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 126.7deg  t 3.0524  scale 2403.0%  chamfer 5.4413  IoU 0.000  chamfer-gt 5.3512  #SQ 6.0
  train N=2: rot 126.1deg  t 3.2707  scale 970.2%  chamfer 2.3123  IoU 0.000  chamfer-gt 2.1988  #SQ 6.0
  train N=4: rot 115.0deg  t 4.0808  scale 1189.6%  chamfer 3.3418  IoU 0.000  chamfer-gt 3.0988  #SQ 6.0
  train N=8: rot 125.8deg  t 4.5566  scale 528.7%  chamfer 3.9170  IoU 0.000  chamfer-gt 3.3248  #SQ 2.0
  test  N=1: rot 129.4deg  t 3.3561  scale 1338.3%  chamfer 4.6319  IoU 0.000  chamfer-gt 4.3756  #SQ 6.0
  test  N=2: rot 124.1deg  t 3.1386  scale 2028.4%  chamfer 2.2636  IoU 0.000  chamfer-gt 2.2248  #SQ 6.0
  test  N=4: rot 117.4deg  t 4.0286  scale 1064.0%  chamfer 3.2193  IoU 0.000  chamfer-gt 2.9429  #SQ 6.0
  test  N=8: rot 128.5deg  t 4.4651  scale 583.5%  chamfer 3.1095  IoU 0.000  chamfer-gt 2.8805  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpnv0zow07/run
loading views and pseudo-GT ...
GT geometry: 256 surface points, 256 volume points (10.5% inside); loss: mse
train views 8, test views 8, GT superquadrics 7, P=16
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0)}
flow t_norm (64 train samples): mean=[0.0420462004840374, -0.12782062590122223, 0.46166229248046875]  std=[0.1473735272884369, 0.19727975130081177, 0.24463875591754913]
decoder parameters: 0.26 M
step     1  loss 9.695  t 2.618  R 1.822  scale 1.762  eps 1.859  exist 1.634  lr 0.00e+00  0.02s/step  vggt 0.002s/step  dec 0.007s/step  peak 0.0 GB
[eval train step 1] loss 9.7484  rot 123.4deg  t 0.9013  scale 1272.9%  chamfer 1.7483  IoU 0.260  chamfer-gt 1.7618  exist-acc 0.594  IoU-target 0.000  #SQ 5.0/7  empty 0
[eval test  step 1] loss 9.8089  rot 124.9deg  t 0.9068  scale 1253.6%  chamfer 1.4828  IoU 0.193  chamfer-gt 1.5697  exist-acc 0.625  IoU-target 0.000  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 126.7deg  t 0.7982  scale 2403.1%  chamfer 4.1975  IoU 0.209  chamfer-gt 4.2317  #SQ 6.0
  train N=2: rot 126.1deg  t 0.7739  scale 970.2%  chamfer 0.4144  IoU 0.571  chamfer-gt 0.4874  #SQ 6.0
  train N=4: rot 115.0deg  t 0.9995  scale 1189.6%  chamfer 1.8181  IoU 0.167  chamfer-gt 1.9407  #SQ 6.0
  train N=8: rot 125.8deg  t 1.0334  scale 528.7%  chamfer 0.5632  IoU 0.091  chamfer-gt 0.3874  #SQ 2.0
  test  N=1: rot 129.4deg  t 0.8445  scale 1338.3%  chamfer 3.3821  IoU 0.214  chamfer-gt 3.4236  #SQ 6.0
  test  N=2: rot 124.1deg  t 0.7694  scale 2028.4%  chamfer 0.4515  IoU 0.412  chamfer-gt 0.5476  #SQ 6.0
  test  N=4: rot 117.4deg  t 0.9966  scale 1064.0%  chamfer 1.7325  IoU 0.146  chamfer-gt 1.8334  #SQ 6.0
  test  N=8: rot 128.5deg  t 1.0166  scale 583.5%  chamfer 0.3650  IoU 0.000  chamfer-gt 0.4740  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpnv0zow07/run_tnorm
loading views and pseudo-GT ...
GT geometry: 256 surface points, 256 volume points (11.7% inside); loss: mse
train views 8, test views 8, GT superquadrics 7, P=16
flow x_1 raw min/max: {'t': (-0.522664487361908, 0.6925036907196045), 'R': (-0.9779821634292603, 0.8160585761070251), 'scale': (-4.011168956756592, -1.8196702003479004), 'eps': (-1.528226613998413, 1.0690619945526123), 'exist': (0.0, 1.0)}
flow norm_all (64 train samples, std>=0.05): {'t': ([0.042, -0.128, 0.462], [0.147, 0.197, 0.245]), 'rot6d': ([0.248, -0.187, 0.198, -0.343, -0.001, 0.37], [0.49, 0.653, 0.444, 0.432, 0.574, 0.479]), 'log_scale': ([-2.868, -2.933, -2.714], [0.834, 0.582, 0.67]), 'eps': ([-1.109, -0.02], [0.459, 0.863]), 'exist': ([0.5], [0.5])}
decoder parameters: 0.26 M
step     1  loss 11.71  t 2.678  R 2.381  scale 2.334  eps 2.626  exist 1.69  lr 0.00e+00  0.02s/step  vggt 0.003s/step  dec 0.006s/step  peak 0.0 GB
[eval train step 1] loss 11.3734  rot 115.0deg  t 0.9013  scale 131.9%  chamfer 0.3955  IoU 0.019  chamfer-gt 0.3831  exist-acc 0.594  IoU-target 0.000  #SQ 5.0/7  empty 0
[eval test  step 1] loss 11.3670  rot 121.0deg  t 0.9068  scale 139.0%  chamfer 0.3743  IoU 0.063  chamfer-gt 0.3963  exist-acc 0.625  IoU-target 0.000  #SQ 5.5/7  empty 0

final (per N):
  train N=1: rot 113.8deg  t 0.7983  scale 186.4%  chamfer 0.3905  IoU 0.077  chamfer-gt 0.4403  #SQ 6.0
  train N=2: rot 116.7deg  t 0.7740  scale 110.5%  chamfer 0.3554  IoU 0.000  chamfer-gt 0.3590  #SQ 6.0
  train N=4: rot 111.4deg  t 0.9995  scale 121.8%  chamfer 0.2478  IoU 0.000  chamfer-gt 0.4425  #SQ 6.0
  train N=8: rot 118.0deg  t 1.0334  scale 109.0%  chamfer 0.5883  IoU 0.000  chamfer-gt 0.2905  #SQ 2.0
  test  N=1: rot 132.7deg  t 0.8445  scale 170.7%  chamfer 0.4252  IoU 0.154  chamfer-gt 0.3840  #SQ 6.0
  test  N=2: rot 117.7deg  t 0.7694  scale 153.5%  chamfer 0.3524  IoU 0.100  chamfer-gt 0.3497  #SQ 6.0
  test  N=4: rot 115.6deg  t 0.9966  scale 118.7%  chamfer 0.2577  IoU 0.000  chamfer-gt 0.4498  #SQ 6.0
  test  N=8: rot 117.9deg  t 1.0166  scale 113.1%  chamfer 0.4618  IoU 0.000  chamfer-gt 0.4017  #SQ 4.0
saved /var/folders/3s/d4w8zlg5201d1x0yfn4fnn8m0000gn/T/tmpnv0zow07/run_norm_all
empty-pred eval check: n_empty=2, chamfer=nan

ALL CPU FLOW TESTS PASSED
```
