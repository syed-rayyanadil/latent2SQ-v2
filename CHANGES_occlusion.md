# Change Report: Occlusion Evaluation (`l2sq/eval_occlusion.py`) & Training Augmentation (`l2sq/train.py`)

## Overview
- **Part A (Eval only)**: Extended `l2sq/eval_occlusion.py` to support current flow checkpoints (`flow_deform`, `flow_slot_sinus`, `flow_fourier`, `flow_geo`, `sq_dim=26` for deform), evaluate multiple flow checkpoints (`runs/flow_geoC100XL`, `runs/flow_deform8L`) alongside one `--parallel` baseline in a single run, sample `K=8` times per input by default, pick `best-of-K` by Chamfer, and compute sample spread as mean pairwise Chamfer across the `K` predictions.
- **Part B (Training augmentation)**: Added `--occlusion-p` (`0.0` = off default) and `--occlusion-prob` (`0.5` default) to `l2sq/train.py`, reusing `apply_box_occlusion` cleanly without circular imports, masking each training view with probability `Q` using a random gray-`0.5` box of area fraction in `[0.25, 0.75]` before VGGT while leaving teacher targets and evaluation/export views untouched.

---

## What Changed & Where

1. **`l2sq/data.py`**
   - **Lines 166–184 (`apply_box_occlusion`)**: Moved `apply_box_occlusion` here so both `l2sq/eval_occlusion.py` and `l2sq/train.py` can share it without circular imports (`eval_occlusion` still imports and re-exports `apply_box_occlusion`).
   - **Lines 187–200 (`augment_occlusion`)**: For each view `(b, v)` independently with probability `prob`, samples `p_box ~ Uniform(0.25, 0.75)` and masks that view using `apply_box_occlusion(..., fill=0.5)`.

2. **`l2sq/eval_occlusion.py`**
   - **Lines 34–77 (`_resolve_ckpt_path`, `_model_label`, `load_decoder`)**:
     - Accepts either `<run_dir>/decoder.pt` or `<run_dir>`.
     - Uses `weights_only=False` and rebuilds `FlowSQDenoiser` identically to `train.py` (`slot_init="sinus" if args.get("flow_slot_sinus", False) else "small"`, `fourier_freqs=args.get("flow_fourier", 0)`, `deform=args.get("flow_deform", False)` -> `sq_dim=26` when `flow_deform=True`), plus `t_mean`/`t_std` and `flow_shift`/`flow_scale` buffer fallbacks.
     - Reads and logs `flow_geo`, `flow_slot_sinus`, `flow_fourier`, `flow_deform`, `sq_dim`, and per-checkpoint `flow_steps` (unless overridden by `--flow-steps`).
   - **Lines 31, 98–133, 223–303 (`METRIC_KEYS`, `_per_sample_metrics`, `_pairwise_chamfer_world`, `_eval_batches`)**:
     - Added `chamfer_gt` (vs `tgt["surf"]`) alongside `chamfer_world` (vs teacher SQs).
     - Extended `--flow` (`nargs="+"`, default `["runs/flow_geoC100XL/decoder.pt", "runs/flow_deform8L/decoder.pt"]`) and changed default `--k` from `5` to `8`.
     - For each flow model on shared VGGT tokens:
       1. Reports mean metrics over the `K` draws (`stat="mean"`).
       2. Selects `best-of-K` by minimum `chamfer_world` (instead of `t_err_world`), handling all-empty draws safely (`stat="best_of_8"`).
       3. Computes `chamfer_spread_world` = mean pairwise Chamfer distance (world units) across all valid `(i, j)` pairs among the `K` sampled point clouds (reusing already-sampled `even_points`), while keeping `t_spread_world` as well.
   - **Lines 190–220, 306–360, 382–470 (`_retarget_batch`, `_format_table`, `_append_csv_rows`, `evaluate_occlusion`)**:
     - Supports evaluating rigid and deformable checkpoints together (retargeting SQ GT fields per model via `_retarget_batch` when `gt_sq` differs, e.g., `gt_sq_rigid.npz` vs `gt_sq_deform.npz`, on the exact same views, `images`, and `GeomTargets`).
     - Prints compact `box` and `half` tables with one row per `(condition, N, model)` showing mean metrics, `best-K cd`, and `pair_cd` / `t_std`, and writes all rows (`parallel`, plus `mean` and `best_of_K` per flow model) to CSV.

3. **`l2sq/train.py`**
   - **Lines 40, 291–294**: Added `--occlusion-p` (`float`, default `0.0` = off) and `--occlusion-prob` (`float`, default `0.5`); both are recorded in `args.json`.
   - **Lines 493–496**: When `a.occlusion_p > 0.0`, applies `augment_occlusion(tgt["images"], a.occlusion_prob, rng)` before `vggt(images)`. Teacher targets (`tgt`) stay unchanged, and `evaluate()` / `export_examples()` never apply occlusion.

4. **`l2sq/test_eval_occlusion.py` & `l2sq/test_flow_matching_decoder.py`**
   - **`l2sq/test_eval_occlusion.py` (lines 136–241)**: Added `test_current_flow_checkpoints_and_multi_eval` (verifies `load_decoder` on `flow_slot_sinus`, `flow_fourier`, `flow_deform=True`/`sq_dim=26`, `flow_geo`, run-dir loading, and multi-model `main()` with 1 parallel + rigid XL + deform 8L flow at `K=8` producing 180 CSV rows, finite `chamfer_spread_world`, `chamfer_gt`, and `best_of_8 chamfer_world <= mean chamfer_world`) and `test_augment_occlusion`.
   - **`l2sq/test_flow_matching_decoder.py` (lines 251–295)**: Verified `--occlusion-p 0.0` / `--occlusion-prob 0.5` saved in `args.json` with default identity, and `--occlusion-p 0.5 --occlusion-prob 1.0` occluding every training view in `[0.25, 0.75]` before `vggt` while keeping `evaluate()` and `export_examples()` views clean.

---

## Old vs. New Behavior

| Component | Old Behavior | New Behavior |
|---|---|---|
| `eval_occlusion.py` `--flow` | Single `--flow` checkpoint only, `--k` default `5` | `nargs="+"` (supports multiple checkpoints e.g. `runs/flow_geoC100XL` `runs/flow_deform8L`), `--k` default `8`, accepts dirs or `.pt` files |
| `eval_occlusion.py` GT handling | Single shared `gt_sq` file | Per-model `gt_sq` from `ckpt["args"]` when `--gt-sq` is not forced, supporting mixed rigid + `flow_deform` (`sq_dim=26`) evaluation |
| `eval_occlusion.py` `best-of-K` | Selected `best_k` by `t_err_world` | Selects `best_k` by `chamfer_world` (best-of-K Chamfer) |
| `eval_occlusion.py` spread | `t_spread_world` (translation std only) | Mean pairwise Chamfer across `K` predictions (`chamfer_spread_world`, `pair_cd`), plus `t_spread_world` and `chamfer_gt` |
| `train.py` occlusion aug | Not supported | Opt-in via `--occlusion-p > 0` (`--occlusion-prob` default `0.5`); off by default (`0.0`) with identical behavior to before |

---

## How to Run

### A) Evaluate Current Flow Checkpoints vs. Parallel Baseline (GPU)
Pass your regression baseline path as `--parallel` and the two flow checkpoints as `--flow` (supports either run directories or `decoder.pt` paths; evaluates box occlusion `p in {0, 0.25, 0.5, 0.75}` and half occlusion for `N in {1, 4, 8}` with `K=8` samples):

```bash
python -u -m l2sq.eval_occlusion \
    --parallel runs/chair_L1_logscale/decoder.pt \
    --flow runs/flow_geoC100XL/decoder.pt runs/flow_deform8L/decoder.pt \
    --k 8 \
    --out runs/occlusion_eval/results.csv
```

If needed, `--test <path>`, `--gt-world <path>`, `--batch <B>`, `--per-n <M>`, or `--flow-steps <S>` can override checkpoint metadata.

### B) Train with Random Box Occlusion Augmentation (GPU)
By default (`--occlusion-p 0.0`), occlusion augmentation is off. Enable it by setting `--occlusion-p > 0` (e.g. `0.5`) and optionally `--occlusion-prob` (default `0.5`):

```bash
python -u -m l2sq.train \
    --train dataset/chair/train --test dataset/chair/test \
    --gt-sq dataset/chair/gt_sq_deform.npz \
    --decoder flow --flow-loss mix --flow-deform --flow-norm-all \
    --flow-slot-sinus --flow-fourier 4 --flow-geo chamfer \
    --occlusion-p 0.5 --occlusion-prob 0.5 \
    --out runs/flow_deform_occ
```

---

## CPU Test Output
Ran:
```bash
PYTHONPATH=. python3 -m l2sq.test_eval_occlusion
PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder
```

Results (`Command exited with code 0` for both):
```text
Saved 108 rows -> .../results.csv
...
Saved 180 rows -> .../multi_results.csv
ALL OCCLUSION EVAL SMOKE TESTS PASSED
```
```text
saved .../run_occ_aug
ALL CPU FLOW TESTS PASSED
```
