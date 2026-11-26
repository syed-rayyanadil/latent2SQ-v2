# Change Report: Picture & SQ Mesh Export in `l2sq/eval_occlusion.py`

## Overview
Extended `l2sq/eval_occlusion.py` with optional visual and 3D mesh export (`--export-dir` and related flags) so occluded inputs, clean inputs, comparison sheets, ground-truth SQ meshes, and per-draw predicted SQ `.ply` meshes can be saved directly alongside metric evaluation without changing default outputs, tables, CSV rows, or RNG streams.

---

## What Changed & Where

1. **`l2sq/train.py`**
   - **Lines 211–215 (`export_sq_ply`)**: Extracted the world-unit SQ `.ply` export call (`sq_args` + `sq_mesh(...).export(path)`) into a small shared helper.
   - **Lines 219–245 (`export_examples`)**: Updated to call `export_sq_ply`, preserving identical `export_examples` behavior.

2. **`l2sq/eval_occlusion.py`**
   - **Lines 230–247 (`_tensor_to_pil`, `_save_occlusion_images`)**: Converts `[0, 1]` RGB float image tensors `(3, H, W)` to 8-bit PNGs (gray `0.5` maps to `128`) and writes `input_clean_<v>.png`, `input_occluded_<v>.png`, and a side-by-side 2-row `input_compare.png` (`N*W x 2*H`, clean top row, occluded bottom row).
   - **Lines 250–309 (`_export_sample_bundle`)**: Reuses `export_sq_ply` to save `gt_sq.ply` and `pred_sq_draw<j>.ply` (keeping slots with `exist > 0.5`, matching `export_examples`'s non-empty fallback when all slots fall below `0.5`) plus `sample.json` containing `model`, `setting`, `mode`, `p`, `side`, `N`, `sample_idx`, `views`, `mask_seed`, `scale_s`, `n_pred` per draw, and per-draw metrics (`rot_deg`, `t_err_world`, `scale_rel_err`, `chamfer_world`, `chamfer_gt`, `iou`, `exist_acc`).
   - **Lines 313–437 (`_eval_batches`) & Lines 522–600 (`evaluate_occlusion`)**: Hooks export directly into the evaluation loop when `export_ctx` matches the current `(setting, N)` and sample index `< --export-samples`, so the saved images and meshes come from the exact same occluded inputs, fixed test batches, `mask_seed` RNG state, and `sample_euler` draws that produced the CSV/table metrics. When `--export-dir` is `None` (default), `export_ctx` is `None` and execution/RNG order is identical to before.
   - **Lines 661–666 (`main`)**: Added the new optional export CLI arguments and forwarded them to `evaluate_occlusion`.

3. **`l2sq/test_eval_occlusion.py`**
   - **Lines 247–358 (`test_occlusion_export`)**: Added CPU tests with tiny `SQDecoder` / `FlowSQDenoiser` models verifying:
     1. Running without `--export-dir` writes no export files and produces byte-for-byte identical CSV output to running with `--export-dir` enabled.
     2. Enabling `--export-dir`, `--export-p`, `--export-halves`, `--export-n`, `--export-samples`, and `--export-draws` creates all expected PNGs, `.ply` meshes, and `sample.json` files.
     3. `input_occluded_<v>.png` equals `input_clean_<v>.png` everywhere outside the occluded box/half and equals gray `128` inside it; `input_compare.png` matches the top/bottom concatenation of clean and occluded views.

---

## New CLI Flags (All Optional, Default Off)

| Flag | Default | Description |
|---|---|---|
| `--export-dir DIR` | `None` | Output directory for images/meshes (`None` = no export) |
| `--export-p P ...` | `0.5` | Box area fractions to export (subset of `0, 0.25, 0.5, 0.75`) |
| `--export-halves` | `False` | Also export all 4 half-occlusion settings (`left`, `right`, `top`, `bottom`) |
| `--export-n N ...` | `1 4` | View counts `N` to export (subset of `1, 4, 8`) |
| `--export-samples K` | `3` | Number of test samples per `N` and setting to export |
| `--export-draws J` | `3` | Number of flow draws to save per input (`parallel` always writes `pred_sq_draw0.ply`) |

---

## Folder Layout

```text
<export_dir>/
  <model_label>/                 # e.g. parallel, flow_geoC100XL, flow_geoC100Occ
    <setting>_N<n>_<k>/          # e.g. p0.5_N1_0, p0.25_N4_2, half_left_N1_0
      input_clean_0.png ... input_clean_<N-1>.png
      input_occluded_0.png ... input_occluded_<N-1>.png
      input_compare.png          # top row clean views, bottom row occluded views
      gt_sq.ply                  # teacher SQ mesh in world units
      pred_sq_draw0.ply          # predicted SQ mesh (draw 0; only pred_sq_draw0.ply for parallel)
      ...                        # pred_sq_draw<j>.ply for j = 0..--export-draws-1 (flow models)
      sample.json                # views, setting, N, mask_seed, scale_s, n_pred, per-draw metrics
```

---

## Example Cluster Commands

Export box occlusions (`0.25`, `0.5`, `0.75`) and half occlusions for `N in {1, 4}` (`3` samples each, `3` flow draws each) comparing your `--parallel` baseline, `runs/flow_geoC100XL`, and `runs/flow_geoC100Occ`:

```bash
python -u -m l2sq.eval_occlusion \
    --parallel runs/chair_L1_logscale/decoder.pt \
    --flow runs/flow_geoC100XL/decoder.pt runs/flow_geoC100Occ/decoder.pt \
    --k 8 \
    --out runs/occlusion_eval/results.csv \
    --export-dir runs/occlusion_export \
    --export-p 0.25 0.5 0.75 \
    --export-halves \
    --export-n 1 4 \
    --export-samples 3 \
    --export-draws 3
```

Metrics-only evaluation (defaults unchanged; no pictures/meshes exported):

```bash
python -u -m l2sq.eval_occlusion \
    --parallel runs/chair_L1_logscale/decoder.pt \
    --flow runs/flow_geoC100XL/decoder.pt runs/flow_geoC100Occ/decoder.pt \
    --k 8 \
    --out runs/occlusion_eval/results.csv
```

---

## Defaults & Metrics Confirmation + Test Results

- Confirmed defaults (`--export-dir None`) produce **exact bit-for-bit identical** CSV rows (`assert csv_no_exp.read_text() == csv_with_exp.read_text()`), tables, and RNG sequences as before.
- Ran CPU test suites:
  - `PYTHONPATH=. python3 -m l2sq.test_eval_occlusion` → exited with code `0` (`ALL OCCLUSION EVAL SMOKE TESTS PASSED`, including `test_occlusion_export` with mocked `sq_mesh` since `trimesh` is absent in the local Mac CPU environment).
  - `PYTHONPATH=. python3 -m l2sq.test_flow_matching_decoder` → exited with code `0` (`ALL CPU FLOW TESTS PASSED`).
