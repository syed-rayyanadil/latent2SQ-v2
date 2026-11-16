# latent2SQ

Multi-view images → VGGT → superquadrics. This repo holds our scripts; [VGGT](https://github.com/facebookresearch/vggt) and [SuperFlex](https://github.com/GabrielTavernini/superflex) are git submodules.

```
latent2SQ/
├── dataset/          rendering, GT sampling          (python -m dataset.<script>)
│   └── <name>/       per-object outputs (gitignored)
├── recon/            VGGT, SuperFlex, viewer         (python -m recon.<script>)
├── l2sq/             SQ decoder on VGGT tokens       (python -m l2sq.train)
├── vggt/             submodule
├── superflex/        submodule
├── requirements.txt  our extra deps
└── constraints.txt   torch/numpy pins from VGGT
```

All commands run **from the repo root** on the server, with `conda activate l2sq`.

Steps 1–6 below are the original single-folder pipeline (`dataset/chair_test`, abandoned). The current setup is the [decoder sanity check](#decoder-sanity-check-single-object-mse) with `dataset/<obj>/{train,test}`.

---

## Setup (once)

Clone with both submodules:
```bash
git clone --recurse-submodules git@github.com:Namozbey/latent2SQ.git
cd latent2SQ
# already cloned? -> git submodule update --init
```

Conda env with VGGT (pins torch 2.3.1 + CUDA 12.1):
```bash
conda create -n l2sq python=3.10 -y
conda activate l2sq
pip install -r vggt/requirements.txt
pip install -e vggt
pip install -r requirements.txt -c constraints.txt   # pyrender, trimesh, tensorboard
```

SuperFlex deps — `-c constraints.txt` stops pip from upgrading torch/numpy, `-f` gets a prebuilt `torch-cluster`:
```bash
pip install -r superflex/requirements.txt -c constraints.txt -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install -e superflex --no-deps
gdown --folder 1Ospu8mFz4kEQf5a6pnOOx87z-xWR6kQe -O superflex/checkpoints/superflex   # ckpt.pt + config.yaml
```

Environment variables (put them in `~/.bashrc`):
```bash
export HF_HOME=/storage/user/hdo/hf_cache    # VGGT weights (~5 GB) go to storage, not $HOME
module load cuda/12.1                        # nvcc for SuperFlex's CUDA kernels (compiled on first run)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
```

---

## Pipeline

Example for one ShapeNet chair; outputs go to `dataset/chair_test/`.

### 1. Render views of a mesh
Renders N views around the mesh (headless, GPU) and saves the cameras in VGGT's convention.
```bash
MESH=$(ls ~/shapenet_partial_views/chair/*/mesh.npz | head -1)
python -m dataset.render --mesh "$MESH" --out dataset/chair_test --n-views 12 --elev 25 --save-mask
```
→ `images/000.png …` (518×518), `masks/`, `cameras.json` (K, per-view `w2c` in OpenCV convention, mesh normalization).
Options: `--mode random --elev-range -10 60` for random viewpoints, `--save-depth`, `--bg none` for RGBA, `--no-texture`.

### 2. VGGT point cloud
Runs VGGT-1B on all images of a folder and fuses depth + predicted cameras into one point cloud.
```bash
python -m recon.vggt_pointcloud --images dataset/chair_test/images --out dataset/chair_test/vggt.ply
```
→ `vggt.ply` (colored points), `vggt_pred.npz` (predicted extrinsics, intrinsics, depth, confidence).
Options: `--conf-percentile 20` keeps more points (default drops the lowest 50 %), `--source pointmap` uses the point-map head instead.

### 3. Ground-truth point cloud
Samples points uniformly on the mesh surface, in the same frame as VGGT's output (camera 0).
```bash
python -m dataset.sample_gt --cameras dataset/chair_test/cameras.json --out dataset/chair_test/gt.ply
```
→ `gt.ply` (points + normals), `gt.npz`.
Options: `-n 100000` points (default), `--frame world` for the upright, unit-sphere render frame.

### 4. Ground-truth superquadrics (SuperFlex)
Predicts superquadrics from the GT point cloud with SuperFlex, refines them against the mesh, and saves them in the GT's frame.
```bash
python -m recon.gt_superquadrics --gt dataset/chair_test/gt.npz \
    --cameras dataset/chair_test/cameras.json --out dataset/chair_test/gt_sq
```
→ `gt_sq.npz` (SuperFlex format: scale, rotation, translation, exponents, tapering, bending, exist), `gt_sq.ply` (mesh, one color per primitive).
Options: `--optimize 0` network only (default 1000 iterations), `--frame world`, `--rigid` (no tapering/bending, 'base' checkpoint).

### 5. Evaluate VGGT against the GT
Aligns VGGT's cloud to the GT using the cameras, then reports camera and point-cloud errors.
```bash
python -m recon.eval_vggt dataset/chair_test
```
→ prints alignment, camera errors (rotation°, center, focal) and accuracy / completeness / Chamfer / F-score; writes `vggt_aligned.ply` (VGGT in the GT frame and scale) and `eval_vggt.json`.
Options: `--align sim3` for full Umeyama (VGGT paper protocol), `--align none`.
Distances are in units of the object's radius (mesh scaled to the unit sphere).

### 6. Visualize
Interactive viewer for superquadrics + point clouds in the browser; starts at rendered camera 0.
```bash
python -m recon.view dataset/chair_test      # loads gt_sq.ply, gt.ply, vggt.ply if present
```
Open http://localhost:8080 on your laptop. VS Code Remote-SSH forwards the port automatically; otherwise:
```bash
ssh -p 58022 -L 8080:localhost:8080 hdo@atcremers66.in.tum.de
```
VGGT's cloud is rescaled to the GT with the camera-based scale (same as step 5).
Options: `--port 8091` if 8080 is taken, `--vggt-align none` to show VGGT at its raw scale.

## Decoder sanity check (single object, MSE)

Frozen VGGT tokens → SQ decoder (`l2sq/`, see `instructions/pipeline.JPG`). One object, 24 train views and 24 test views from different random cameras; GT superquadric *m* is tied to query *m*, so plain MSE is valid. Two kinds of pseudo-GT, same pipeline:

| | pseudo-GT | flag | outputs per query | run |
|---|---|---|---|---|
| rigid | `gt_sq_rigid.npz` (SuperFlex `base`) | — | 15: t 3, rot6D 6, scale 3, exponents 2, exist 1 | `runs/chair_L1_logscale` |
| deformable (branch `deformable`) | `gt_sq_deform.npz` (SuperFlex `superflex`) | `--deform` | 26: + tapering 2, bending amount 3, bending direction 3×2D | `runs/chair_L1_deform` |

### Data layout (per object, same names for every object)
```
dataset/<obj>/train/, test/        images/ depth/ masks/ cameras.json  (24 views each, different cameras)
                                   vggt.ply vggt_pred.npz vggt_aligned.ply eval_vggt.json  (VGGT on all 24 views)
                                   gt.ply gt.npz                       (GT points, this split's cam0)
dataset/<obj>/gt_world.{ply,npz}       GT points, world frame
dataset/<obj>/gt_sq_rigid.{ply,npz}    rigid pseudo-GT, world frame
dataset/<obj>/gt_sq_deform.{ply,npz}   deformable pseudo-GT (tapering + bending), world frame
runs/<obj>_<setting>/                  one training run
```

### 1. Data and pseudo-GT
```bash
OBJ=dataset/chair
MESH=$(ls ~/shapenet_partial_views/chair/*/mesh.npz | head -1)

# views + depth (depth is needed for VGGT's per-sample scale normalization)
python -m dataset.render --mesh "$MESH" --out $OBJ/train --n-views 24 --mode random --seed 0 --save-depth --save-mask
python -m dataset.render --mesh "$MESH" --out $OBJ/test  --n-views 24 --mode random --seed 1 --save-depth --save-mask

# GT points + pseudo-GT superquadrics in the world frame (needs module load cuda/12.1)
python -m dataset.sample_gt --cameras $OBJ/train/cameras.json --frame world --out $OBJ/gt_world.ply
gdown --folder 1yz4rwho0rsQGb5pEQcPIucZUjohOlfBN -O superflex/checkpoints/base     # rigid checkpoint, once
python -m recon.gt_superquadrics --rigid --gt $OBJ/gt_world.npz --cameras $OBJ/train/cameras.json --out $OBJ/gt_sq_rigid
python -m recon.gt_superquadrics         --gt $OBJ/gt_world.npz --cameras $OBJ/train/cameras.json --out $OBJ/gt_sq_deform

# reference: VGGT on all 24 views of each split (expect scale ratio ~1, Chamfer ~0.01)
for SPLIT in train test; do
  python -m recon.vggt_pointcloud --images $OBJ/$SPLIT/images --out $OBJ/$SPLIT/vggt.ply
  python -m dataset.sample_gt --cameras $OBJ/$SPLIT/cameras.json --out $OBJ/$SPLIT/gt.ply
  python -m recon.eval_vggt $OBJ/$SPLIT
done

python -m recon.view $OBJ          # world frame: both fits (toggle in "Layers") + GT points
python -m l2sq.sq $OBJ/gt_sq_deform.npz   # deformable only: our deformed surface vs SuperFlex's mesh, Chamfer ~1e-3 or below
```
`recon.eval_vggt` on a folder with `depth/` also prints the **scale ratio**: VGGT's fitted scale divided by *s* from the GT depth. ~1 means our targets use the same normalization VGGT was trained with.

### 2. Train
```bash
# rigid
python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_rigid.npz --out runs/chair_L1_logscale
# deformable
python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_deform.npz --deform --out runs/chair_L1_deform
```
Per step: one N ∈ [1, 8] drawn uniformly (the whole run, no schedule), then per sample N random views out of the 24 train views; the first is the reference. Targets = GT superquadrics moved into that camera's frame and divided by *s* = mean distance of the GT depth points of all N views to it (VGGT's normalization). Tapering and the bending amount (curvature × SuperFlex's minimum radius, in [0, 1]) are scale-free, so the frame change leaves them as they are.

Loss (all MSE over the GT slots, weights `--w-*`, default 1): `t`, `R` (3×3 matrix), `scale` on log(scale) (`--scale-loss linear` = old behaviour, used by `runs/chair_L1`), `eps`, `exist` (BCE over all P queries); with `--deform` also `taper`, `bend`, and `alpha` (direction as a unit 2D vector, weighted by the target bending amount; `--w-bend` sets both).

**Geometric loss (`--loss superflex`)**, SuperFlex's training loss (paper Sec. 4.1, App. B.2; code `superflex/loss/loss.py` IoULoss), in `l2sq/geom_loss.py`. No pseudo-GT parameters: only the GT geometry in the sample frame, so the decoder chooses its own decomposition and slot order:
- `iou`: −log soft IoU between the existence-weighted union of the predicted superquadrics (soft occupancy σ(−d/τ), τ = 1e-3) and the mesh occupancy at `--n-vol` volume points (uniform in the padded bounding cube, labels from the voxelized mesh, computed at startup)
- `sdf`: radial distance of `--n-surf` GT surface points (`gt_world.npz`) to the union, truncated (`--truncation` 0.05), points inside weighted by `--neg-slope` 0.1
- `overlap` (volume inside more than one primitive), `sparsity` (0.5-norm of primitive usage), `exist_sf` (BCE, target = primitive is closest for > 24 of 4096 surface points)

Weights default to SuperFlex's `configs/train.yaml`: `--w-iou 0.2 --w-sdf 0.2 --w-overlap 5 --w-sparsity 1.26 --w-exist-sf 0.02`. Distances are computed after normalizing each sample as SuperFlex normalizes its input (centered, object in a unit cube), so these hyperparameters keep their meaning. One deviation: SuperFlex weights `sdf` with a learned point-to-primitive assignment from its point encoder; our decoder sees images, not points, so `sdf` uses the distance to the union (soft minimum over primitives, as SuperFlex's test-time optimization does), and sparsity/existence use the same soft assignment. `--w-mse 1` adds the parameter MSE (needs fixed slots) as extra supervision.
```bash
python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_rigid.npz --loss superflex --out runs/chair_L1_sf
python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_deform.npz --deform --loss superflex --out runs/chair_L1_deform_sf
```
(`--gt-sq` is still read: for the parameter metrics and `--w-mse`.)

Every `--eval-every` steps: fixed samples for N ∈ {1, 2, 4, 8} on train and test views → rotation error, translation error, relative scale error, exponent error, Chamfer between predicted and **pseudo-GT** superquadric surfaces (×*s*, object-radius units), existence accuracy, number of predicted superquadrics; `--deform` adds `taper_err`, `bend_err`. Against the GT geometry (any loss): `iou` = occupancy IoU of the predicted union vs the mesh, `chamfer_gt` = Chamfer vs GT surface points (×*s*). With `--loss superflex` the slot order is free, so the per-slot parameter errors are not meaningful there: use `iou` and `chamfer_gt`.

→ `runs/<name>/`: `args.json`, `metrics.jsonl`, `eval_final.json`, `decoder.pt`, `examples/<split>_N<n>_<k>/{gt_sq,pred_sq}.ply` (random views), `tb/`.
Options: `--layers 3`, `--queries 16`, `--steps 3000`, `--batch 2`, `--n-min/--n-max`, `--vggt-layer 23`, `--w-t/--w-R/--w-scale/--w-eps/--w-exist/--w-taper/--w-bend` (MSE), `--loss`, `--w-iou/--w-sdf/--w-overlap/--w-sparsity/--w-exist-sf/--w-mse` (superflex), `--n-surf/--n-vol`, `--no-tb`. One run (3000 steps) takes ~2 h on the RTX A2000; each log line shows the peak GPU memory actually used (`nvidia-smi` shows PyTorch's cache, which is larger).

### 3. Predict and visualize
`l2sq.predict` uses views 0…N−1 of a split, so the reference is camera 0 and the prediction lines up with the split viewer. It reads the run's `args.json`, so a rigid run predicts rigid and a `--deform` run deformable superquadrics (`sample.json` → `"deform"`).
```bash
python -m l2sq.predict runs/chair_L1_deform $OBJ/test -n 1 8 24      # N > 8 = more views than seen in training
python -m recon.view $OBJ/test --mesh runs/chair_L1_deform/predict/test_N8_0/pred_sq.ply
```
→ `runs/<name>/predict/<split>_N<n>_0/{pred_sq.ply, gt_sq.ply, sample.json, metrics.json}`.

Layers in the split viewer: `pred_sq.ply` = decoder prediction, `gt_sq_rigid.ply` / `gt_sq_deform.ply` = the two pseudo-GT fits, `gt.ply` = GT points, `vggt.ply` = VGGT's cloud **from all 24 views of the split** (the reference from step 1, not what the decoder saw: the decoder reads VGGT's tokens of the N views and never builds a point cloud). Same colors per slot in prediction and fit, so toggle one off to compare.

### 4. Evaluate against the GT points
`recon.eval_sq`: everything vs `gt_world.npz` with the `eval_vggt` metric (accuracy, completeness, Chamfer, F-scores): both pseudo-GT fits (= ceilings for the decoder), VGGT's aligned clouds, and the run's exported target / predicted superquadrics (target rows should reproduce the fit the run was trained on).
```bash
python -m recon.eval_sq $OBJ --run runs/chair_L1_deform   # -> runs/chair_L1_deform/eval_sq.json (without --run: $OBJ/eval_sq.json)
```
Use this table, not the training log's `chamfer`, to compare rigid with deformable runs: the log measures against each run's own pseudo-GT (9 rigid vs 8 deformable superquadrics for the chair).

### Results so far (chair, final eval, Chamfer = training-log metric vs own pseudo-GT)

| run | scale loss | train N=8: rot / t / scale / Chamfer | test N=4: rot / t / scale / Chamfer |
|---|---|---|---|
| `chair_L1` (rigid) | linear | 2.8° / 0.033 / 39.1 % / 0.031 | 9.2° / 0.095 / 37.6 % / 0.068 |
| `chair_L1_logscale` (rigid) | log | 3.0° / 0.037 / 0.5 % / 0.025 | 10.1° / 0.089 / 1.1 % / 0.059 |
| `chair_L1_deform` | log | 3.0° / 0.036 / 0.5 % / 0.025 | 11.5° / 0.087 / 1.1 % / 0.064 |

Against the GT points (`eval_sq`): rigid pseudo-GT 0.0117, VGGT (24 views) 0.009–0.011, `chair_L1` predictions 0.04 (train views) / 0.05–0.10 (test views). The remaining error is mostly position and rotation; next: `--w-t 10`, `--layers 3`, more training views.

### TensorBoard
`runs/<name>/tb`: training losses, lr, s/step, peak GPU memory, and all eval metrics with train and test views overlaid (`eval_all/…` averaged over N, `eval_N<n>/…` per N). Total losses are not comparable across `--scale-loss` settings; compare the eval metrics.
```bash
tensorboard --logdir runs --port 6006     # on the server (localhost only), shows all runs; finished runs stay viewable
ssh -p 58022 -L 6006:localhost:6006 hdo@atcremers66.in.tum.de   # on your laptop, then open http://localhost:6006
```
(VS Code Remote-SSH forwards the port automatically.)

### Download results to your laptop
```bash
scp -P 58022 "hdo@atcremers66.in.tum.de:~/latent2SQ/dataset/chair_test/*.ply" ./dataset/chair_test/
```

---

## Coordinate frames

| Frame | Used by | Definition |
|---|---|---|
| `cam0` | `vggt.ply`, `gt.ply`, `gt_sq.*` (default) | rendered camera 0, OpenCV (x right, y down, z forward) — VGGT's world frame |
| `world` | renderer, SuperFlex input, `gt_world.*`, `gt_sq_rigid.*`, `gt_sq_deform.*` | mesh centered and scaled to the unit sphere, y-up (ShapeNet orientation) |

`w2c` in `cameras.json` maps world → camera (`x_cam = R x_world + t`), same as VGGT's predicted extrinsics.
VGGT's cloud is in `cam0` but at an arbitrary scale. VGGT fixes camera 0 at the identity, like our GT, so only the scale is unknown; `recon.eval_vggt` estimates it from the predicted vs GT camera centers (not from the points, whose coverage differs).
Superquadric pose (SuperFlex): `x = rotation @ x_local + translation`. Deformable ones are tapered, then bent around y, x, z in the local frame before that (see `l2sq/sq.py`).

## Notes

- Renders and results under `dataset/<name>/` are gitignored, as are `*.ply` and `*.npz`.
- `recon.gt_superquadrics` uses its own FPS (recent `torch_geometric.nn.fps` needs `pyg-lib`, which has no build for torch 2.3.1) and fixes the bending-curvature rescaling that SuperFlex's `update_handler(denormalize=True)` skips.
- The server account is shared: don't store personal GitHub or Claude logins there (the repo is pulled with a read/write deploy key).
