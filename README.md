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

Frozen VGGT tokens → SQ decoder (`l2sq/`, see `instructions/pipeline.JPG`). One object, 24 train views and 24 test views from different random cameras; GT superquadric *m* is tied to query *m*, so plain MSE is valid.

```bash
OBJ=dataset/chair
MESH=$(ls ~/shapenet_partial_views/chair/*/mesh.npz | head -1)

# views + depth (depth is needed for VGGT's per-sample scale normalization)
python -m dataset.render --mesh "$MESH" --out $OBJ/train --n-views 24 --mode random --seed 0 --save-depth --save-mask
python -m dataset.render --mesh "$MESH" --out $OBJ/test  --n-views 24 --mode random --seed 1 --save-depth --save-mask

# rigid pseudo-GT in the world frame (SuperFlex 'base' checkpoint, once)
gdown --folder 1yz4rwho0rsQGb5pEQcPIucZUjohOlfBN -O superflex/checkpoints/base
python -m dataset.sample_gt --cameras $OBJ/train/cameras.json --frame world --out $OBJ/gt_world.ply
python -m recon.gt_superquadrics --rigid --gt $OBJ/gt_world.npz --cameras $OBJ/train/cameras.json --out $OBJ/gt_sq_rigid

# sanity check: VGGT on all 24 views of each split (expect scale ratio ~1, Chamfer ~0.01)
for SPLIT in train test; do
  python -m recon.vggt_pointcloud --images $OBJ/$SPLIT/images --out $OBJ/$SPLIT/vggt.ply
  python -m dataset.sample_gt --cameras $OBJ/$SPLIT/cameras.json --out $OBJ/$SPLIT/gt.ply
  python -m recon.eval_vggt $OBJ/$SPLIT
done

# train (L = 1 decoder layer, P = 16 queries)
python -m l2sq.train --train $OBJ/train --test $OBJ/test --gt-sq $OBJ/gt_sq_rigid.npz --out runs/chair_L1
```
Per sample: N ∈ [1, 8] random train views, the first is the reference. Targets = GT superquadrics moved into that camera's frame and divided by *s* = mean distance of the GT depth points of all N views to it (VGGT's normalization). Every `--eval-every` steps it evaluates fixed samples for N ∈ {1, 2, 4, 8} on train and test views: rotation error, translation error, scale error, Chamfer between superquadric surfaces (×*s*, object-radius units), existence accuracy, number of predicted superquadrics.

→ `runs/chair_L1/`: `metrics.jsonl`, `eval_final.json`, `decoder.pt`, `examples/<split>_N<n>_<k>/{gt_sq,pred_sq}.ply`:
```bash
python -m recon.view --mesh runs/chair_L1/examples/test_N8_0/gt_sq.ply runs/chair_L1/examples/test_N8_0/pred_sq.ply
```
Options: `--layers 3`, `--queries 16`, `--steps 3000`, `--batch 2`, `--vggt-layer 23`, `--w-t/--w-R/--w-scale/--w-eps/--w-exist` loss weights, `--scale-loss log|linear` (default log: MSE on log(scale), i.e. relative size errors; `runs/chair_L1` used linear), `--no-tb`.

**What can we expect?** All against the GT point cloud (`gt_world.npz`), same metric as `eval_vggt`: the pseudo-GT superquadrics (the decoder's ceiling), VGGT's aligned clouds, and the run's predicted / target superquadrics:
```bash
python -m recon.eval_sq $OBJ --run runs/chair_L1     # -> runs/chair_L1/eval_sq.json (without --run: $OBJ/eval_sq.json)
```
Note the training log's `chamfer` is predicted vs **pseudo-GT superquadrics**, not vs the GT points.

`recon.eval_vggt` on a folder with `depth/` also prints the **scale ratio**: VGGT's fitted scale divided by *s* from the GT depth. ~1 means our targets use the same normalization VGGT was trained with.

**TensorBoard** (`runs/<name>/tb`): training losses, lr, s/step, and all eval metrics with train and test views overlaid (`eval_all/…` averaged over N, `eval_N<n>/…` per N):
```bash
tensorboard --logdir runs --port 6006     # on the server (localhost only)
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
| `world` | renderer, SuperFlex input | mesh centered and scaled to the unit sphere, y-up (ShapeNet orientation) |

`w2c` in `cameras.json` maps world → camera (`x_cam = R x_world + t`), same as VGGT's predicted extrinsics.
VGGT's cloud is in `cam0` but at an arbitrary scale. VGGT fixes camera 0 at the identity, like our GT, so only the scale is unknown; `recon.eval_vggt` estimates it from the predicted vs GT camera centers (not from the points, whose coverage differs).
Superquadric pose (SuperFlex): `x = rotation @ x_local + translation`.

## Notes

- Renders and results under `dataset/<name>/` are gitignored, as are `*.ply` and `*.npz`.
- `recon.gt_superquadrics` uses its own FPS (recent `torch_geometric.nn.fps` needs `pyg-lib`, which has no build for torch 2.3.1) and fixes the bending-curvature rescaling that SuperFlex's `update_handler(denormalize=True)` skips.
- The server account is shared: don't store personal GitHub or Claude logins there (the repo is pulled with a read/write deploy key).
