"""Fit ground-truth superquadrics to the GT point cloud with SuperFlex.

Run from the repo root:

    python -m recon.gt_superquadrics --gt dataset/chair_test/gt.npz \
        --cameras dataset/chair_test/cameras.json --out dataset/chair_test/gt_sq

Steps:
  1. take the GT points (gt.npz from dataset.sample_gt) back to the upright render world
     (y-up, like ShapeNet -- the orientation SuperFlex was trained on), FPS them to 4096 points
  2. SuperFlex feed-forward prediction
  3. (--optimize N, default 1000) refine the superquadrics against the mesh: occupancy IoU +
     surface SDF, as in SuperFlex's demo_optimization.py; the best iterate is kept
  4. move the superquadrics into the GT file's frame (cam0 = VGGT's frame) and save

Writes:
    <out>.npz   SuperFlex PredictionHandler format: scale, rotation, translation, exponents,
                exist, tapering, bending, assign_matrix, pc  (load with PredictionHandler.from_npz)
    <out>.ply   superquadric mesh for viewing (one color per primitive)

--rigid uses SuperFlex's 'base' checkpoint (no tapering/bending) and optimizes only the rigid
parameters; tapering/bending are then stored as zeros.

Superquadric pose convention (SuperFlex): x_frame = rotation @ x_local + translation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
# Put the submodule first on sys.path so `import superflex` finds the real package
# (superflex/superflex/) and not the submodule folder itself when run from the repo root.
sys.path.insert(0, str(REPO / "superflex"))

from dataset.sample_gt import load_trimesh  # noqa: E402

N_SURF = 4096       # SuperFlex input size
N_IOU = 100_000     # volume points for the optimization (fixed inside SuperQOptimization)


# --------------------------------------------------------------------------- frames

def frame_transform(cams: dict, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) mapping render-world points into `frame`: x_frame = R @ x_world + t."""
    if frame == "world":
        return np.eye(3), np.zeros(3)
    if frame == "cam0":
        w2c = np.asarray(cams["frames"][0]["w2c"])
        return w2c[:, :3], w2c[:, 3]
    raise ValueError(f"unsupported frame {frame!r}")


def world_mesh(cams: dict):
    """GT mesh in the render world (centered, unit sphere), same normalization as dataset.render."""
    mesh = load_trimesh(cams["mesh"])
    mesh.apply_translation(-np.asarray(cams["normalization"]["center"]))
    mesh.apply_scale(float(cams["normalization"]["scale"]))
    return mesh


# --------------------------------------------------------------------------- SuperFlex

def ensure_cuda_home() -> None:
    """SuperFlex JIT-loads CUDA kernels (PVCNN); torch needs CUDA_HOME even when they are cached."""
    import os
    import shutil

    if os.environ.get("CUDA_HOME"):
        return
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        sys.exit("CUDA_HOME is not set and nvcc is not on PATH. SuperFlex compiles CUDA kernels; run e.g.\n"
                 "    module load cuda/12.1\n"
                 "    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))\n"
                 "(or add both lines to ~/.bashrc)")
    os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parents[1])
    print(f"CUDA_HOME not set, using {os.environ['CUDA_HOME']} (from nvcc)", file=sys.stderr)


def load_model(ckpt_dir: Path, device: str):
    from omegaconf import OmegaConf

    ensure_cuda_home()
    from superflex.superflex import SuperFlex

    ckpt_path, cfg_path = ckpt_dir / "ckpt.pt", ckpt_dir / "config.yaml"
    if not ckpt_path.is_file():
        sys.exit(f"Checkpoint not found: {ckpt_path}\n"
                 "Download it (see superflex/scripts/download_checkpoints.sh).")
    cfg = OmegaConf.load(cfg_path)
    model = SuperFlex(cfg.superflex).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


@torch.no_grad()
def fps(points: np.ndarray, n: int, device: str, seed: int = 0) -> np.ndarray:
    """Farthest point sampling to exactly n points (what SuperFlex uses on ShapeNet).

    Plain PyTorch on purpose: recent torch_geometric.nn.fps needs pyg-lib, which has
    no build for our torch 2.3.1.
    """
    if len(points) <= n:
        idx = np.random.choice(len(points), n, replace=len(points) < n)
        return points[idx]
    x = torch.from_numpy(points).float().to(device)
    idx = torch.empty(n, dtype=torch.long, device=device)
    min_dist = torch.full((len(x),), float("inf"), device=device)
    farthest = torch.tensor(np.random.default_rng(seed).integers(len(x)), device=device)
    for i in range(n):
        idx[i] = farthest
        min_dist = torch.minimum(min_dist, ((x - x[farthest]) ** 2).sum(1))
        farthest = torch.argmax(min_dist)
    return points[idx.cpu().numpy()]


@torch.no_grad()
def predict(model, points: np.ndarray, device: str, name: str):
    from superflex.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
    from superflex.utils.predictions_handler import PredictionHandler

    pts_n, trans, scale = normalize_points(points)
    x = torch.from_numpy(pts_n).unsqueeze(0).float().to(device)
    out = model(x)
    out = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    trans, scale = np.array([trans]), np.array([scale])
    out = denormalize_outdict(out, trans, scale, False)
    pc = denormalize_points(x.cpu(), trans, scale, False)
    return PredictionHandler.from_outdict(out, pc, [name])


def occupancy_samples(mesh, n: int = N_IOU, seed: int = 0):
    """Uniform volume points in the padded bbox + inside/outside labels (as in SuperFlex's demo)."""
    rng = np.random.default_rng(seed)
    center, extent = mesh.bounding_box.centroid, float(np.max(mesh.extents))
    half = extent * 1.1 / 2
    pts = rng.uniform(center - half, center + half, (n, 3))
    occ = mesh.voxelized(pitch=extent * 1.1 / 100).fill().is_filled(pts)
    return pts.astype(np.float32), np.asarray(occ, bool)


def optimize(handler, surf: np.ndarray, pts_iou: np.ndarray, occ: np.ndarray, iters: int, device: str,
             deform: bool = True):
    from omegaconf import OmegaConf
    from superflex.optimization import SuperQOptimization

    ext = {
        "points_iou": [torch.tensor(pts_iou, dtype=torch.float32, device=device)],
        "occupancies": [torch.tensor(occ, dtype=torch.bool, device=device)],
        "points": [torch.tensor(surf, dtype=torch.float32, device=device)],
    }
    sq = SuperQOptimization(pred_handler=handler, indices=[0], external_data=ext, device=device,
                            cfg=OmegaConf.create({"tapering": deform, "bending": deform}))
    opt = torch.optim.Adam(sq.get_param_groups())
    names = ["raw_scale", "raw_exponents", "raw_rotation", "raw_tapering", "raw_bending", "translation"]

    best, best_iou, best_loss = None, float("nan"), float("inf")
    for it in range(iters):
        opt.zero_grad()
        loss, losses = sq.compute_losses(sq.forward())
        loss = loss[0]
        if torch.isnan(loss):
            print(f"NaN loss at iteration {it}, keeping best so far", file=sys.stderr)
            break
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_iou = loss.item(), losses["iou"][0].item()
            best = {k: getattr(sq, k).detach().clone() for k in names}
        if it % 100 == 0:
            print(f"  iter {it:4d}  loss {loss.item():.4f}  IoU {losses['iou'][0].item():.3f}", file=sys.stderr)

    if best is not None:
        with torch.no_grad():
            for k in names:
                getattr(sq, k).copy_(best[k])

    # Write back in SuperQOptimization's normalized frame, then undo its normalization here.
    # (Upstream update_handler(denormalize=True) rescales size/position but not the bending
    # curvature, which has units of 1/length.)
    sq.update_handler(denormalize=False, compute_meshes=False)
    ns = float(sq.normalization_scale[0])
    nt = sq.normalization_translation[0].detach().cpu().numpy()
    m = (handler.exist[0] > 0.5).reshape(-1)
    handler.scale[0][m] = handler.scale[0][m] * ns
    handler.translation[0][m] = handler.translation[0][m] * ns + nt
    handler.bending[0][m, 0::2] = handler.bending[0][m, 0::2] / ns
    return handler, best_iou


def transform_handler(handler, R: np.ndarray, t: np.ndarray):
    """Rigidly move all superquadrics (and the input cloud) by x -> R x + t."""
    handler.rotation[0] = np.einsum("ij,njk->nik", R, handler.rotation[0])
    handler.translation[0] = handler.translation[0] @ R.T + t
    handler.pc[0] = handler.pc[0] @ R.T + t
    return handler


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.gt_superquadrics", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="gt.npz from dataset.sample_gt")
    p.add_argument("--cameras", required=True, help="cameras.json from dataset.render (mesh + frames)")
    p.add_argument("--out", required=True, help="output path without extension (writes .npz and .ply)")
    p.add_argument("--rigid", action="store_true",
                   help="rigid superquadrics only (no tapering/bending): 'base' checkpoint + rigid optimization")
    p.add_argument("--ckpt", default=None,
                   help="SuperFlex checkpoint folder with ckpt.pt + config.yaml "
                        "(default: superflex/checkpoints/base with --rigid, else .../superflex)")
    p.add_argument("--optimize", type=int, default=1000, help="optimization iterations (0 = network only)")
    p.add_argument("--frame", choices=["same", "world", "cam0"], default="same",
                   help="output frame; 'same' = the GT file's frame")
    p.add_argument("--mesh-res", type=int, default=30, help="superquadric mesh resolution for the .ply")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")
    device = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cams = json.loads(Path(a.cameras).read_text())
    gt = np.load(a.gt)
    gt_frame = str(gt["frame"])
    if gt_frame not in ("world", "cam0"):
        sys.exit(f"{a.gt} is in frame {gt_frame!r}; re-run dataset.sample_gt with --cameras.")

    # 1. GT points -> upright world frame, 4096 FPS points
    R, t = frame_transform(cams, gt_frame)
    world_pts = (gt["points"].astype(np.float64) - t) @ R  # inverse of x = R w + t
    surf = fps(world_pts, N_SURF, device, a.seed)

    # 2. network prediction
    ckpt = Path(a.ckpt) if a.ckpt else REPO / "superflex" / "checkpoints" / ("base" if a.rigid else "superflex")
    model = load_model(ckpt, device)
    name = Path(a.out).stem
    handler = predict(model, surf, device, name)
    n_prim = int((handler.exist[0] > 0.5).sum())
    print(f"SuperFlex predicted {n_prim} superquadrics", file=sys.stderr)

    # 3. optimization against the mesh
    if a.optimize > 0:
        pts_iou, occ = occupancy_samples(world_mesh(cams), seed=a.seed)
        handler, iou = optimize(handler, surf, pts_iou, occ, a.optimize, device, deform=not a.rigid)
        print(f"Optimized {a.optimize} iterations, best IoU {iou:.3f}", file=sys.stderr)

    # 4. into the output frame and save
    out_frame = gt_frame if a.frame == "same" else a.frame
    handler = transform_handler(handler, *frame_transform(cams, out_frame))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    handler.save_npz(str(out.with_suffix(".npz")))
    handler.get_mesh(0, resolution=a.mesh_res).export(str(out.with_suffix(".ply")))
    print(f"Saved superquadrics (frame: {out_frame}) -> {out.with_suffix('.npz')}, {out.with_suffix('.ply')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
