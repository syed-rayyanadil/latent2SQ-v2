"""How close can superquadrics get to the GT surface? Everything vs the GT point cloud, one protocol.

Run from the repo root:

    python -m recon.eval_sq dataset/chair                      # pseudo-GT SQs + VGGT
    python -m recon.eval_sq dataset/chair --run runs/chair_L1  # + the decoder's exported examples

Rows, all compared with <obj>/gt_world.npz in the world frame with the same metric as
recon.eval_vggt (accuracy = pred->GT, completeness = GT->pred, Chamfer = their mean, F-scores):
    gt_sq_rigid      the pseudo-GT the decoder is trained on -> ceiling for the MSE decoder
    gt_sq            the deformable fit, if present          -> ceiling with tapering/bending
    vggt <split>     <split>/vggt_aligned.ply (run recon.eval_vggt first), moved cam0 -> world
    pred/target ...  with --run: <run>/examples/*/{pred_sq,gt_sq}.ply, moved from the sample's
                     reference camera to world (target rows should reproduce gt_sq_rigid)

Superquadric meshes are sampled uniformly by area. Surface parts that lie inside another
primitive count as accuracy errors, completeness does not see them.

Writes <run>/eval_sq.json with --run, else <obj>/eval_sq.json. Distances in object-radius units.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from recon.align import chamfer_metrics, read_ply_xyz_rgb

SQ_ROWS = ["gt_sq_rigid", "gt_sq"]


def mesh_points(path: Path, n: int, seed: int) -> np.ndarray:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float64)


def cam_to_world(pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (pts - t) @ R          # x_w = R^T (x_c - t)


def subsample(pts: np.ndarray, n: int, rng) -> np.ndarray:
    return pts[rng.choice(len(pts), n, replace=False)] if len(pts) > n else pts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.eval_sq", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("obj", help="object folder, e.g. dataset/chair")
    p.add_argument("--run", help="training run with examples/, e.g. runs/chair_L1")
    p.add_argument("-n", type=int, default=100_000, help="points per surface / cloud")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    obj = Path(a.obj)
    gt = np.load(obj / "gt_world.npz")
    if str(gt["frame"]) != "world":
        sys.exit(f"{obj / 'gt_world.npz'} is in frame {gt['frame']!s}, expected world")
    gt_pts = gt["points"].astype(np.float64)
    rng = np.random.default_rng(a.seed)
    rows: dict[str, dict] = {}

    for name in SQ_ROWS:
        if (obj / f"{name}.ply").exists():
            rows[name] = chamfer_metrics(mesh_points(obj / f"{name}.ply", a.n, a.seed), gt_pts)

    for split in ["train", "test"]:
        f = obj / split / "vggt_aligned.ply"
        if not f.exists():
            print(f"{f} missing (run recon.eval_vggt {obj / split}), skipped", file=sys.stderr)
            continue
        w2c0 = np.asarray(json.loads((obj / split / "cameras.json").read_text())["frames"][0]["w2c"])
        pts = cam_to_world(read_ply_xyz_rgb(f)[0].astype(np.float64), w2c0)
        rows[f"vggt {split}"] = chamfer_metrics(subsample(pts, a.n, rng), gt_pts)

    per_example = {}
    if a.run:
        groups = defaultdict(list)       # "pred test N8" -> [metrics, ...]
        cams_cache = {}
        for d in sorted((Path(a.run) / "examples").glob("*_N*_*")):
            info = json.loads((d / "sample.json").read_text())
            pool = Path(info["pool"])
            if pool not in cams_cache:
                cams_cache[pool] = json.loads((pool / "cameras.json").read_text())
            w2c = np.asarray(cams_cache[pool]["frames"][info["views"][0]]["w2c"])
            n_views = len(info["views"])
            for kind, fname in [("pred", "pred_sq.ply"), ("target", "gt_sq.ply")]:
                pts = cam_to_world(mesh_points(d / fname, a.n, a.seed), w2c)
                m = chamfer_metrics(pts, gt_pts)
                per_example[f"{d.name} {kind}"] = m
                groups[f"{kind} {info['split']} N{n_views}"].append(m)
        for key in sorted(groups, key=lambda k: (k.split()[0] != "target", k)):
            ms = groups[key]
            rows[f"{key} (x{len(ms)})"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    if not rows:
        sys.exit(f"nothing to evaluate in {obj}")
    w = max(len(k) for k in rows)
    print(f"\nvs {obj / 'gt_world.npz'} ({len(gt_pts)} pts, world frame, object-radius units)")
    print(f"{'':{w}}  {'acc':>7} {'comp':>7} {'chamfer':>8} {'F@0.01':>7} {'F@0.02':>7} {'F@0.05':>7}")
    for k, m in rows.items():
        print(f"{k:{w}}  {m['accuracy']:7.4f} {m['completeness']:7.4f} {m['chamfer']:8.4f} "
              f"{m['fscore@0.01']:7.3f} {m['fscore@0.02']:7.3f} {m['fscore@0.05']:7.3f}")

    out = Path(a.run) / "eval_sq.json" if a.run else obj / "eval_sq.json"
    out.write_text(json.dumps({"gt": str(obj / "gt_world.npz"), "rows": rows, "examples": per_example}, indent=2))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
