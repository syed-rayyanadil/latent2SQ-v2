"""Evaluate VGGT's reconstruction against the GT: align, then camera + point cloud metrics.

Run from the repo root (needs cameras.json, vggt_pred.npz, vggt.ply, gt.npz in the folder):

    python -m recon.eval_vggt dataset/chair_test

Alignment is estimated from the cameras, not the points (see recon/align.py): by default only
the scale (VGGT and our GT both put camera 0 at the origin), --align sim3 for full Umeyama. Writes:
    <folder>/vggt_aligned.ply   VGGT cloud in the GT cam0 frame, metric scale
    <folder>/eval_vggt.json     all numbers below

If <folder>/depth/ exists (dataset.render --save-depth), it also compares the fitted scale with
VGGT's normalization computed from the GT depth (the same s the decoder's targets use).

Distances are in the render world's units (mesh scaled to the unit sphere), so a Chamfer of
0.015 means 1.5 % of the object's radius.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dataset.ply import write_ply
from recon.align import (ALIGN_MODES, align_from_cameras, apply_sim3, camera_errors, chamfer_metrics,
                         depth_points_world, load_case, read_ply_xyz_rgb, rotation_angle_deg,
                         vggt_normalization)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.eval_vggt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="dataset folder with cameras.json, vggt_pred.npz, vggt.ply, gt.npz")
    p.add_argument("--align", choices=ALIGN_MODES, default="scale")
    p.add_argument("--max-points", type=int, default=300_000, help="subsample the VGGT cloud for speed")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    folder = Path(a.folder)
    cams, pred = load_case(folder)
    gt = np.load(folder / "gt.npz")
    if str(gt["frame"]) != "cam0":
        sys.exit(f"gt.npz is in frame {gt['frame']!s}; re-run dataset.sample_gt with --frame cam0.")
    gt_pts = gt["points"].astype(np.float64)
    vggt_pts, vggt_cols = read_ply_xyz_rgb(folder / "vggt.ply")

    s, R, t = align_from_cameras(cams, pred["extrinsic"], a.align)
    aligned = apply_sim3(vggt_pts, s, R, t)

    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(aligned), min(a.max_points, len(aligned)), replace=False)
    result = {
        "folder": str(folder), "align": a.align,
        "sim3": {"scale": s, "rotation_deg": rotation_angle_deg(R), "translation": t.tolist()},
        "n_points": {"vggt": int(len(aligned)), "vggt_used": int(len(sub)), "gt": int(len(gt_pts))},
        "cameras": camera_errors(cams, pred, (s, R, t)),
        "points": chamfer_metrics(aligned[sub], gt_pts),
    }

    # VGGT is trained to output scenes normalized by s = mean distance of all views' GT depth points
    # to camera 0. If our depth/camera conventions are right, the fitted scale should be close to s.
    if (folder / "depth").is_dir():
        s_depth = vggt_normalization(depth_points_world(folder, cams), np.asarray(cams["frames"][0]["w2c"]))
        result["normalization"] = {"s_from_gt_depth": s_depth, "fitted_scale": s, "ratio": s / s_depth}

    write_ply(folder / "vggt_aligned.ply", aligned.astype(np.float32), colors=vggt_cols)
    (folder / "eval_vggt.json").write_text(json.dumps(result, indent=2))

    c, m = result["cameras"], result["points"]
    print(f"alignment ({a.align}): scale {s:.4f}, rotation {rotation_angle_deg(R):.2f} deg, "
          f"translation {np.round(t, 4).tolist()}")
    print(f"cameras: rotation err {c['rot_err_deg_mean']:.2f} deg (max {c['rot_err_deg_max']:.2f}), "
          f"center err {c['center_err_mean']:.4f} (max {c['center_err_max']:.4f}), "
          f"focal {c['focal_pred_px_mean']:.1f} vs {c['focal_gt_px']:.1f} px ({100 * c['focal_rel_err']:.1f} %)")
    print(f"points:  accuracy {m['accuracy']:.4f}, completeness {m['completeness']:.4f}, "
          f"chamfer {m['chamfer']:.4f}, F@0.02 {m['fscore@0.02']:.3f}, F@0.05 {m['fscore@0.05']:.3f}")
    if "normalization" in result:
        r = result["normalization"]
        print(f"scale:   fitted from cameras {r['fitted_scale']:.4f} vs s from GT depth {r['s_from_gt_depth']:.4f} "
              f"(ratio {r['ratio']:.3f}; ~1 means VGGT's normalization = our training targets' normalization)")
    print(f"wrote {folder / 'vggt_aligned.ply'}, {folder / 'eval_vggt.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
