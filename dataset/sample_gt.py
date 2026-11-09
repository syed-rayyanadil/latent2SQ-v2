"""Sample a ground-truth point cloud from a mesh surface (uniform by area).

Run from the repo root:

    # GT in the same frame as VGGT's output (first rendered camera), using the render's cameras.json:
    python -m dataset.sample_gt --cameras dataset/chair_test/cameras.json --out dataset/chair_test/gt.ply

    # or just the raw mesh coordinates:
    python -m dataset.sample_gt --mesh path/to/mesh.npz --out gt.ply

--frame (needs --cameras):
    world  normalized render world: x = (x_mesh - center) * scale   (y-up, unit sphere)
    cam0   camera 0 of the render, OpenCV convention -- the frame VGGT predicts in.
           VGGT's cloud then differs from this GT only by a global scale (plus prediction error).

Writes <out>.ply (xyz + normals) and <out>.npz (points, normals, frame).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from dataset.ply import write_ply


def load_trimesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file (or shapenet_partial_views mesh.npz) as a single Trimesh."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
    loaded = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in loaded.graph.nodes_geometry:
        transform, name = loaded.graph[node]
        geom = loaded.geometry[name]
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
            geom = geom.copy()
            geom.apply_transform(transform)
            parts.append(geom)
    if not parts:
        raise ValueError(f"No triangle geometry found in {path}")
    return trimesh.util.concatenate(parts)


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, np.float64), np.asarray(mesh.face_normals[face_idx], np.float64)


def to_frame(points: np.ndarray, normals: np.ndarray, cams: dict, frame: str):
    """Apply the render's normalization (and optionally camera 0's w2c) to points and normals."""
    center = np.asarray(cams["normalization"]["center"])
    scale = float(cams["normalization"]["scale"])
    points = (points - center) * scale  # uniform scale: normals unchanged
    if frame == "cam0":
        w2c = np.asarray(cams["frames"][0]["w2c"])  # (3, 4), OpenCV camera-from-world
        R, t = w2c[:, :3], w2c[:, 3]
        points = points @ R.T + t
        normals = normals @ R.T
    return points, normals


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dataset.sample_gt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", help="mesh file; default: the mesh recorded in --cameras")
    p.add_argument("--cameras", help="cameras.json written by dataset.render")
    p.add_argument("--frame", choices=["world", "cam0"], default="cam0",
                   help="output frame when --cameras is given (default: cam0 = VGGT's frame)")
    p.add_argument("--out", required=True, help="output .ply path")
    p.add_argument("-n", "--n-points", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    cams = json.loads(Path(a.cameras).read_text()) if a.cameras else None
    mesh_path = a.mesh or (cams and cams["mesh"])
    if not mesh_path:
        p.error("give --mesh or --cameras")

    mesh = load_trimesh(mesh_path)
    points, normals = sample_surface(mesh, a.n_points, a.seed)
    frame = "mesh"
    if cams is not None:
        points, normals = to_frame(points, normals, cams, a.frame)
        frame = a.frame

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gray = np.full((len(points), 3), 180, np.uint8)
    write_ply(out.with_suffix(".ply"), points, colors=gray, normals=normals)
    np.savez_compressed(out.with_suffix(".npz"), points=points.astype(np.float32),
                        normals=normals.astype(np.float32), frame=frame)
    print(f"Sampled {len(points)} points from {mesh_path} (frame: {frame}) -> {out.with_suffix('.ply')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
