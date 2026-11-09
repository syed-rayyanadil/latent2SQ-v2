"""Interactive 3D viewer (viser) for superquadrics + point clouds, opened in your browser.

Run on the server from the repo root:

    python -m recon.view dataset/chair_test            # loads gt_sq.ply, gt.ply, vggt.ply if present
    python -m recon.view --mesh a.ply --pc b.ply c.ply  # or explicit files

then open http://localhost:8080 on your Mac. VS Code Remote-SSH forwards the port
automatically; from a plain terminal use:
    ssh -p 58022 -L 8080:localhost:8080 hdo@atcremers66.in.tum.de

Everything is shown in the frame it was saved in. For our files that is cam0 (OpenCV:
y down, z forward), and the initial view is rendered camera 0, i.e. images/000.png.
VGGT's cloud has an arbitrary scale; --vggt-scale auto rescales it about the camera
center to the GT's size (a visual aid only, not an alignment for metrics).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

DEFAULT_MESHES = ["gt_sq.ply"]
DEFAULT_PCS = ["gt.ply", "vggt.ply"]


def load_pc(path: Path, max_points: int, rng) -> tuple[np.ndarray, np.ndarray]:
    pc = trimesh.load(str(path), process=False)
    pts = np.asarray(pc.vertices, np.float32)
    cols = getattr(pc, "colors", None)
    cols = np.asarray(cols)[:, :3] if cols is not None and len(cols) == len(pts) else np.full((len(pts), 3), 180)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols.astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.view", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help=f"folder with {DEFAULT_MESHES + DEFAULT_PCS}")
    p.add_argument("--mesh", nargs="*", default=[], help="mesh files (e.g. superquadric .ply)")
    p.add_argument("--pc", nargs="*", default=[], help="point cloud files (.ply)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-points", type=int, default=50_000, help="subsample each cloud for the browser")
    p.add_argument("--point-size", type=float, default=0.01)
    p.add_argument("--vggt-scale", default="auto", help="'auto', 'none', or a number")
    a = p.parse_args(argv)

    meshes, pcs = [Path(m) for m in a.mesh], [Path(c) for c in a.pc]
    if a.folder:
        folder = Path(a.folder)
        meshes += [folder / f for f in DEFAULT_MESHES if (folder / f).exists()]
        pcs += [folder / f for f in DEFAULT_PCS if (folder / f).exists()]
    if not meshes and not pcs:
        p.error("nothing to show: give a folder or --mesh/--pc files")

    import viser  # imported late so --help works without it

    rng = np.random.default_rng(0)
    clouds = {path: load_pc(path, a.max_points, rng) for path in pcs}

    # Rough scale for VGGT: match median distance from the camera (origin) to the GT's.
    ref = next((c[0] for path, c in clouds.items() if path.stem.startswith("gt")), None)
    for path, (pts, cols) in clouds.items():
        if "vggt" not in path.stem or a.vggt_scale == "none":
            continue
        if a.vggt_scale == "auto":
            if ref is None:
                continue
            s = float(np.median(np.linalg.norm(ref, axis=1)) / np.median(np.linalg.norm(pts, axis=1)))
        else:
            s = float(a.vggt_scale)
        clouds[path] = (pts * s, cols)
        print(f"{path.name}: scaled x{s:.3f} about the camera center", file=sys.stderr)

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("-y")  # cam0 / OpenCV frame: y points down
    server.scene.add_frame("/cam0", axes_length=0.2, axes_radius=0.005)

    handles = {}
    for path in meshes:
        handles[path.name] = server.scene.add_mesh_trimesh(f"/mesh/{path.stem}", mesh=trimesh.load(str(path), process=False))
    for path, (pts, cols) in clouds.items():
        handles[path.name] = server.scene.add_point_cloud(f"/pc/{path.stem}", points=pts, colors=cols,
                                                          point_size=a.point_size)

    with server.gui.add_folder("Layers"):
        for name, handle in handles.items():
            cb = server.gui.add_checkbox(name, initial_value=True)

            @cb.on_update
            def _(event, handle=handle) -> None:
                handle.visible = event.target.value

    size = server.gui.add_slider("point size", min=0.001, max=0.05, step=0.001, initial_value=a.point_size)

    @size.on_update
    def _(event) -> None:
        for h in handles.values():
            if hasattr(h, "point_size"):
                h.point_size = event.target.value

    all_pts = np.concatenate([c[0] for c in clouds.values()]) if clouds else trimesh.load(str(meshes[0])).vertices
    center = tuple(float(v) for v in np.median(all_pts, axis=0))

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = (0.0, 0.0, 0.0)  # start at rendered camera 0
        client.camera.look_at = center
        client.camera.up_direction = (0.0, -1.0, 0.0)

    print(f"Viewer running: open http://localhost:{a.port}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
