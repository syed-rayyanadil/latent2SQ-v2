"""Interactive 3D viewer (viser) for superquadrics + point clouds, opened in your browser.

Run on the server from the repo root:

    python -m recon.view dataset/chair/train   # split folder: gt.ply, vggt.ply + ../gt_sq_{rigid,deform}.ply, in this split's cam0
    python -m recon.view dataset/chair         # object folder: gt_sq_{rigid,deform}.ply + gt_world.ply, world frame
    python -m recon.view --mesh a.ply --pc b.ply c.ply   # explicit files, shown as saved

then open the printed URL (http://localhost:<port>) on your Mac. VS Code Remote-SSH forwards
the port automatically; from a plain terminal use:
    ssh -p 58022 -L 8080:localhost:8080 hdo@atcremers66.in.tum.de

Frames (dataset layout dataset/<obj>/{train,test}/):
  * split folder (has cameras.json): everything in that split's camera-0 frame (OpenCV: y down,
    z forward); the initial view is rendered camera 0, i.e. images/000.png. The world-frame
    superquadrics from the object folder are moved in with camera 0's w2c.
  * object folder: world frame (mesh centered, unit sphere, y up).
VGGT's cloud has an arbitrary scale; by default it is rescaled to the GT with the scale
estimated from the cameras (same as recon.eval_vggt), see --vggt-align.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

SQ_NAMES = ["gt_sq_rigid.ply", "gt_sq_deform.ply", "gt_sq.ply"]   # world-frame SQs in the object folder (all shown)
SPLIT_PCS = ["gt.ply", "vggt.ply"]            # cam0-frame clouds in a split folder
OBJECT_PCS = ["gt_world.ply"]                 # world-frame cloud in the object folder


def load_pc(path: Path, max_points: int, rng) -> tuple[np.ndarray, np.ndarray]:
    pc = trimesh.load(str(path), process=False)
    pts = np.asarray(pc.vertices, np.float32)
    cols = getattr(pc, "colors", None)
    cols = np.asarray(cols)[:, :3] if cols is not None and len(cols) == len(pts) else np.full((len(pts), 3), 180)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols.astype(np.uint8)


def collect(folder: Path):
    """Files to show for a split or object folder: (meshes [(path, 4x4 or None)], clouds, frame)."""
    if (folder / "cameras.json").exists():  # split folder -> its camera-0 frame
        cams = json.loads((folder / "cameras.json").read_text())
        T = np.eye(4)
        T[:3] = np.asarray(cams["frames"][0]["w2c"])   # world -> cam0 (rigid, metric)
        meshes = [(folder.parent / n, T) for n in SQ_NAMES if (folder.parent / n).exists()]
        pcs = [folder / f for f in SPLIT_PCS if (folder / f).exists()]
        return meshes, pcs, "cam0"
    meshes = [(folder / n, None) for n in SQ_NAMES if (folder / n).exists()]
    pcs = [folder / f for f in OBJECT_PCS if (folder / f).exists()]
    return meshes, pcs, "world"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.view", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="split folder (dataset/<obj>/train) or object folder (dataset/<obj>)")
    p.add_argument("--mesh", nargs="*", default=[], help="extra mesh files, shown as saved")
    p.add_argument("--pc", nargs="*", default=[], help="extra point cloud files (.ply), shown as saved")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-points", type=int, default=50_000, help="subsample each cloud for the browser")
    p.add_argument("--point-size", type=float, default=0.01)
    p.add_argument("--vggt-align", default="scale",
                   help="how to place vggt.ply on the GT: 'scale' (from cameras, default), 'sim3' "
                        "(Umeyama from cameras), 'none' (raw), or a number (manual scale)")
    a = p.parse_args(argv)

    meshes = [(Path(m), None) for m in a.mesh]
    pcs = [Path(c) for c in a.pc]
    frame = "cam0"
    if a.folder:
        m, c, frame = collect(Path(a.folder))
        meshes += m
        pcs += c
    if not meshes and not pcs:
        p.error(f"nothing to show in {a.folder}: expected {SPLIT_PCS} (split folder) or "
                f"{SQ_NAMES[:2]} / {OBJECT_PCS} (object folder), or give --mesh/--pc files")

    import viser  # imported late so --help works without it

    rng = np.random.default_rng(0)
    clouds = {path: load_pc(path, a.max_points, rng) for path in pcs}

    # Place VGGT's cloud on the GT. Its scale is arbitrary; the alignment comes from the cameras
    # (see recon/align.py), which needs cameras.json + vggt_pred.npz next to vggt.ply.
    for path, (pts, cols) in clouds.items():
        if path.stem != "vggt" or a.vggt_align == "none":
            continue
        try:
            s, R, t = float(a.vggt_align), np.eye(3), np.zeros(3)
            how = "manual scale"
        except ValueError:
            from recon.align import align_from_cameras, load_case
            if not ((path.parent / "cameras.json").exists() and (path.parent / "vggt_pred.npz").exists()):
                print(f"{path.name}: no cameras.json / vggt_pred.npz next to it, shown unaligned", file=sys.stderr)
                continue
            cams, pred = load_case(path.parent)
            s, R, t = align_from_cameras(cams, pred["extrinsic"], a.vggt_align)
            how = f"{a.vggt_align} from cameras"
        clouds[path] = ((s * pts @ R.T + t).astype(np.float32), cols)
        print(f"{path.name}: aligned ({how}), scale x{s:.3f}", file=sys.stderr)

    server = viser.ViserServer(port=a.port)
    up = (0.0, -1.0, 0.0) if frame == "cam0" else (0.0, 1.0, 0.0)  # OpenCV camera: y down; world: y up
    server.scene.set_up_direction(up)
    server.scene.add_frame("/origin", axes_length=0.2, axes_radius=0.005)

    handles = {}
    for path, T in meshes:
        mesh = trimesh.load(str(path), process=False)
        if T is not None:
            mesh.apply_transform(T)
        handles[path.name] = server.scene.add_mesh_trimesh(f"/mesh/{path.stem}", mesh=mesh)
        print(f"{path}: {'moved into ' + frame if T is not None else 'as saved'}", file=sys.stderr)
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

    all_pts = (np.concatenate([c[0] for c in clouds.values()]) if clouds
               else np.asarray(trimesh.load(str(meshes[0][0])).vertices))
    center = np.median(all_pts, axis=0)
    # cam0: start at rendered camera 0; world: start 2.5 units in front of and slightly above the object
    start = np.zeros(3) if frame == "cam0" else center + np.array([0.0, 0.8, 2.5])

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = tuple(float(v) for v in start)
        client.camera.look_at = tuple(float(v) for v in center)
        client.camera.up_direction = up

    port = server.get_port()
    print(f"Viewer running ({frame} frame): open http://localhost:{port}  (Ctrl+C to stop)", file=sys.stderr)
    if port != a.port:
        print(f"  note: port {a.port} was busy (an old viewer still running?), using {port} -- forward that one",
              file=sys.stderr)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
