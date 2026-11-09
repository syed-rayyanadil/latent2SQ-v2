"""Render multi-view images of a mesh, headless on the GPU (EGL).

Run from the repo root:

    python -m dataset.render --mesh path/to/model.obj --out /storage/user/hdo/renders/chair_0001

Output layout:

    <out>/images/000.png ...   RGB renders (RGBA if --bg none)
    <out>/masks/000.png  ...   object masks        (--save-mask)
    <out>/depth/000.npy  ...   z-depth, 0 = empty  (--save-depth)
    <out>/cameras.json         intrinsics + extrinsics per view

Camera convention in cameras.json is OpenCV (x right, y down, z forward), and
"w2c" is the 3x4 camera-from-world matrix -- the same convention as VGGT's
predicted extrinsics, so they can be compared directly. The mesh is centered
and scaled to fit the unit sphere before rendering (world is Y-up, like ShapeNet).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before pyrender / OpenGL is imported. EGL = headless GPU rendering.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image

# OpenGL camera (x right, y up, looks along -z) -> OpenCV camera (x right, y down, looks along +z)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

BACKGROUNDS = {"white": (255, 255, 255), "black": (0, 0, 0), "none": None}


# --------------------------------------------------------------------------- mesh

def load_mesh(path: str | Path, use_texture: bool = True, smooth: bool = False):
    """Load a mesh file, normalize it to the unit sphere, return pyrender meshes + normalization."""
    scene = trimesh.load(str(path), force="scene", process=False)

    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        geom.apply_transform(transform)
        parts.append(geom)
    if not parts:
        raise ValueError(f"No triangle geometry found in {path}")

    # Center the bounding box at the origin and scale so the mesh fits in the unit sphere.
    verts = np.concatenate([p.vertices for p in parts])
    center = (verts.min(0) + verts.max(0)) / 2.0
    scale = 1.0 / np.linalg.norm(verts - center, axis=1).max()

    gray = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0], metallicFactor=0.0, roughnessFactor=0.8, doubleSided=True
    )
    meshes = []
    for p in parts:
        p.apply_translation(-center)
        p.apply_scale(scale)
        if use_texture:
            mesh = pyrender.Mesh.from_trimesh(p, smooth=smooth)
        else:
            p.visual = trimesh.visual.ColorVisuals()
            mesh = pyrender.Mesh.from_trimesh(p, material=gray, smooth=smooth)
        # ShapeNet often has flipped / single-sided faces: render both sides.
        for prim in mesh.primitives:
            if prim.material is not None:
                prim.material.doubleSided = True
        meshes.append(mesh)

    return meshes, {"center": center.tolist(), "scale": float(scale)}


# --------------------------------------------------------------------------- cameras

def look_at_gl(eye: np.ndarray, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """4x4 camera-to-world pose in OpenGL convention, camera at `eye` looking at `target`."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down: pick another up vector
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def sample_views(n: int, mode: str, elev: float, elev_range: tuple[float, float], seed: int):
    """Return a list of (azimuth_deg, elevation_deg)."""
    if mode == "orbit":
        az = np.linspace(0.0, 360.0, n, endpoint=False)
        return [(float(a), float(elev)) for a in az]
    if mode == "random":
        rng = np.random.default_rng(seed)
        az = rng.uniform(0.0, 360.0, n)
        # uniform over the sphere band: sample sin(elevation) uniformly
        lo, hi = np.sin(np.radians(elev_range[0])), np.sin(np.radians(elev_range[1]))
        el = np.degrees(np.arcsin(rng.uniform(lo, hi, n)))
        return [(float(a), float(e)) for a, e in zip(az, el)]
    raise ValueError(f"unknown mode {mode!r}")


def spherical_to_xyz(az_deg: float, el_deg: float, dist: float) -> np.ndarray:
    """Y-up world: elevation from the xz-plane, azimuth around y starting at +z."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


# --------------------------------------------------------------------------- render

def render_views(
    mesh_path: str | Path,
    out_dir: str | Path,
    n_views: int = 24,
    res: int = 518,
    fov: float = 50.0,
    mode: str = "orbit",
    elev: float = 20.0,
    elev_range: tuple[float, float] = (-10.0, 60.0),
    dist: float | None = None,
    seed: int = 0,
    bg: str = "white",
    use_texture: bool = True,
    smooth: bool = False,
    light_intensity: float = 3.0,
    save_depth: bool = False,
    save_mask: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    if save_depth:
        (out_dir / "depth").mkdir(exist_ok=True)
    if save_mask:
        (out_dir / "masks").mkdir(exist_ok=True)

    meshes, norm = load_mesh(mesh_path, use_texture=use_texture, smooth=smooth)

    # Pinhole intrinsics; default distance makes the unit sphere just fit in the view.
    f = 0.5 * res / np.tan(np.radians(fov) / 2.0)
    K = np.array([[f, 0.0, res / 2.0], [0.0, f, res / 2.0], [0.0, 0.0, 1.0]])
    if dist is None:
        dist = 1.05 / np.sin(np.radians(fov) / 2.0)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    for m in meshes:
        scene.add(m)
    cam_node = scene.add(
        pyrender.IntrinsicsCamera(fx=f, fy=f, cx=res / 2.0, cy=res / 2.0, znear=0.01, zfar=100.0)
    )
    # Headlight: a directional light that moves with the camera, so every view is lit.
    light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity))

    bg_rgb = BACKGROUNDS[bg]
    frames = []
    renderer = pyrender.OffscreenRenderer(res, res)
    try:
        for i, (az, el) in enumerate(sample_views(n_views, mode, elev, elev_range, seed)):
            c2w_gl = look_at_gl(spherical_to_xyz(az, el, dist))
            scene.set_pose(cam_node, c2w_gl)
            scene.set_pose(light_node, c2w_gl)

            rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            name = f"{i:03d}"

            if bg_rgb is None:
                Image.fromarray(rgba, "RGBA").save(out_dir / "images" / f"{name}.png")
            else:
                alpha = rgba[..., 3:4].astype(np.float32) / 255.0
                rgb = rgba[..., :3] * alpha + np.array(bg_rgb, np.float32) * (1.0 - alpha)
                Image.fromarray(rgb.round().astype(np.uint8), "RGB").save(out_dir / "images" / f"{name}.png")
            if save_mask:
                Image.fromarray(((depth > 0) * 255).astype(np.uint8), "L").save(out_dir / "masks" / f"{name}.png")
            if save_depth:
                np.save(out_dir / "depth" / f"{name}.npy", depth.astype(np.float32))

            c2w_cv = c2w_gl @ GL_TO_CV
            w2c_cv = np.linalg.inv(c2w_cv)
            frames.append({
                "file": f"images/{name}.png",
                "azimuth": az,
                "elevation": el,
                "w2c": w2c_cv[:3].tolist(),
                "c2w": c2w_cv.tolist(),
            })
    finally:
        renderer.delete()

    meta = {
        "mesh": str(Path(mesh_path).resolve()),
        "convention": "opencv (x right, y down, z forward); w2c = camera-from-world, like VGGT extrinsics",
        "world": "y-up; mesh centered and scaled to the unit sphere: x_world = (x_mesh - center) * scale",
        "normalization": norm,
        "width": res,
        "height": res,
        "fov_deg": fov,
        "distance": float(dist),
        "K": K.tolist(),
        "frames": frames,
    }
    (out_dir / "cameras.json").write_text(json.dumps(meta, indent=2))
    return out_dir


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dataset.render", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="mesh file (.obj, .glb, .ply, .off, ...)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--n-views", type=int, default=24)
    p.add_argument("--res", type=int, default=518, help="square image size (518 = VGGT input width)")
    p.add_argument("--fov", type=float, default=50.0, help="field of view in degrees")
    p.add_argument("--mode", choices=["orbit", "random"], default="orbit",
                   help="orbit: evenly spaced around at --elev; random: random azimuth/elevation in --elev-range")
    p.add_argument("--elev", type=float, default=20.0, help="elevation for orbit mode (degrees)")
    p.add_argument("--elev-range", type=float, nargs=2, default=(-10.0, 60.0), metavar=("MIN", "MAX"),
                   help="elevation range for random mode (degrees)")
    p.add_argument("--dist", type=float, default=None, help="camera distance (default: fit unit sphere in view)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bg", choices=list(BACKGROUNDS), default="white", help="background; 'none' saves RGBA")
    p.add_argument("--no-texture", action="store_true", help="ignore materials, render plain gray")
    p.add_argument("--smooth", action="store_true", help="smooth shading (default flat; flat is safer for ShapeNet)")
    p.add_argument("--light", type=float, default=3.0, help="headlight intensity")
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--save-mask", action="store_true")
    a = p.parse_args(argv)

    out = render_views(
        a.mesh, a.out, n_views=a.n_views, res=a.res, fov=a.fov, mode=a.mode, elev=a.elev,
        elev_range=tuple(a.elev_range), dist=a.dist, seed=a.seed, bg=a.bg, use_texture=not a.no_texture,
        smooth=a.smooth, light_intensity=a.light, save_depth=a.save_depth, save_mask=a.save_mask,
    )
    print(f"Rendered {a.n_views} views of {a.mesh} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
