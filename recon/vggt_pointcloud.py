"""Run VGGT on a folder of images and save the fused global point cloud as .ply.

Run from the repo root:

    python -m recon.vggt_pointcloud --images data/chair_test/images --out data/chair_test/vggt.ply

Writes:
    <out>.ply         colored point cloud (binary PLY, opens in MeshLab / CloudCompare / Open3D)
    <out>_pred.npz    extrinsic (S,3,4), intrinsic (S,3,3), depth (S,H,W), depth_conf (S,H,W), image_files

Notes:
  * VGGT's world frame is the FIRST camera's frame (OpenCV convention), and the scale is
    arbitrary -- the cloud is not aligned with the GT mesh until you align it (e.g. Umeyama/ICP).
  * --source depth (default) unprojects the depth head with the predicted cameras, which the
    VGGT authors report as more accurate than the direct point-map head (--source pointmap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder: str | Path) -> list[str]:
    files = sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY with xyz float32 + rgb uint8."""
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


@torch.no_grad()
def run_vggt(image_files: list[str], device: str = "cuda") -> dict:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # bf16 on Ampere+ (your RTX A2000 supports it), fp16 otherwise
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()  # first run downloads ~5 GB to $HF_HOME
    images = load_and_preprocess_images(image_files).to(device)  # (S, 3, H, W) in [0, 1]

    with torch.autocast("cuda", dtype=dtype):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])

    out = {
        "images": images.float().cpu().numpy(),                     # (S, 3, H, W)
        "extrinsic": extrinsic[0].float().cpu().numpy(),            # (S, 3, 4) camera-from-world
        "intrinsic": intrinsic[0].float().cpu().numpy(),            # (S, 3, 3)
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),    # (S, H, W)
        "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),  # (S, H, W)
        "world_points": pred["world_points"][0].float().cpu().numpy(),            # (S, H, W, 3)
        "world_points_conf": pred["world_points_conf"][0].float().cpu().numpy(),  # (S, H, W)
    }
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", file=sys.stderr)
    return out


def build_pointcloud(pred: dict, source: str = "depth", conf_percentile: float = 50.0,
                     filter_white_bg: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if source == "depth":
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        points = unproject_depth_map_to_point_map(pred["depth"][..., None], pred["extrinsic"], pred["intrinsic"])
        conf = pred["depth_conf"]
    else:
        points, conf = pred["world_points"], pred["world_points_conf"]

    colors = np.transpose(pred["images"], (0, 2, 3, 1))  # (S, H, W, 3) in [0, 1]

    keep = conf > max(np.percentile(conf, conf_percentile), 1e-5)
    if filter_white_bg:  # rendered background is pure white; VGGT still predicts points there
        keep &= ~(colors > 0.97).all(-1)

    return points[keep].astype(np.float32), (colors[keep] * 255).round().astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m recon.vggt_pointcloud", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder with the input images (sorted by name)")
    p.add_argument("--out", required=True, help="output .ply path")
    p.add_argument("--source", choices=["depth", "pointmap"], default="depth",
                   help="depth: depth head + cameras (default); pointmap: direct point-map head")
    p.add_argument("--conf-percentile", type=float, default=50.0,
                   help="drop the lowest X%% confidence points (0 keeps all)")
    p.add_argument("--keep-white-bg", action="store_true", help="do not remove white background pixels")
    a = p.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = list_images(a.images)
    print(f"Running VGGT on {len(files)} images from {a.images}", file=sys.stderr)
    pred = run_vggt(files)

    points, colors = build_pointcloud(pred, a.source, a.conf_percentile, not a.keep_white_bg)
    write_ply(out.with_suffix(".ply"), points, colors)
    np.savez_compressed(
        out.with_name(out.stem + "_pred.npz"),
        extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"],
        depth=pred["depth"], depth_conf=pred["depth_conf"], image_files=np.array(files),
    )
    print(f"Saved {len(points)} points -> {out.with_suffix('.ply')}", file=sys.stderr)


if __name__ == "__main__":
    main()
