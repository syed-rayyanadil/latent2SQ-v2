"""Views of one object + its pseudo-GT superquadrics, turned into VGGT-frame training samples.

A sample = N views (the first is the reference) + targets in VGGT's frame for that view set:
    1. world -> camera of the first view (its GT w2c)
    2. divide by s = mean distance to that camera of the GT depth points of ALL N views
       (VGGT's training normalization, paper Sec. 3.4)
The depth points are precomputed once per view in the world frame (every `stride`-th valid
pixel, which keeps each view's weight proportional to its pixel count, as in VGGT).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from l2sq.sq import min_radius, to_view_frame
from recon.align import depth_points_world, vggt_normalization


@dataclass
class Sample:
    views: list[int]          # indices into the pool; views[0] = reference frame
    scale_s: float            # VGGT normalization of this view set
    R: np.ndarray             # (M,3,3) targets in the sample frame
    t: np.ndarray             # (M,3)
    scale: np.ndarray         # (M,3)
    eps: np.ndarray           # (M,2)
    taper: np.ndarray | None = None   # (M,2)   deformable only (scale-free, unchanged by the frame change)
    bend: np.ndarray | None = None    # (M,3)   bending amount kb * min_radius, axes (z, x, y)
    alpha: np.ndarray | None = None   # (M,3)   bending direction angles, axes (z, x, y)


class ViewPool:
    """All rendered views of one object (a dataset.render output folder with --save-depth)."""

    def __init__(self, folder: str | Path, stride: int = 4):
        from vggt.utils.load_fn import load_and_preprocess_images

        self.folder = Path(folder)
        cams = json.loads((self.folder / "cameras.json").read_text())
        self.K = np.asarray(cams["K"])
        self.w2c = np.stack([np.asarray(f["w2c"]) for f in cams["frames"]])     # (V,3,4)
        files = [str(self.folder / f["file"]) for f in cams["frames"]]
        self.images = load_and_preprocess_images(files)                         # (V,3,518,518) in [0,1]
        if self.images.shape[-2:] != (cams["height"], cams["width"]):
            raise ValueError(f"VGGT preprocessing changed the image size to {tuple(self.images.shape[-2:])}; "
                             "render square 518x518 views.")

        self.points_world = depth_points_world(self.folder, cams, stride)  # per view, world frame

    def __len__(self) -> int:
        return len(self.points_world)

    def normalization(self, views: list[int]) -> float:
        return vggt_normalization(self.points_world, self.w2c[views[0]], views)


class GTSuperquadrics:
    """Pseudo-GT superquadrics in the render WORLD frame (recon.gt_superquadrics --frame world).

    deform=False: rigid only (a fit with tapering/bending is rejected).
    deform=True:  also tapering and bending (SuperFlex bending (6,) -> bend amount + angle per axis).
    """

    def __init__(self, path: str | Path, deform: bool = False):
        d = np.load(path, allow_pickle=True)
        keep = (d["exist"][0] > 0.5).reshape(-1)
        self.R, self.t = d["rotation"][0][keep], d["translation"][0][keep]
        self.scale, self.eps = d["scale"][0][keep], d["exponents"][0][keep]
        taper, bending = d["tapering"][0][keep].reshape(-1, 2), d["bending"][0][keep].reshape(-1, 6)
        self.deform = deform
        self.taper = self.bend = self.alpha = None
        if deform:
            self.taper = taper.astype(np.float64)
            self.bend = np.clip(bending[:, 0::2] * min_radius(self.scale, self.taper), 0.0, 1.0)
            self.alpha = bending[:, 1::2].astype(np.float64)
        elif np.abs(taper).max(initial=0) > 1e-6 or np.abs(bending[:, 0::2]).max(initial=0) > 1e-6:
            raise ValueError(f"{path} has tapering/bending; train with --deform, or use a rigid fit "
                             "(recon.gt_superquadrics --rigid)")

    def __len__(self) -> int:
        return len(self.R)


def make_sample(pool: ViewPool, gt: GTSuperquadrics, views: list[int]) -> Sample:
    s = pool.normalization(views)
    R, t, scale = to_view_frame(gt.R, gt.t, gt.scale, pool.w2c[views[0]], s)
    if gt.deform:
        return Sample(views, s, R, t, scale, gt.eps.copy(), gt.taper.copy(), gt.bend.copy(), gt.alpha.copy())
    return Sample(views, s, R, t, scale, gt.eps.copy())


def random_views(rng: np.random.Generator, n_pool: int, n: int) -> list[int]:
    """n distinct views in random order; the first one becomes the reference frame."""
    return [int(v) for v in rng.permutation(n_pool)[:n]]


def collate(pool: ViewPool, samples: list[Sample], P: int, device) -> dict:
    """Stack a batch (all samples have the same N). GT slot m <-> query m (fixed assignment)."""
    B, M = len(samples), len(samples[0].R)
    if M > P:
        raise ValueError(f"{M} GT superquadrics but only P={P} queries")
    out = {
        "images": torch.stack([pool.images[s.views] for s in samples]).to(device),     # (B,N,3,H,W)
        "R": torch.zeros(B, P, 3, 3), "t": torch.zeros(B, P, 3),
        "scale": torch.ones(B, P, 3), "eps": torch.ones(B, P, 2),
        "exist": torch.zeros(B, P),
        "s": torch.tensor([s.scale_s for s in samples]),
    }
    deform = samples[0].taper is not None
    if deform:
        out.update(taper=torch.zeros(B, P, 2), bend=torch.zeros(B, P, 3), alpha=torch.zeros(B, P, 3))
    for b, s in enumerate(samples):
        out["R"][b, :M] = torch.as_tensor(s.R, dtype=torch.float32)
        out["t"][b, :M] = torch.as_tensor(s.t, dtype=torch.float32)
        out["scale"][b, :M] = torch.as_tensor(s.scale, dtype=torch.float32)
        out["eps"][b, :M] = torch.as_tensor(s.eps, dtype=torch.float32)
        out["exist"][b, :M] = 1.0
        if deform:
            for k in ("taper", "bend", "alpha"):
                out[k][b, :M] = torch.as_tensor(getattr(s, k), dtype=torch.float32)
    return {k: (v.to(device) if k != "images" else v) for k, v in out.items()}
