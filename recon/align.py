"""Align VGGT predictions to the ground truth and measure them.

Both live in the frame of camera 0 (OpenCV), but VGGT's scale is arbitrary (it is trained to
output scenes whose points are ~1 unit from camera 0 on average). We recover the similarity
transform from the CAMERAS: Umeyama on the 12 predicted vs. GT camera centers. This needs no
point correspondences and is unbiased by how differently the two clouds cover the surface,
unlike matching point distances.

Conventions: extrinsics are w2c 3x4 (x_cam = R x + t); a similarity (s, R, t) maps
x -> s * R @ x + t.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALIGN_MODES = ("scale", "sim3", "none")


# --------------------------------------------------------------------------- geometry

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity (s, R, t) with dst ~ s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:  # avoid reflections
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / xs.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * points @ R.T + t


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# --------------------------------------------------------------------------- cameras

def gt_cameras_cam0(cams: dict) -> tuple[np.ndarray, np.ndarray]:
    """GT camera centers (S,3) and camera-to-world rotations (S,3,3), expressed in camera 0's frame."""
    w2c0 = np.asarray(cams["frames"][0]["w2c"])
    R0, t0 = w2c0[:, :3], w2c0[:, 3]
    c2w = np.stack([np.asarray(f["c2w"]) for f in cams["frames"]])  # world frame
    centers = c2w[:, :3, 3] @ R0.T + t0
    rots = np.einsum("ij,njk->nik", R0, c2w[:, :3, :3])
    return centers, rots


def pred_cameras(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted camera centers (S,3) and camera-to-world rotations (S,3,3) from w2c (S,3,4)."""
    Rw2c, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    Rc2w = np.transpose(Rw2c, (0, 2, 1))
    return -np.einsum("nij,nj->ni", Rc2w, t), Rc2w


def align_from_cameras(cams: dict, extrinsic: np.ndarray, mode: str = "sim3"):
    """Similarity mapping VGGT's frame onto the GT cam0 frame, estimated from camera centers.

    scale : scale only (default). VGGT fixes camera 0 to the identity, exactly like our GT
            frame, so origin and orientation already agree and only the scale is unknown.
            Least-squares s = <c_pred, c_gt> / <c_pred, c_pred> over camera centers.
    sim3  : Umeyama scale + rotation + translation (the protocol in the VGGT paper, which
            aligns point clouds; here the extra DOF mostly fit camera-center noise)
    none  : identity
    """
    if mode == "none":
        return 1.0, np.eye(3), np.zeros(3)
    c_gt, _ = gt_cameras_cam0(cams)
    c_pr, _ = pred_cameras(extrinsic)
    if len(c_gt) != len(c_pr):
        raise ValueError(f"{len(c_pr)} predicted vs {len(c_gt)} GT cameras -- were all views passed to VGGT?")
    if mode == "scale":
        return float((c_pr * c_gt).sum() / (c_pr * c_pr).sum()), np.eye(3), np.zeros(3)
    if mode == "sim3":
        return umeyama(c_pr, c_gt)
    raise ValueError(f"unknown align mode {mode!r}")


def camera_errors(cams: dict, pred: dict, sim3) -> dict:
    """Pose and focal errors of the predicted cameras after alignment."""
    s, R, t = sim3
    c_gt, r_gt = gt_cameras_cam0(cams)
    c_pr, r_pr = pred_cameras(pred["extrinsic"])
    c_pr = apply_sim3(c_pr, s, R, t)
    r_pr = np.einsum("ij,njk->nik", R, r_pr)
    rot_err = np.array([rotation_angle_deg(a.T @ b) for a, b in zip(r_gt, r_pr)])
    center_err = np.linalg.norm(c_pr - c_gt, axis=1)
    f_gt = float(cams["K"][0][0])
    f_pr = pred["intrinsic"][:, 0, 0]
    return {
        "rot_err_deg_mean": float(rot_err.mean()), "rot_err_deg_max": float(rot_err.max()),
        "center_err_mean": float(center_err.mean()), "center_err_max": float(center_err.max()),
        "cam_distance_mean": float(np.linalg.norm(c_gt - c_gt.mean(0), axis=1).mean()),
        "focal_gt_px": f_gt, "focal_pred_px_mean": float(f_pr.mean()),
        "focal_rel_err": float(abs(f_pr.mean() - f_gt) / f_gt),
    }


# --------------------------------------------------------------------------- point clouds

def chamfer_metrics(pred: np.ndarray, gt: np.ndarray, taus=(0.01, 0.02, 0.05)) -> dict:
    """Accuracy (pred->gt), completeness (gt->pred), Chamfer (their mean) and F-scores, as in VGGT/DTU."""
    from scipy.spatial import cKDTree

    d_pred = cKDTree(gt).query(pred, workers=-1)[0]   # accuracy distances
    d_gt = cKDTree(pred).query(gt, workers=-1)[0]     # completeness distances
    out = {
        "accuracy": float(d_pred.mean()), "completeness": float(d_gt.mean()),
        "chamfer": float((d_pred.mean() + d_gt.mean()) / 2),
        "accuracy_median": float(np.median(d_pred)), "completeness_median": float(np.median(d_gt)),
    }
    for tau in taus:
        p, r = float((d_pred < tau).mean()), float((d_gt < tau).mean())
        out[f"fscore@{tau}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out


def read_ply_xyz_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read points (and colors if present) from a PLY via trimesh."""
    import trimesh

    pc = trimesh.load(str(path), process=False)
    pts = np.asarray(pc.vertices, np.float64)
    cols = getattr(pc, "colors", None)
    cols = np.asarray(cols)[:, :3].astype(np.uint8) if cols is not None and len(cols) == len(pts) else None
    return pts, cols


def load_case(folder: str | Path) -> tuple[dict, dict]:
    """cameras.json + vggt_pred.npz from a dataset folder."""
    folder = Path(folder)
    cams = json.loads((folder / "cameras.json").read_text())
    pred = dict(np.load(folder / "vggt_pred.npz"))
    return cams, pred
