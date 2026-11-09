"""Minimal binary PLY writer (no extra dependencies)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None,
              normals: np.ndarray | None = None) -> None:
    """Binary little-endian PLY: xyz float32, optional normals float32, optional rgb uint8."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    vertex = np.empty(len(points), dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    if normals is not None:
        vertex["nx"], vertex["ny"], vertex["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if colors is not None:
        vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    ply_type = {"<f4": "float", "u1": "uchar"}
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
    header += "".join(f"property {ply_type[t]} {name}\n" for name, t in fields)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
