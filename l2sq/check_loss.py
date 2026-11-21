"""Check the SuperFlex loss without the network: optimize superquadric PARAMETERS directly.

Run from the repo root (GPU):

    python -m l2sq.check_loss --train dataset/chair/train --gt-sq dataset/chair/gt_sq_rigid.npz \
        --init runs/chair_L1_logscale/decoder.pt

One sample frame (train views 0..N-1, reference view 0), three starting points, each optimized
with Adam on the raw parameters (like SuperFlex's test-time optimization) for --iters steps:
    gt         the pseudo-GT itself      -> IoU must stay ~0.94 (it was fitted with this loss)
    gt+noise   perturbed pseudo-GT       -> IoU must recover
    decoder    the --init decoder's prediction (MSE run) -> IoU must rise
If all three behave, the loss is fine and a diverging fine-tune is an optimization problem of
the network (step size, optimizer); if they diverge too, the loss is the problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l2sq.data import GeomTargets, GTSuperquadrics, ViewPool, collate, make_sample
from l2sq.geom_loss import occupancy_iou, superflex_loss
from l2sq.sq import EPS_MAX, EPS_MIN, rot6d_to_matrix


def to_raw(sq: dict) -> dict:
    """Unconstrained parameters (the decoder's own parameterization) from superquadric values."""
    raw = {"t": sq["t"].clone(), "r6": sq["R"][..., :2].transpose(-1, -2).reshape(*sq["R"].shape[:-2], 6).clone(),
           "ls": sq["scale"].log().clone(),
           "le": torch.logit(((sq["eps"] - EPS_MIN) / (EPS_MAX - EPS_MIN)).clamp(1e-4, 1 - 1e-4)).clone()}
    if "taper" in sq:
        raw.update(ta=torch.atanh(sq["taper"].clamp(-0.999, 0.999)).clone(),
                   be=torch.logit(sq["bend"].clamp(1e-4, 1 - 1e-4)).clone(),
                   al=torch.stack([sq["alpha"].sin(), sq["alpha"].cos()], -1).clone())
    return {k: v.requires_grad_(True) for k, v in raw.items()}


def from_raw(raw: dict, exist_logit: torch.Tensor) -> dict:
    sq = {"t": raw["t"], "R": rot6d_to_matrix(raw["r6"]), "scale": raw["ls"].exp(),
          "eps": EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(raw["le"]), "exist_logit": exist_logit}
    if "ta" in raw:
        vec = F.normalize(raw["al"], dim=-1)
        sq.update(taper=torch.tanh(raw["ta"]), bend=torch.sigmoid(raw["be"]),
                  alpha=torch.atan2(vec[..., 0], vec[..., 1]), alpha_vec=vec)
    return sq


def run(name: str, start: dict, tgt: dict, w: dict, iters: int, lr: float, mode: str) -> None:
    raw = to_raw(start)
    exist_logit = start["exist_logit"].detach()
    opt = torch.optim.Adam(raw.values(), lr=lr)
    keep = torch.sigmoid(exist_logit) > 0.5
    line = []
    for it in range(iters + 1):
        sq = from_raw(raw, exist_logit)
        if it % max(1, iters // 5) == 0:
            line.append(f"{it}: IoU {occupancy_iou(sq, tgt, keep).item():.3f}")
        if it == iters:
            break
        loss, parts = superflex_loss(sq, tgt, w, exist_mode=mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{name:9s} " + "  ".join(line) + f"   (last loss {loss.item():.3f}, sdf {parts['sdf']:.3f})", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.check_loss", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--gt-sq", required=True)
    p.add_argument("--deform", action="store_true")
    p.add_argument("--init", help="decoder.pt to get the 'decoder' starting point from (optional)")
    p.add_argument("-n", type=int, default=8, help="views 0..n-1 of --train define the sample frame")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3, help="Adam on raw parameters (SuperFlex: 2e-3 .. 2e-2)")
    p.add_argument("--noise", type=float, default=0.05, help="perturbation of t (sample units) and log-scale")
    p.add_argument("--exist-mode", default="hard")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    torch.manual_seed(0)
    dev = a.device
    pool = ViewPool(a.train)
    gt = GTSuperquadrics(a.gt_sq, deform=a.deform)
    geom = GeomTargets(Path(a.train).parent / "gt_world.npz", Path(a.train) / "cameras.json")
    smp = make_sample(pool, gt, list(range(a.n)))
    P = 16
    tgt = collate(pool, [smp], P, dev, geom, np.random.default_rng(0), 4096, 32768)
    w = dict(iou=0.2, sdf=0.2, overlap=5.0, sparsity=1.26, exist_sf=0.02)
    M = len(gt)
    keys = ["t", "R", "scale", "eps"] + (["taper", "bend", "alpha"] if a.deform else [])
    gt_sq = {k: tgt[k][:, :M] for k in keys}
    gt_sq["exist_logit"] = torch.full((1, M), 10.0, device=dev)
    print(f"sample frame: views 0..{a.n - 1}, s = {smp.scale_s:.3f}, {M} pseudo-GT superquadrics; "
          f"Adam lr {a.lr}, {a.iters} iterations, exist-mode {a.exist_mode}", file=sys.stderr)

    run("gt", gt_sq, tgt, w, a.iters, a.lr, a.exist_mode)
    noisy = dict(gt_sq, t=gt_sq["t"] + a.noise * torch.randn_like(gt_sq["t"]),
                 scale=gt_sq["scale"] * torch.exp(a.noise * torch.randn_like(gt_sq["scale"])))
    run("gt+noise", noisy, tgt, w, a.iters, a.lr, a.exist_mode)

    if a.init:
        from l2sq.model import FrozenVGGT, SQDecoder
        ck = torch.load(a.init, map_location="cpu")
        ar = ck["args"]
        dec = SQDecoder(ar["d_model"], ar["queries"], ar["layers"], ar["heads"], ar["ffn"],
                        deform=ar.get("deform", False)).to(dev).eval()
        dec.load_state_dict(ck["decoder"])
        with torch.no_grad():
            pred = dec(FrozenVGGT(ar["vggt_layer"]).to(dev)(tgt["images"]))
        start = {k: pred[k].detach() for k in keys + ["exist_logit"]}
        run("decoder", start, tgt, w, a.iters, a.lr, a.exist_mode)


if __name__ == "__main__":
    main()
