"""Why is a geometric-loss run's eval (chamfer-gt, IoU) much worse than its training loss?

Run from the repo root (GPU):

    python -m l2sq.diagnose runs/chair_L1_cd
    python -m l2sq.diagnose runs/chair_L1_cd --split dataset/chair/test -n 1 8

Prints for one split (random views, like the eval):
  1. the loss terms in eval mode (compare with the last train_loss values), and the same loss
     for the pseudo-GT superquadrics in the same samples: if the pseudo-GT scores lower, the loss
     prefers the better decomposition and the decoder is stuck in a local minimum (optimization
     problem); if it scores higher, the loss itself prefers what the decoder found
  2. per query: existence p, usage (share of GT points it is the closest primitive to),
     sq->points distance (how far its own surface is from the GT points), largest half-axis
  3. chamfer-gt / acc / comp / IoU for several choices of the kept primitives:
        p>0.5     what eval, predict and export use
        all       all 16 queries
        used      the ones closest to > 24 of 4096 GT points (the existence target of the loss)
     If 'all' or 'used' is much better than 'p>0.5', the existence head disagrees with the
     geometry: the loss is fine on what it optimizes, the eval just drops parts it relied on.
Distances in world units (object radius ~1), symmetric mean like the training eval, but with
even (area-weighted) surface samples, so 'p>0.5' differs slightly from the logged chamfer-gt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from l2sq.data import GeomTargets, GTSuperquadrics, ViewPool, collate, make_sample, random_views
from l2sq.geom_loss import EXIST_POINTS, occupancy_iou
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.sq import sample_surface_uniform
from l2sq.train import compute_loss, loss_config


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.diagnose", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="training run folder with decoder.pt")
    p.add_argument("--split", help="split folder (default: the run's --train)")
    p.add_argument("-n", type=int, nargs="+", default=[1, 8], help="numbers of input views")
    p.add_argument("--samples", type=int, default=8, help="random view sets per N")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--sq-samples", type=int, default=1024, help="surface points per primitive")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    run = Path(a.run)
    ckpt = torch.load(run / "decoder.pt", map_location="cpu")
    args = ckpt["args"]
    deform = args.get("deform", False)
    dec = SQDecoder(args["d_model"], args["queries"], args["layers"], args["heads"], args["ffn"], deform=deform)
    dec.load_state_dict(ckpt["decoder"])
    dec.to(a.device).eval()
    vggt = FrozenVGGT(args["vggt_layer"]).to(a.device)
    split = Path(a.split or args["train"])
    pool = ViewPool(split)
    gt = GTSuperquadrics(args["gt_sq"], deform=deform)
    geom = GeomTargets(args.get("gt_world") or Path(args["train"]).parent / "gt_world.npz",
                       Path(args["train"]) / "cameras.json")
    cfg = loss_config(args)
    P = args["queries"]
    rng = np.random.default_rng(1234)
    print(f"{run} (step {ckpt['step']}, loss {cfg['loss']}, exist-mode {cfg['exist_mode']}) on {split}, "
          f"N in {a.n}, {a.samples} view sets each", file=sys.stderr)

    parts_sum: dict[str, list] = {}
    parts_gt: dict[str, list] = {}
    q = {k: [] for k in ("p", "usage", "sq2p", "size")}           # per sample (P,)
    rows = {k: {m: [] for m in ("chamfer", "acc", "comp", "iou", "count")} for k in ("p>0.5", "all", "used")}
    for n in a.n:
        samples = [make_sample(pool, gt, random_views(rng, len(pool), n)) for _ in range(a.samples)]
        for i in range(0, len(samples), a.batch):
            tgt = collate(pool, samples[i:i + a.batch], P, a.device, geom, rng,
                          args.get("n_surf", 4096), args.get("n_vol", 32768))
            pred = dec(vggt(tgt["images"]))
            _, parts = compute_loss(pred, tgt, cfg)
            for k, v in parts.items():
                parts_sum.setdefault(k, []).append(v)
            gt_pred = {k: tgt[k] for k in ("t", "R", "scale", "eps", "taper", "bend", "alpha") if k in tgt}
            gt_pred["exist_logit"] = torch.where(tgt["exist"] > 0.5, 10.0, -10.0)
            _, parts_g = compute_loss(gt_pred, tgt, {**cfg, "w_mse": 0.0})
            for k, v in parts_g.items():
                parts_gt.setdefault(k, []).append(v)

            s = tgt["s"]                                                            # (B,)
            pts = sample_surface_uniform(pred["scale"], pred["eps"], pred["R"], pred["t"], a.sq_samples,
                                         pred.get("taper"), pred.get("bend"), pred.get("alpha"))   # (B,P,S,3)
            B, _, S, _ = pts.shape
            surf = tgt["surf"]
            M = surf.shape[1]
            D = torch.cdist(surf, pts.reshape(B, P * S, 3)).view(B, M, P, S)
            d_pt = D.min(-1).values                                                 # (B,M,P) GT point -> primitive
            d_sq = D.min(1).values                                                  # (B,P,S) primitive sample -> GT
            counts = torch.bincount((d_pt.argmin(-1) + torch.arange(B, device=a.device)[:, None] * P).flatten(),
                                    minlength=B * P).view(B, P).float()
            prob = torch.sigmoid(pred["exist_logit"])
            masks = {"p>0.5": prob > 0.5, "all": torch.ones_like(prob, dtype=torch.bool),
                     "used": counts > EXIST_POINTS * M / 4096}
            for name, keep in masks.items():
                keep = torch.where(keep.any(1, keepdim=True), keep, torch.ones_like(keep))
                acc = torch.stack([d_sq[b][keep[b]].mean() for b in range(B)])
                comp = d_pt.masked_fill(~keep[:, None], float("inf")).min(-1).values.mean(-1)
                r = rows[name]
                r["acc"] += (acc * s).tolist()
                r["comp"] += (comp * s).tolist()
                r["chamfer"] += (0.5 * (acc + comp) * s).tolist()
                r["iou"] += occupancy_iou(pred, tgt, keep).tolist()
                r["count"] += keep.sum(1).float().tolist()
            q["p"].append(prob.cpu())
            q["usage"].append((counts / M).cpu())
            q["sq2p"].append((d_sq.mean(-1) * s[:, None]).cpu())
            q["size"].append((pred["scale"].amax(-1) * s[:, None]).cpu())

    w = cfg["w"]
    total = lambda d: sum(w.get(k, 0.0) * np.mean(v) for k, v in d.items() if k in w)
    print("\n1. loss terms (eval mode, mean over batches):")
    print(f"   decoder    total {total(parts_sum):.4f}  " + "  ".join(f"{k} {np.mean(v):.4g}" for k, v in parts_sum.items()))
    print(f"   pseudo-GT  total {total(parts_gt):.4f}  " + "  ".join(f"{k} {np.mean(v):.4g}" for k, v in parts_gt.items()))

    qs = {k: torch.cat(v).mean(0) for k, v in q.items()}
    on = torch.cat(q["p"]) > 0.5
    print("\n2. per query (mean over samples), sorted by usage:")
    print("   query  p_exist  on(%)  usage  sq->pts  max half-axis")
    for j in qs["usage"].argsort(descending=True).tolist():
        print(f"   {j:5d}  {qs['p'][j]:7.3f}  {100 * on[:, j].float().mean():5.0f}  {qs['usage'][j]:5.3f}  "
              f"{qs['sq2p'][j]:7.4f}  {qs['size'][j]:7.3f}")

    print("\n3. kept primitives -> metrics vs GT (world units):")
    print("   keep     #SQ   chamfer   acc     comp    IoU")
    for name, r in rows.items():
        print(f"   {name:6s} {np.mean(r['count']):5.1f}  {np.mean(r['chamfer']):.4f}  {np.mean(r['acc']):.4f}  "
              f"{np.mean(r['comp']):.4f}  {np.mean(r['iou']):.3f}")


if __name__ == "__main__":
    main()
