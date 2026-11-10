"""Train the superquadric decoder on frozen VGGT tokens (single-object sanity check, MSE).

Run from the repo root:

    python -m l2sq.train --train dataset/chair/train --test dataset/chair/test \
        --gt-sq dataset/chair/gt_sq_rigid.npz --out runs/chair_L1

Each batch draws one N in [n_min, n_max]; each sample draws N distinct views from the TRAIN pool,
the first being the reference. Targets are the pseudo-GT superquadrics in that sample's VGGT
frame (l2sq/data.py). Loss: MSE on translation, rotation matrix, scale, exponents for the GT
slots (GT superquadric m <-> query m, a fixed assignment that is valid for a single object)
+ BCE on existence for all P queries.

Evaluation uses fixed samples for N in {1,2,4,8} from the TRAIN views and from the TEST views
(different cameras). Distances marked 'world' are multiplied back by s, i.e. in units of the
object's radius (mesh scaled to the unit sphere), comparable with recon.eval_vggt.

Writes to --out: args.json, metrics.jsonl (one line per evaluation), decoder.pt,
examples/<split>_N<n>_<k>/{gt_sq.ply, pred_sq.ply, sample.json} for the viewer:
    python -m recon.view --mesh runs/chair_L1/examples/test_N8_0/gt_sq.ply runs/chair_L1/examples/test_N8_0/pred_sq.ply
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l2sq.data import GTSuperquadrics, ViewPool, collate, make_sample, random_views
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.sq import chamfer, geodesic_deg, sample_surface, sq_mesh

EVAL_NS = (1, 2, 4, 8)


# --------------------------------------------------------------------------- loss

def mse_loss(pred: dict, tgt: dict, w: dict) -> tuple[torch.Tensor, dict]:
    m = tgt["exist"]                                   # (B,P), 1 for GT slots
    n = m.sum().clamp_min(1.0)

    def masked(a, b):  # per-slot mean squared error, averaged over the GT slots only
        return (((a - b) ** 2).flatten(2).mean(-1) * m).sum() / n

    parts = {
        "t": masked(pred["t"], tgt["t"]),
        "R": masked(pred["R"], tgt["R"]),
        "scale": masked(pred["scale"], tgt["scale"]),
        "eps": masked(pred["eps"], tgt["eps"]),
        "exist": F.binary_cross_entropy_with_logits(pred["exist_logit"], m),
    }
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: v.item() for k, v in parts.items()}


# --------------------------------------------------------------------------- evaluation

def eval_samples(pool, gt, ns, per_n, seed):
    rng = np.random.default_rng(seed)
    return {n: [make_sample(pool, gt, random_views(rng, len(pool), n)) for _ in range(per_n)] for n in ns}


@torch.no_grad()
def evaluate(vggt, dec, pool, samples_by_n, P, batch, device, w) -> dict:
    dec.eval()
    out = {}
    for n, samples in samples_by_n.items():
        acc = {k: [] for k in ["loss", "rot_deg", "t_err_world", "scale_rel_err", "eps_err",
                               "exist_acc", "n_pred", "chamfer_world"]}
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            tgt = collate(pool, chunk, P, device)
            pred = dec(vggt(tgt["images"]))
            loss, _ = mse_loss(pred, tgt, w)
            acc["loss"].append(float(loss))
            m = tgt["exist"] > 0.5
            s = tgt["s"][:, None]
            acc["rot_deg"] += geodesic_deg(pred["R"], tgt["R"])[m].tolist()
            acc["t_err_world"] += ((pred["t"] - tgt["t"]).norm(dim=-1) * s)[m].tolist()
            acc["scale_rel_err"] += ((pred["scale"] - tgt["scale"]).abs() / tgt["scale"]).mean(-1)[m].tolist()
            acc["eps_err"] += (pred["eps"] - tgt["eps"]).abs().mean(-1)[m].tolist()
            exist_pred = torch.sigmoid(pred["exist_logit"]) > 0.5
            acc["exist_acc"] += (exist_pred == m).float().mean(-1).tolist()
            acc["n_pred"] += exist_pred.sum(-1).float().tolist()
            for b in range(len(chunk)):
                keep = exist_pred[b] if exist_pred[b].any() else m[b]
                p_pts = sample_surface(pred["scale"][b][keep], pred["eps"][b][keep], pred["R"][b][keep], pred["t"][b][keep]).reshape(-1, 3)
                g_pts = sample_surface(tgt["scale"][b][m[b]], tgt["eps"][b][m[b]], tgt["R"][b][m[b]], tgt["t"][b][m[b]]).reshape(-1, 3)
                acc["chamfer_world"].append(float(chamfer(p_pts, g_pts) * tgt["s"][b]))
        out[f"N{n}"] = {k: float(np.mean(v)) for k, v in acc.items()}
    out["all"] = {k: float(np.mean([out[f"N{n}"][k] for n in samples_by_n])) for k in out[f"N{EVAL_NS[0]}"]}
    dec.train()
    return out


@torch.no_grad()
def export_examples(vggt, dec, pool, samples, split, P, device, folder: Path):
    dec.eval()
    for k, smp in enumerate(samples):
        tgt = collate(pool, [smp], P, device)
        pred = {key: v[0].cpu() for key, v in dec(vggt(tgt["images"])).items()}
        s = smp.scale_s
        d = folder / f"{split}_N{len(smp.views)}_{k}"
        d.mkdir(parents=True, exist_ok=True)
        keep = torch.sigmoid(pred["exist_logit"]) > 0.5
        if not keep.any():
            keep[:] = True
        # metric (world units) in the reference camera's frame -> recon.view starts at this camera
        sq_mesh(smp.scale * s, smp.eps, smp.R, smp.t * s).export(d / "gt_sq.ply")
        sq_mesh(pred["scale"][keep] * s, pred["eps"][keep], pred["R"][keep], pred["t"][keep] * s).export(d / "pred_sq.ply")
        (d / "sample.json").write_text(json.dumps({
            "split": split, "views": smp.views, "images": [f"images/{v:03d}.png" for v in smp.views],
            "pool": str(pool.folder), "s": s, "n_pred": int(keep.sum())}, indent=2))
    dec.train()


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="render folder of the training views (with depth/)")
    p.add_argument("--test", required=True, help="render folder of the test views (different cameras)")
    p.add_argument("--gt-sq", required=True, help="rigid pseudo-GT superquadrics, world frame (.npz)")
    p.add_argument("--out", required=True, help="run folder")
    # model
    p.add_argument("--layers", type=int, default=1, help="decoder layers L")
    p.add_argument("--queries", type=int, default=16, help="P")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--vggt-layer", type=int, default=23, help="aggregator layer (4, 11, 17 or 23)")
    # optimization
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=2, help="samples per step (each N views)")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--w-t", type=float, default=1.0)
    p.add_argument("--w-R", type=float, default=1.0)
    p.add_argument("--w-scale", type=float, default=1.0)
    p.add_argument("--w-eps", type=float, default=1.0)
    p.add_argument("--w-exist", type=float, default=1.0)
    # logging / eval
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-per-n", type=int, default=8, help="fixed eval samples per N and split")
    p.add_argument("--examples", type=int, default=2, help="exported example meshes per N in {1,8} and split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")
    device = a.device
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    w = {"t": a.w_t, "R": a.w_R, "scale": a.w_scale, "eps": a.w_eps, "exist": a.w_exist}

    print("loading views and pseudo-GT ...", file=sys.stderr)
    train_pool, test_pool = ViewPool(a.train), ViewPool(a.test)
    gt = GTSuperquadrics(a.gt_sq)
    print(f"train views {len(train_pool)}, test views {len(test_pool)}, GT superquadrics {len(gt)}, P={a.queries}",
          file=sys.stderr)
    eval_sets = {
        "train": (train_pool, eval_samples(train_pool, gt, EVAL_NS, a.eval_per_n, a.seed + 1)),
        "test": (test_pool, eval_samples(test_pool, gt, EVAL_NS, a.eval_per_n, a.seed + 2)),
    }

    vggt = FrozenVGGT(a.vggt_layer).to(device)
    dec = SQDecoder(a.d_model, a.queries, a.layers, a.heads, a.ffn).to(device)
    print(f"decoder parameters: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M", file=sys.stderr)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))

    rng = np.random.default_rng(a.seed)
    metrics_f = open(out / "metrics.jsonl", "a")
    running, t0 = {}, time.time()
    for step in range(1, a.steps + 1):
        n = int(rng.integers(a.n_min, a.n_max + 1))
        batch = [make_sample(train_pool, gt, random_views(rng, len(train_pool), n)) for _ in range(a.batch)]
        tgt = collate(train_pool, batch, a.queries, device)
        loss, parts = mse_loss(dec(vggt(tgt["images"])), tgt, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()

        for k, v in {"loss": loss.item(), **parts}.items():
            running[k] = running.get(k, 0.0) + v
        if step % a.log_every == 0:
            msg = "  ".join(f"{k} {v / a.log_every:.4g}" for k, v in running.items())
            print(f"step {step:5d}  {msg}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / a.log_every:.2f}s/step", file=sys.stderr)
            running, t0 = {}, time.time()

        if step % a.eval_every == 0 or step == a.steps:
            res = {"step": step}
            for split, (pool, samples) in eval_sets.items():
                res[split] = evaluate(vggt, dec, pool, samples, a.queries, a.batch, device, w)
            metrics_f.write(json.dumps(res) + "\n")
            metrics_f.flush()
            for split in eval_sets:
                r = res[split]["all"]
                print(f"[eval {split:5s} step {step}] loss {r['loss']:.4f}  rot {r['rot_deg']:.1f}deg  "
                      f"t {r['t_err_world']:.4f}  scale {100 * r['scale_rel_err']:.1f}%  "
                      f"chamfer {r['chamfer_world']:.4f}  exist-acc {r['exist_acc']:.3f}  "
                      f"#SQ {r['n_pred']:.1f}/{len(gt)}", file=sys.stderr)
            torch.save({"decoder": dec.state_dict(), "args": vars(a), "step": step}, out / "decoder.pt")
            t0 = time.time()

    for split, (pool, samples) in eval_sets.items():
        for n in (1, 8):
            export_examples(vggt, dec, pool, samples[n][:a.examples], split, a.queries, device, out / "examples")
    final = {split: res[split] for split in eval_sets}
    (out / "eval_final.json").write_text(json.dumps(final, indent=2))
    print("\nfinal (per N):", file=sys.stderr)
    for split in eval_sets:
        for n in EVAL_NS:
            r = final[split][f"N{n}"]
            print(f"  {split:5s} N={n}: rot {r['rot_deg']:.1f}deg  t {r['t_err_world']:.4f}  "
                  f"scale {100 * r['scale_rel_err']:.1f}%  chamfer {r['chamfer_world']:.4f}  #SQ {r['n_pred']:.1f}",
                  file=sys.stderr)
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
