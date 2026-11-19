"""Train the superquadric decoder on frozen VGGT tokens (single-object sanity check, MSE).

Run from the repo root:

    python -m l2sq.train --train dataset/chair/train --test dataset/chair/test \
        --gt-sq dataset/chair/gt_sq_rigid.npz --out runs/chair_L1

Each batch draws one N in [n_min, n_max]; each sample draws N distinct views from the TRAIN pool,
the first being the reference. Targets are the pseudo-GT superquadrics in that sample's VGGT
frame (l2sq/data.py). Loss: MSE on translation, rotation matrix, log(scale) (--scale-loss),
exponents (--deform: + tapering, bending amount, bending direction) for the GT slots (GT superquadric m <-> query m, a fixed assignment that is valid for a single object)
+ BCE on existence for all P queries.

Evaluation uses fixed samples for N in {1,2,4,8} from the TRAIN views and from the TEST views
(different cameras). Distances marked 'world' are multiplied back by s, i.e. in units of the
object's radius (mesh scaled to the unit sphere), comparable with recon.eval_vggt.

TensorBoard (<out>/tb): training losses, lr, s/step, and every eval metric with train and test
views overlaid ("eval_all/..." averaged over N, "eval_N<n>/..." per N):
    tensorboard --logdir runs --port 6006     (then forward the port like the viewer)

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

from l2sq.data import GeomTargets, GTSuperquadrics, ViewPool, collate, make_sample, random_views
from l2sq.geom_loss import chamfer_loss, occupancy_iou, superflex_loss
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.sq import chamfer, geodesic_deg, sample_surface_uniform, sq_mesh

EVAL_NS = (1, 2, 4, 8)


# --------------------------------------------------------------------------- loss

def mse_loss(pred: dict, tgt: dict, w: dict, scale_loss: str = "log") -> tuple[torch.Tensor, dict]:
    """scale_loss 'log': MSE of log(scale), i.e. relative size errors, so a 50 % error on a
    thin chair leg costs as much as on the seat ('linear' lets thin axes go almost free)."""
    m = tgt["exist"]                                   # (B,P), 1 for GT slots
    n = m.sum().clamp_min(1.0)

    def masked(a, b):  # per-slot mean squared error, averaged over the GT slots only
        return (((a - b) ** 2).flatten(2).mean(-1) * m).sum() / n

    parts = {
        "t": masked(pred["t"], tgt["t"]),
        "R": masked(pred["R"], tgt["R"]),
        "scale": (masked(pred["scale"].log(), tgt["scale"].log()) if scale_loss == "log"
                  else masked(pred["scale"], tgt["scale"])),
        "eps": masked(pred["eps"], tgt["eps"]),
        "exist": F.binary_cross_entropy_with_logits(pred["exist_logit"], m),
    }
    if "taper" in tgt:
        parts["taper"] = masked(pred["taper"], tgt["taper"])
        parts["bend"] = masked(pred["bend"], tgt["bend"])
        # direction as a unit 2D vector, weighted by the target bending amount (meaningless when unbent)
        tvec = torch.stack([tgt["alpha"].sin(), tgt["alpha"].cos()], -1)            # (B,P,3,2)
        d2 = ((pred["alpha_vec"] - tvec) ** 2).sum(-1) * tgt["bend"]               # (B,P,3)
        parts["alpha"] = (d2.mean(-1) * m).sum() / n
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: v.item() for k, v in parts.items()}


def loss_weights(args: dict) -> dict:
    w = {"t": args["w_t"], "R": args["w_R"], "scale": args["w_scale"], "eps": args["w_eps"], "exist": args["w_exist"]}
    if args.get("deform"):
        w.update(taper=args["w_taper"], bend=args["w_bend"], alpha=args["w_bend"])
    return w


def loss_config(args: dict) -> dict:
    """Everything compute_loss needs, from the run's arguments (also used by l2sq.predict)."""
    w = loss_weights(args)
    w.update(iou=args.get("w_iou", 0.2), sdf=args.get("w_sdf", 0.2), overlap=args.get("w_overlap", 5.0),
             sparsity=args.get("w_sparsity", 1.26), exist_sf=args.get("w_exist_sf", 0.02),
             cd_p2sq=args.get("w_p2sq", 1.0), cd_sq2p=args.get("w_sq2p", 1.0), parsimony=args.get("w_par", 0.06),
             exist_cd=args.get("w_exist_cd", 0.01), cd_rigid=args.get("w_rigid", 1.0))
    # runs before 2026-11-17 stored None and were trained with "soft" for chamfer
    exist_mode = args.get("exist_mode") or ("detach" if args.get("loss") == "chamfer" else "hard")
    return {"loss": args.get("loss", "mse"), "w": w, "scale_loss": args.get("scale_loss", "linear"),
            "w_mse": args.get("w_mse", 0.0), "truncation": args.get("truncation", 0.05),
            "neg_slope": args.get("neg_slope", 0.1), "exist_mode": exist_mode,
            "n_sq_samples": args.get("n_sq_samples", 256), "tau_assign": args.get("tau_assign", 0.01)}


def compute_loss(pred: dict, tgt: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """--loss mse: parameter MSE vs the pseudo-GT (fixed slots).
    --loss chamfer: SuperDec's Chamfer + parsimony + existence vs the GT surface points (from scratch).
    --loss superflex: SuperFlex's IoU + SDF (+ overlap, sparsity, existence) vs the GT geometry,
    plus w_mse x the parameter MSE if --w-mse > 0."""
    if cfg["loss"] == "mse":
        return mse_loss(pred, tgt, cfg["w"], cfg["scale_loss"])
    if cfg["loss"] == "chamfer":
        total, parts = chamfer_loss(pred, tgt, cfg["w"], cfg["exist_mode"], cfg["n_sq_samples"], cfg["tau_assign"],
                                    rigid_term=cfg["w"]["cd_rigid"] > 0)
    else:
        total, parts = superflex_loss(pred, tgt, cfg["w"], cfg["truncation"], cfg["neg_slope"], cfg["exist_mode"])
    if cfg["w_mse"] > 0:
        l_mse, mse_parts = mse_loss(pred, tgt, cfg["w"], cfg["scale_loss"])
        total = total + cfg["w_mse"] * l_mse
        parts.update({f"mse_{k}": v for k, v in mse_parts.items()}, mse=l_mse.item())
    return total, parts


# --------------------------------------------------------------------------- evaluation

def eval_samples(pool, gt, ns, per_n, seed):
    rng = np.random.default_rng(seed)
    return {n: [make_sample(pool, gt, random_views(rng, len(pool), n)) for _ in range(per_n)] for n in ns}


def sq_args(d: dict, b: int, sel, s: float = 1.0) -> tuple:
    """(scale, eps, R, t, n_eta, n_omega, taper, bend, alpha) of sample b, slots sel, for
    sample_surface; lengths multiplied by s (bend is scale-free)."""
    deform = [d[k][b][sel] for k in ("taper", "bend", "alpha")] if "taper" in d else [None] * 3
    return (d["scale"][b][sel] * s, d["eps"][b][sel], d["R"][b][sel], d["t"][b][sel] * s, 16, 32, *deform)


def even_points(d: dict, b: int, sel, gen: torch.Generator, n: int = 512) -> torch.Tensor:
    """n points per kept superquadric, spread evenly by area (as recon.eval_sq / l2sq.diagnose)."""
    sc, ep, R, t, _, _, ta, be, al = sq_args(d, b, sel)
    return sample_surface_uniform(sc, ep, R, t, n, ta, be, al, generator=gen).reshape(-1, 3)


@torch.no_grad()
def evaluate(vggt, dec, pool, samples_by_n, P, batch, device, cfg: dict, geom: GeomTargets) -> dict:
    """Parameter errors are per GT slot (query m vs GT superquadric m): meaningful for --loss mse
    (and --w-mse > 0); with the pure geometric loss the decoder picks its own slot order, so use
    iou (occupancy of the predicted union vs the mesh) and chamfer_gt (vs GT surface points) there."""
    dec.eval()
    out = {}
    rng = np.random.default_rng(1234)   # same GT point subsets at every evaluation
    gen = torch.Generator(device=device).manual_seed(1234)   # same SQ surface samples, training RNG untouched
    for n, samples in samples_by_n.items():
        acc = {k: [] for k in ["loss", "rot_deg", "t_err_world", "scale_rel_err", "eps_err",
                               "exist_acc", "n_pred", "chamfer_world", "iou", "chamfer_gt", "iou_target"]}
        if dec.deform:
            acc.update(taper_err=[], bend_err=[])
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            tgt = collate(pool, chunk, P, device, geom, rng, cfg.get("n_surf", 4096), cfg.get("n_vol", 32768))
            pred = dec(vggt(tgt["images"]))
            loss, _ = compute_loss(pred, tgt, cfg)
            acc["loss"].append(float(loss))
            m = tgt["exist"] > 0.5
            s = tgt["s"][:, None]
            acc["rot_deg"] += geodesic_deg(pred["R"], tgt["R"])[m].tolist()
            acc["t_err_world"] += ((pred["t"] - tgt["t"]).norm(dim=-1) * s)[m].tolist()
            acc["scale_rel_err"] += ((pred["scale"] - tgt["scale"]).abs() / tgt["scale"]).mean(-1)[m].tolist()
            acc["eps_err"] += (pred["eps"] - tgt["eps"]).abs().mean(-1)[m].tolist()
            if dec.deform:
                acc["taper_err"] += (pred["taper"] - tgt["taper"]).abs().mean(-1)[m].tolist()
                acc["bend_err"] += (pred["bend"] - tgt["bend"]).abs().mean(-1)[m].tolist()
            exist_pred = torch.sigmoid(pred["exist_logit"]) > 0.5
            acc["exist_acc"] += (exist_pred == m).float().mean(-1).tolist()
            acc["n_pred"] += exist_pred.sum(-1).float().tolist()
            for b in range(len(chunk)):
                keep = exist_pred[b] if exist_pred[b].any() else m[b]
                p_pts = even_points(pred, b, keep, gen)
                g_pts = even_points(tgt, b, m[b], gen)
                acc["chamfer_world"].append(float(chamfer(p_pts, g_pts) * tgt["s"][b]))
                acc["chamfer_gt"].append(float(chamfer(p_pts, tgt["surf"][b]) * tgt["s"][b]))
            keep_all = torch.where(exist_pred.any(-1, keepdim=True), exist_pred, m)
            acc["iou"] += occupancy_iou(pred, tgt, keep_all).tolist()
            acc["iou_target"] += occupancy_iou(tgt, tgt, m).tolist()   # pseudo-GT itself: checks labels + distances
        out[f"N{n}"] = {k: float(np.mean(v)) for k, v in acc.items()}
    out["all"] = {k: float(np.mean([out[f"N{n}"][k] for n in samples_by_n])) for k in next(iter(out.values()))}
    dec.train()
    return out


@torch.no_grad()
def export_examples(vggt, dec, pool, samples, split, P, device, folder: Path):
    dec.eval()
    for k, smp in enumerate(samples):
        tgt = collate(pool, [smp], P, device)
        pred = {key: v.cpu() for key, v in dec(vggt(tgt["images"])).items()}
        tgt = {key: (v.cpu() if isinstance(v, torch.Tensor) else v) for key, v in tgt.items()}
        s = smp.scale_s
        d = folder / f"{split}_N{len(smp.views)}_{k}"
        d.mkdir(parents=True, exist_ok=True)
        keep = torch.sigmoid(pred["exist_logit"][0]) > 0.5
        if not keep.any():
            keep[:] = True
        gt_slots = tgt["exist"][0] > 0.5
        # metric (world units) in the reference camera's frame -> recon.view starts at this camera
        for name, d_, sel in [("gt_sq.ply", tgt, gt_slots), ("pred_sq.ply", pred, keep)]:
            sc, ep, R, t, _, _, ta, be, al = sq_args(d_, 0, sel, s)
            sq_mesh(sc, ep, R, t, taper=ta, bend=be, alpha=al).export(d / name)
        (d / "sample.json").write_text(json.dumps({
            "split": split, "views": smp.views, "images": [f"images/{v:03d}.png" for v in smp.views],
            "pool": str(pool.folder), "s": s, "n_pred": int(keep.sum()), "deform": dec.deform}, indent=2))
    dec.train()


# --------------------------------------------------------------------------- main

def make_writer(logdir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed (pip install tensorboard) -- logging to metrics.jsonl only", file=sys.stderr)
        return None
    return SummaryWriter(str(logdir))


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
    p.add_argument("--w-taper", type=float, default=1.0, help="--deform only")
    p.add_argument("--w-bend", type=float, default=1.0, help="--deform only: bending amount and direction")
    p.add_argument("--deform", action="store_true",
                   help="deformable superquadrics: GT fit with tapering/bending (gt_sq_deform.npz), 26 outputs per query")
    p.add_argument("--loss", choices=["mse", "chamfer", "superflex"], default="mse",
                   help="mse: parameters vs pseudo-GT (fixed slots); chamfer: SuperDec (self-supervised, from "
                        "scratch); superflex: IoU + SDF vs the GT geometry (needs --init)")
    p.add_argument("--w-p2sq", type=float, default=1.0, help="--loss chamfer weights (SuperDec)")
    p.add_argument("--w-sq2p", type=float, default=1.0)
    p.add_argument("--w-par", type=float, default=0.06, help="parsimony (SuperDec: 0.06; supp.: 0.1 then 0.6)")
    p.add_argument("--w-exist-cd", type=float, default=0.01)
    p.add_argument("--w-rigid", type=float, default=1.0,
                   help="--deform: Chamfer of the rigid part (tapering/bending off), SuperFlex L_rigid; 0 = off")
    p.add_argument("--deform-start", type=int, default=0,
                   help="--deform: tapering/bending forced to 0 in the loss until this step (rigid first)")
    p.add_argument("--n-sq-samples", type=int, default=256, help="--loss chamfer: surface points per superquadric")
    p.add_argument("--tau-assign", type=float, default=0.01, help="soft point-to-primitive assignment temperature")
    p.add_argument("--tau-start", type=float, default=None,
                   help="--loss chamfer: start the assignment temperature here and lower it geometrically to "
                        "--tau-assign over --tau-anneal steps (soft early = every primitive feels distant "
                        "points, as in deterministic annealing; against parts merged into one box)")
    p.add_argument("--tau-anneal", type=int, default=1500, help="steps of the --tau-start schedule")
    p.add_argument("--w-iou", type=float, default=0.2, help="--loss superflex weights (SuperFlex train.yaml)")
    p.add_argument("--w-sdf", type=float, default=0.2)
    p.add_argument("--w-overlap", type=float, default=5.0)
    p.add_argument("--w-sparsity", type=float, default=1.26)
    p.add_argument("--w-exist-sf", type=float, default=0.02)
    p.add_argument("--w-mse", type=float, default=0.0, help="--loss superflex: + this x the parameter MSE")
    p.add_argument("--exist-mode", choices=["hard", "soft", "detach"], default=None,
                   help="hard: only existing primitives in the loss (for --init; default for superflex); "
                        "detach: existence-weighted with a constant weight, existence learned only from "
                        "the 'used' target (default for chamfer); soft: existence also gets the "
                        "Chamfer gradient (chamfer: inverts existence, see geom_loss.chamfer_loss)")
    p.add_argument("--truncation", type=float, default=0.05, help="SDF truncation delta (normalized units)")
    p.add_argument("--neg-slope", type=float, default=0.1, help="SDF weight omega for points inside")
    p.add_argument("--n-surf", type=int, default=4096, help="GT surface points per sample")
    p.add_argument("--n-vol", type=int, default=32768, help="GT volume (occupancy) points per sample")
    p.add_argument("--init", help="start from a trained decoder.pt (same --deform, layers, ...), e.g. an MSE run; "
                        "--loss superflex needs this: its losses are local and give no gradient from a random init")
    p.add_argument("--gt-world", help="GT surface points, world frame (default: <train>/../gt_world.npz)")
    p.add_argument("--scale-loss", choices=["log", "linear"], default="log",
                   help="MSE on log(scale) (relative errors, default) or on scale (runs/chair_L1 used linear)")
    # logging / eval
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-per-n", type=int, default=8, help="fixed eval samples per N and split")
    p.add_argument("--examples", type=int, default=2, help="exported example meshes per N in {1,8} and split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tb", action="store_true", help="disable TensorBoard logging (<out>/tb)")
    a = p.parse_args(argv)

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")
    device = a.device
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.gt_world is None:
        a.gt_world = str(Path(a.train).parent / "gt_world.npz")
    a.exist_mode = loss_config(vars(a))["exist_mode"]     # store the resolved default with the run
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    cfg = {**loss_config(vars(a)), "n_surf": a.n_surf, "n_vol": a.n_vol}

    print("loading views and pseudo-GT ...", file=sys.stderr)
    train_pool, test_pool = ViewPool(a.train), ViewPool(a.test)
    gt = GTSuperquadrics(a.gt_sq, deform=a.deform)
    geom = GeomTargets(a.gt_world, Path(a.train) / "cameras.json", seed=a.seed)
    print(f"GT geometry: {len(geom.surf)} surface points, {len(geom.vol)} volume points "
          f"({100 * geom.occ.mean():.1f}% inside); loss: {a.loss}", file=sys.stderr)
    print(f"train views {len(train_pool)}, test views {len(test_pool)}, GT superquadrics {len(gt)}, P={a.queries}",
          file=sys.stderr)
    eval_sets = {
        "train": (train_pool, eval_samples(train_pool, gt, EVAL_NS, a.eval_per_n, a.seed + 1)),
        "test": (test_pool, eval_samples(test_pool, gt, EVAL_NS, a.eval_per_n, a.seed + 2)),
    }

    vggt = FrozenVGGT(a.vggt_layer).to(device)
    dec = SQDecoder(a.d_model, a.queries, a.layers, a.heads, a.ffn, deform=a.deform).to(device)
    print(f"decoder parameters: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M", file=sys.stderr)
    if a.init:
        ckpt = torch.load(a.init, map_location="cpu")
        for k in ("d_model", "queries", "layers", "heads", "ffn", "vggt_layer", "deform"):
            if ckpt["args"].get(k, False) != getattr(a, k):
                sys.exit(f"--init {a.init}: {k} = {ckpt['args'].get(k, False)} there, {getattr(a, k)} here")
        dec.load_state_dict(ckpt["decoder"])
        print(f"initialized from {a.init} (step {ckpt['step']}, loss {ckpt['args'].get('loss', 'mse')})", file=sys.stderr)
    elif a.loss == "superflex":
        print("WARNING: --loss superflex from a random init usually collapses (no gradient far from the "
              "object); use --init <mse run>/decoder.pt", file=sys.stderr)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))

    rng = np.random.default_rng(a.seed)
    metrics_f = open(out / "metrics.jsonl", "a")
    tb = None if a.no_tb else make_writer(out / "tb")
    # one writer per eval split: same tags -> TensorBoard overlays "eval_train" and "eval_test" in each chart
    tb_eval = {split: make_writer(out / "tb" / f"eval_{split}") for split in ("train", "test")} if tb else {}
    if tb is not None:
        tb.add_text("args", "```\n" + json.dumps(vars(a), indent=2) + "\n```")
    res = {}

    def run_eval(step: int) -> None:
        res.clear()
        res["step"] = step
        for split, (pool, samples) in eval_sets.items():
            res[split] = evaluate(vggt, dec, pool, samples, a.queries, a.batch, device, cfg, geom)
        metrics_f.write(json.dumps(res) + "\n")
        metrics_f.flush()
        for split, writer in tb_eval.items():
            for key in ["all"] + [f"N{n}" for n in EVAL_NS]:
                for metric, value in res[split][key].items():
                    writer.add_scalar(f"eval_{key}/{metric}", value, step)
            writer.flush()
        for split in eval_sets:
            r = res[split]["all"]
            print(f"[eval {split:5s} step {step}] loss {r['loss']:.4f}  rot {r['rot_deg']:.1f}deg  "
                  f"t {r['t_err_world']:.4f}  scale {100 * r['scale_rel_err']:.1f}%  "
                  f"chamfer {r['chamfer_world']:.4f}  IoU {r['iou']:.3f}  chamfer-gt {r['chamfer_gt']:.4f}  "
                  f"exist-acc {r['exist_acc']:.3f}  IoU-target {r['iou_target']:.3f}  "
                  f"#SQ {r['n_pred']:.1f}/{len(gt)}", file=sys.stderr)
        if step > 0:
            torch.save({"decoder": dec.state_dict(), "args": vars(a), "step": step}, out / "decoder.pt")

    if a.init:   # starting point of the fine-tuning
        run_eval(0)

    running, t0 = {}, time.time()
    for step in range(1, a.steps + 1):
        n = int(rng.integers(a.n_min, a.n_max + 1))
        batch = [make_sample(train_pool, gt, random_views(rng, len(train_pool), n)) for _ in range(a.batch)]
        tgt = collate(train_pool, batch, a.queries, device, geom, rng, a.n_surf, a.n_vol)
        pred = dec(vggt(tgt["images"]))
        if a.deform and step <= a.deform_start:      # rigid phase: deformation outputs get no gradient
            pred = {**pred, "taper": torch.zeros_like(pred["taper"]), "bend": torch.zeros_like(pred["bend"])}
        if a.tau_start is not None:   # geometric from tau_start to tau_assign, then constant
            frac = min(1.0, (step - 1) / max(1, a.tau_anneal))
            cfg["tau_assign"] = a.tau_start * (a.tau_assign / a.tau_start) ** frac
        loss, parts = compute_loss(pred, tgt, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()

        for k, v in {"loss": loss.item(), **parts}.items():
            running[k] = running.get(k, 0.0) + v
        if step % a.log_every == 0:
            msg = "  ".join(f"{k} {v / a.log_every:.4g}" for k, v in running.items())
            sec = (time.time() - t0) / a.log_every
            # peak memory actually used by tensors in this interval (nvidia-smi shows PyTorch's cache instead)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(f"step {step:5d}  {msg}  lr {sched.get_last_lr()[0]:.2e}  {sec:.2f}s/step  "
                  f"peak {mem:.1f} GB", file=sys.stderr)
            if tb is not None:
                for k, v in running.items():
                    tb.add_scalar(f"train_loss/{k}", v / a.log_every, step)
                tb.add_scalar("misc/lr", sched.get_last_lr()[0], step)
                if a.loss == "chamfer":
                    tb.add_scalar("misc/tau_assign", cfg["tau_assign"], step)
                tb.add_scalar("misc/sec_per_step", sec, step)
                tb.add_scalar("misc/gpu_peak_gb", mem, step)
            running, t0 = {}, time.time()

        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
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
                  f"scale {100 * r['scale_rel_err']:.1f}%  chamfer {r['chamfer_world']:.4f}  IoU {r['iou']:.3f}  "
                  f"chamfer-gt {r['chamfer_gt']:.4f}  #SQ {r['n_pred']:.1f}",
                  file=sys.stderr)
    for writer in [tb, *tb_eval.values()]:
        if writer is not None:
            writer.close()
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
