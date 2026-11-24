"""Evaluate parallel (regression) vs flow-matching SQ decoders under image occlusion.

Run from the repo root (GPU):

    python -m l2sq.eval_occlusion \
        --parallel ~/latent2SQ/runs/chair_L1_logscale/decoder.pt \
        --flow ~/rayyan/latent2SQ-v2/runs/flow_tnorm/decoder.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from l2sq.data import GeomTargets, GTSuperquadrics, ViewPool, collate
from l2sq.flow_matching_decoder import FlowSQDenoiser, decode_sq, sample_euler
from l2sq.geom_loss import occupancy_iou
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.sq import chamfer, geodesic_deg
from l2sq.train import _nanmean, eval_samples, even_points

EVAL_NS = (1, 4, 8)
BOX_PS = (0.0, 0.25, 0.5, 0.75)
HALF_SIDES = ("left", "right", "top", "bottom")
METRIC_KEYS = ("rot_deg", "t_err_world", "scale_rel_err", "chamfer_world", "iou", "exist_acc", "n_pred")


def load_decoder(ckpt_path: str | Path, device: str) -> tuple[str, torch.nn.Module, dict, dict]:
    """Load SQDecoder or FlowSQDenoiser checkpoint, auto-detecting model type from ckpt['args']."""
    ckpt = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
    args = ckpt["args"]
    kind = args.get("decoder", "parallel")
    if kind == "parallel":
        dec = SQDecoder(
            args["d_model"], args["queries"], args["layers"], args["heads"], args["ffn"],
            deform=args.get("deform", False),
        )
        dec.load_state_dict(ckpt["decoder"])
    elif kind == "flow":
        dec = FlowSQDenoiser(
            args["d_model"], args["queries"], args["layers"], args["heads"], args["ffn"],
        )
        dec.load_state_dict(ckpt["decoder"])
        if "t_mean" not in ckpt["decoder"] and "t_mean" in args and "t_std" in args:
            dec.set_t_norm(torch.as_tensor(args["t_mean"]), torch.as_tensor(args["t_std"]))
        if "flow_shift" not in ckpt["decoder"] and "flow_shift" in args and "flow_scale" in args:
            dec.set_flow_norm(torch.as_tensor(args["flow_shift"]), torch.as_tensor(args["flow_scale"]))
    else:
        raise ValueError(f"Unknown decoder type {kind!r} in {ckpt_path}")
    return kind, dec.to(device).eval(), args, ckpt


def apply_box_occlusion(images: torch.Tensor, p: float, rng: np.random.Generator, fill: float = 0.5) -> torch.Tensor:
    """Mask one random rectangle of area fraction `p` per view with mid-gray (`fill`)."""
    out = images.clone()
    if p <= 0.0:
        return out
    B, N, _, H, W = out.shape
    target_area = p * H * W
    r_min = max(p * H / W, 0.5)
    r_max = min(H / (p * W), 2.0)
    for b in range(B):
        for v in range(N):
            ratio = float(np.exp(rng.uniform(math.log(r_min), math.log(r_max)))) if r_max > r_min else r_min
            bh = int(np.clip(round(math.sqrt(target_area * ratio)), 1, H))
            bw = int(np.clip(round(target_area / bh), 1, W))
            bh = int(np.clip(round(target_area / bw), 1, H))
            y0 = int(rng.integers(0, H - bh + 1))
            x0 = int(rng.integers(0, W - bw + 1))
            out[b, v, :, y0:y0 + bh, x0:x0 + bw] = fill
    return out


def apply_half_occlusion(images: torch.Tensor, side: str, fill: float = 0.5) -> torch.Tensor:
    """Mask the left, right, top, or bottom half of every view with mid-gray (`fill`)."""
    out = images.clone()
    _, _, _, H, W = out.shape
    if side == "left":
        out[..., :, :W // 2] = fill
    elif side == "right":
        out[..., :, W // 2:] = fill
    elif side == "top":
        out[..., :H // 2, :] = fill
    elif side == "bottom":
        out[..., H // 2:, :] = fill
    else:
        raise ValueError(f"Unknown half-occlusion side {side!r}")
    return out


@torch.no_grad()
def _per_sample_metrics(pred: dict, tgt: dict, surf_gen: torch.Generator) -> dict[str, list[float]]:
    """Compute per-input metrics matching train.evaluate / diagnose_flow."""
    m = tgt["exist"] > 0.5
    s = tgt["s"][:, None]
    exist_pred = torch.sigmoid(pred["exist_logit"]) > 0.5
    rot = geodesic_deg(pred["R"], tgt["R"])
    t_err = (pred["t"] - tgt["t"]).norm(dim=-1) * s
    sc_rel = ((pred["scale"] - tgt["scale"]).abs() / tgt["scale"]).mean(-1)
    ious = occupancy_iou(pred, tgt, exist_pred).tolist()
    exist_acc = (exist_pred == m).float().mean(-1).tolist()
    n_pred = exist_pred.sum(-1).float().tolist()

    out = {k: [] for k in METRIC_KEYS}
    for b in range(m.shape[0]):
        mb = m[b]
        out["rot_deg"].append(float(rot[b][mb].mean()))
        out["t_err_world"].append(float(t_err[b][mb].mean()))
        out["scale_rel_err"].append(float(sc_rel[b][mb].mean()))
        out["iou"].append(float(ious[b]))
        out["exist_acc"].append(float(exist_acc[b]))
        out["n_pred"].append(float(n_pred[b]))
        if exist_pred[b].any():
            p_pts = even_points(pred, b, exist_pred[b], surf_gen)
            g_pts = even_points(tgt, b, mb, surf_gen)
            out["chamfer_world"].append(float(chamfer(p_pts, g_pts) * tgt["s"][b]))
        else:
            out["chamfer_world"].append(float("nan"))
    return out


def _empty_bucket() -> dict:
    return {
        "parallel": {k: [] for k in METRIC_KEYS},
        "flow_mean": {k: [] for k in METRIC_KEYS},
        "flow_best": {k: [] for k in METRIC_KEYS},
        "flow_spread": [],
        "parallel_n_empty": 0,
        "flow_n_empty": 0,
    }


def _summarize_bucket(bkt: dict) -> dict[str, dict[str, float]]:
    par = {k: _nanmean(bkt["parallel"][k]) for k in METRIC_KEYS}
    par["n_empty"] = bkt["parallel_n_empty"]
    fm = {k: _nanmean(bkt["flow_mean"][k]) for k in METRIC_KEYS}
    fm["t_spread_world"] = _nanmean(bkt["flow_spread"])
    fm["n_empty"] = bkt["flow_n_empty"]
    fb = {k: _nanmean(bkt["flow_best"][k]) for k in METRIC_KEYS}
    fb["n_empty"] = sum(1 for v in bkt["flow_best"]["chamfer_world"] if not np.isfinite(v))
    return {"parallel": par, "flow_mean": fm, "flow_best": fb}


def _avg_summaries(items: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    out = {}
    for group in ("parallel", "flow_mean", "flow_best"):
        keys = items[0][group].keys()
        out[group] = {
            k: (sum(it[group][k] for it in items) if k == "n_empty" else _nanmean([it[group][k] for it in items]))
            for k in keys
        }
    return out


@torch.no_grad()
def _eval_batches(
    batches: list[dict],
    occlude_fn,
    vggt: torch.nn.Module,
    dec_parallel: SQDecoder,
    dec_flow: FlowSQDenoiser,
    P: int,
    k_samples: int,
    flow_steps: int,
    seed: int,
    device: str,
) -> dict[str, dict[str, float]]:
    bucket = _empty_bucket()
    surf_gen_p = torch.Generator(device=device).manual_seed(seed)
    surf_gen_f = torch.Generator(device=device).manual_seed(seed)
    noise_gens = [torch.Generator(device=device).manual_seed(seed + k) for k in range(k_samples)]

    for tgt in batches:
        masked_imgs = occlude_fn(tgt["images"])
        tokens = vggt(masked_imgs)

        # 1. Parallel decoder (single deterministic forward pass)
        pred_p = dec_parallel(tokens)
        m_p = _per_sample_metrics(pred_p, tgt, surf_gen_p)
        for k in METRIC_KEYS:
            bucket["parallel"][k].extend(m_p[k])
        bucket["parallel_n_empty"] += sum(1 for v in m_p["chamfer_world"] if not np.isfinite(v))

        # 2. Flow decoder (K Euler draws on the exact same VGGT tokens)
        draw_metrics = []
        draw_t_world = []
        s = tgt["s"][:, None, None]
        for k in range(k_samples):
            x_k = sample_euler(dec_flow, tokens, n_queries=P, n_steps=flow_steps, generator=noise_gens[k])
            pred_k = decode_sq(
                x_k,
                t_mean=dec_flow.t_mean, t_std=dec_flow.t_std,
                shift=dec_flow.flow_shift, scale=dec_flow.flow_scale,
            )
            draw_metrics.append(_per_sample_metrics(pred_k, tgt, surf_gen_f))
            draw_t_world.append(pred_k["t"] * s)

        t_stack = torch.stack(draw_t_world, dim=0)   # (K, B, P, 3)
        gt_mask = tgt["exist"] > 0.5
        B = gt_mask.shape[0]
        for b in range(B):
            best_k = min(range(k_samples), key=lambda kk: draw_metrics[kk]["t_err_world"][b])
            for key in METRIC_KEYS:
                vals_k = [draw_metrics[kk][key][b] for kk in range(k_samples)]
                bucket["flow_mean"][key].append(_nanmean(vals_k))
                bucket["flow_best"][key].append(draw_metrics[best_k][key][b])
            bucket["flow_n_empty"] += sum(1 for kk in range(k_samples) if not np.isfinite(draw_metrics[kk]["chamfer_world"][b]))
            spread_b = float(t_stack[:, b, gt_mask[b], :].std(dim=0, unbiased=False).norm(dim=-1).mean())
            bucket["flow_spread"].append(spread_b)

    return _summarize_bucket(bucket)


def _format_table(title: str, row_label: str, rows: list[tuple[str, str, dict]], k_samples: int = 5) -> str:
    best_hdr = f"flow best-of-{k_samples}"
    lines = [
        f"\n=== {title} ===",
        f"{row_label:<7s} | {'N':<3s} || {'parallel':^15s} || {'flow mean':^15s} || {best_hdr:^15s} || flow spread",
        f"{'':7s} | {'':3s} || {'t_err':>7s} {'chamf':>7s} || {'t_err':>7s} {'chamf':>7s} || {'t_err':>7s} {'chamf':>7s} || {'t_std':>11s}",
        "-" * 88,
    ]
    for param, n_lbl, sm in rows:
        p, fm, fb = sm["parallel"], sm["flow_mean"], sm["flow_best"]
        lines.append(
            f"{param:<7s} | {n_lbl:<3s} || "
            f"{p['t_err_world']:7.4f} {p['chamfer_world']:7.4f} || "
            f"{fm['t_err_world']:7.4f} {fm['chamfer_world']:7.4f} || "
            f"{fb['t_err_world']:7.4f} {fb['chamfer_world']:7.4f} || "
            f"{fm['t_spread_world']:11.4f}"
        )
    return "\n".join(lines)


def _append_csv_rows(csv_rows: list[dict], mode: str, p_val: str, side: str, n_lbl: str, sm: dict,
                     k_samples: int = 5) -> None:
    for model_lbl, stat_lbl, grp_key in [
        ("parallel", "single", "parallel"),
        ("flow", "mean", "flow_mean"),
        ("flow", f"best_of_{k_samples}", "flow_best"),
    ]:
        g = sm[grp_key]
        row = {
            "mode": mode, "p": p_val, "side": side, "N": n_lbl,
            "model": model_lbl, "stat": stat_lbl,
            "t_spread_world": g.get("t_spread_world", float("nan")),
            "n_empty": g["n_empty"],
        }
        for k in METRIC_KEYS:
            row[k] = g[k]
        csv_rows.append(row)


@torch.no_grad()
def evaluate_occlusion(
    vggt: torch.nn.Module,
    dec_parallel: SQDecoder,
    dec_flow: FlowSQDenoiser,
    test_pool: ViewPool,
    gt: GTSuperquadrics,
    geom: GeomTargets,
    P: int,
    batch_size: int,
    per_n: int,
    out_csv: str | Path,
    k_samples: int = 5,
    flow_steps: int = 20,
    seed: int = 1234,
    mask_seed: int = 1234,
    device: str = "cuda",
) -> list[dict]:
    """Run box and half occlusion sweeps and return/save all CSV rows."""
    samples_by_n = eval_samples(test_pool, gt, EVAL_NS, per_n, seed)
    rng_geom = np.random.default_rng(seed)
    batches_by_n: dict[int, list[dict]] = {}
    for n in EVAL_NS:
        smps = samples_by_n[n]
        batches_by_n[n] = [
            collate(test_pool, smps[i:i + batch_size], P, device, geom, rng_geom)
            for i in range(0, len(smps), batch_size)
        ]

    csv_rows: list[dict] = []

    # 1. Mode = box
    box_table_rows: list[tuple[str, str, dict]] = []
    for p in BOX_PS:
        per_n_res = {}
        for n in EVAL_NS:
            mask_rng = np.random.default_rng(mask_seed)
            res = _eval_batches(
                batches_by_n[n],
                lambda imgs, _p=p, _r=mask_rng: apply_box_occlusion(imgs, _p, _r),
                vggt, dec_parallel, dec_flow, P, k_samples, flow_steps, seed, device,
            )
            per_n_res[n] = res
            box_table_rows.append((f"{p:.2f}", str(n), res))
            _append_csv_rows(csv_rows, "box", f"{p}", "", str(n), res, k_samples)
        all_res = _avg_summaries([per_n_res[n] for n in EVAL_NS])
        box_table_rows.append((f"{p:.2f}", "all", all_res))
        _append_csv_rows(csv_rows, "box", f"{p}", "", "all", all_res, k_samples)

    print(_format_table("Mode: box (random rectangle area fraction p, mid-gray)", "p", box_table_rows, k_samples))

    # 2. Mode = half (4 sides averaged)
    half_table_rows: list[tuple[str, str, dict]] = []
    side_res_by_n: dict[int, list[dict]] = {n: [] for n in EVAL_NS}
    for side in HALF_SIDES:
        per_n_res = {}
        for n in EVAL_NS:
            res = _eval_batches(
                batches_by_n[n],
                lambda imgs, _s=side: apply_half_occlusion(imgs, _s),
                vggt, dec_parallel, dec_flow, P, k_samples, flow_steps, seed, device,
            )
            per_n_res[n] = res
            side_res_by_n[n].append(res)
            _append_csv_rows(csv_rows, "half", "0.5", side, str(n), res, k_samples)
        side_all = _avg_summaries([per_n_res[n] for n in EVAL_NS])
        half_table_rows.append((side, "all", side_all))
        _append_csv_rows(csv_rows, "half", "0.5", side, "all", side_all, k_samples)

    avg_by_n = {}
    for n in EVAL_NS:
        avg_n = _avg_summaries(side_res_by_n[n])
        avg_by_n[n] = avg_n
        half_table_rows.append(("avg", str(n), avg_n))
        _append_csv_rows(csv_rows, "half", "0.5", "avg", str(n), avg_n, k_samples)
    avg_all = _avg_summaries([avg_by_n[n] for n in EVAL_NS])
    half_table_rows.append(("avg", "all", avg_all))
    _append_csv_rows(csv_rows, "half", "0.5", "avg", "all", avg_all, k_samples)

    print(_format_table("Mode: half (left/right/top/bottom half hidden, avg over 4 sides)", "side", half_table_rows, k_samples))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "p", "side", "N", "model", "stat", *METRIC_KEYS, "t_spread_world", "n_empty"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved {len(csv_rows)} rows -> {out_path}")
    return csv_rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.eval_occlusion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parallel", default="~/latent2SQ/runs/chair_L1_logscale/decoder.pt",
                   help="path to parallel (regression) decoder.pt")
    p.add_argument("--flow", default="~/rayyan/latent2SQ-v2/runs/flow_tnorm/decoder.pt",
                   help="path to flow-matching decoder.pt")
    p.add_argument("--test", default=None, help="override test split folder (default: from checkpoint args)")
    p.add_argument("--gt-sq", default=None, help="override pseudo-GT .npz (default: from checkpoint args)")
    p.add_argument("--gt-world", default=None, help="override gt_world.npz (default: <test>/../gt_world.npz)")
    p.add_argument("--batch", type=int, default=None, help="override batch size (default: from flow checkpoint)")
    p.add_argument("--per-n", type=int, default=None, help="fixed test samples per N (default: from checkpoint)")
    p.add_argument("--k", type=int, default=5, help="Euler draws per input for flow decoder")
    p.add_argument("--flow-steps", type=int, default=20, help="Euler ODE steps for flow decoder")
    p.add_argument("--seed", type=int, default=1234, help="seed for fixed test batches, surface samples, and flow noise")
    p.add_argument("--mask-seed", type=int, default=1234, help="seed for random occlusion boxes")
    p.add_argument("--out", default="runs/occlusion_eval/results.csv", help="output CSV path")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")

    k1, dec1, args1, ckpt1 = load_decoder(a.parallel, a.device)
    k2, dec2, args2, ckpt2 = load_decoder(a.flow, a.device)
    if {k1, k2} != {"parallel", "flow"}:
        sys.exit(f"Expected one parallel and one flow checkpoint, got {k1!r} and {k2!r}")
    dec_parallel, args_p, ckpt_p = (dec1, args1, ckpt1) if k1 == "parallel" else (dec2, args2, ckpt2)
    dec_flow, args_f, ckpt_f = (dec2, args2, ckpt2) if k2 == "flow" else (dec1, args1, ckpt1)

    if args_p["vggt_layer"] != args_f["vggt_layer"]:
        sys.exit(f"VGGT layer mismatch: parallel={args_p['vggt_layer']}, flow={args_f['vggt_layer']}")
    if args_p["queries"] != args_f["queries"]:
        sys.exit(f"Query count P mismatch: parallel={args_p['queries']}, flow={args_f['queries']}")

    print(f"parallel: {Path(a.parallel).expanduser()} (step {ckpt_p.get('step')}, deform={dec_parallel.deform})",
          file=sys.stderr)
    print(f"flow:     {Path(a.flow).expanduser()} (step {ckpt_f.get('step')}, "
          f"t_mean={dec_flow.t_mean.tolist()}, t_std={dec_flow.t_std.tolist()})", file=sys.stderr)

    test_dir = a.test or args_f.get("test") or args_p["test"]
    gt_sq_path = a.gt_sq or args_f.get("gt_sq") or args_p["gt_sq"]
    gt_world_path = a.gt_world or args_f.get("gt_world") or str(Path(test_dir).parent / "gt_world.npz")
    P = args_f["queries"]
    batch_size = a.batch or args_f.get("batch", 2)
    per_n = a.per_n or args_f.get("eval_per_n", 8)

    test_pool = ViewPool(test_dir)
    gt = GTSuperquadrics(gt_sq_path, deform=dec_parallel.deform)
    geom = GeomTargets(gt_world_path, Path(test_dir) / "cameras.json", seed=0)
    vggt = FrozenVGGT(args_f["vggt_layer"]).to(a.device)

    evaluate_occlusion(
        vggt, dec_parallel, dec_flow, test_pool, gt, geom, P,
        batch_size=batch_size, per_n=per_n, out_csv=a.out,
        k_samples=a.k, flow_steps=a.flow_steps,
        seed=a.seed, mask_seed=a.mask_seed, device=a.device,
    )


if __name__ == "__main__":
    main()
