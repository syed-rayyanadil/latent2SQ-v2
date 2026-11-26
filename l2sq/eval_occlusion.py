"""Evaluate parallel (regression) vs flow-matching SQ decoders under image occlusion.

Run from the repo root (GPU):

    python -m l2sq.eval_occlusion \
        --parallel runs/chair_L1_logscale/decoder.pt \
        --flow runs/flow_geoC100XL/decoder.pt runs/flow_deform8L/decoder.pt \
        --k 8 --out runs/occlusion_eval/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from l2sq.data import GeomTargets, GTSuperquadrics, ViewPool, apply_box_occlusion, collate
from l2sq.flow_matching_decoder import FlowSQDenoiser, decode_sq, sample_euler
from l2sq.geom_loss import occupancy_iou
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.sq import chamfer, geodesic_deg, to_view_frame
from l2sq.train import _nanmean, eval_samples, even_points

EVAL_NS = (1, 4, 8)
BOX_PS = (0.0, 0.25, 0.5, 0.75)
HALF_SIDES = ("left", "right", "top", "bottom")
METRIC_KEYS = ("rot_deg", "t_err_world", "scale_rel_err", "chamfer_world", "chamfer_gt", "iou", "exist_acc", "n_pred")


def _resolve_ckpt_path(ckpt_path: str | Path) -> Path:
    p = Path(ckpt_path).expanduser()
    return (p / "decoder.pt") if p.is_dir() else p


def _model_label(ckpt_path: str | Path, used: set[str]) -> str:
    p = _resolve_ckpt_path(ckpt_path)
    base = p.parent.name if p.name == "decoder.pt" else p.stem
    name = base or "flow"
    idx = 2
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def load_decoder(ckpt_path: str | Path, device: str) -> tuple[str, torch.nn.Module, dict, dict]:
    """Load SQDecoder or FlowSQDenoiser checkpoint, rebuilding architecture from ckpt['args'] like train.py."""
    p = _resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
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
            slot_init="sinus" if args.get("flow_slot_sinus", False) else "small",
            fourier_freqs=args.get("flow_fourier", 0),
            deform=args.get("flow_deform", False),
        )
        dec.load_state_dict(ckpt["decoder"])
        if "t_mean" not in ckpt["decoder"] and "t_mean" in args and "t_std" in args:
            dec.set_t_norm(torch.as_tensor(args["t_mean"]), torch.as_tensor(args["t_std"]))
        if "flow_shift" not in ckpt["decoder"] and "flow_shift" in args and "flow_scale" in args:
            dec.set_flow_norm(torch.as_tensor(args["flow_shift"]), torch.as_tensor(args["flow_scale"]))
    else:
        raise ValueError(f"Unknown decoder type {kind!r} in {p}")
    return kind, dec.to(device).eval(), args, ckpt


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
def _per_sample_metrics(
    pred: dict, tgt: dict, surf_gen: torch.Generator, return_pts: bool = False,
) -> tuple[dict[str, list[float]], list[torch.Tensor | None]]:
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
    has_surf = "surf" in tgt

    out = {k: [] for k in METRIC_KEYS}
    pred_pts_list: list[torch.Tensor | None] = []
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
            out["chamfer_gt"].append(float(chamfer(p_pts, tgt["surf"][b]) * tgt["s"][b]) if has_surf else float("nan"))
            pred_pts_list.append(p_pts if return_pts else None)
        else:
            out["chamfer_world"].append(float("nan"))
            out["chamfer_gt"].append(float("nan"))
            pred_pts_list.append(None)
    return out, pred_pts_list


def _empty_flow_acc() -> dict:
    return {
        "mean": {k: [] for k in METRIC_KEYS},
        "best": {k: [] for k in METRIC_KEYS},
        "chamfer_spread": [],
        "t_spread": [],
        "n_empty": 0,
    }


def _summarize_bucket(par_acc: dict, flow_acc: dict[str, dict]) -> dict:
    par = {k: _nanmean(par_acc[k]) for k in METRIC_KEYS}
    par["n_empty"] = par_acc["n_empty"]
    out = {"parallel": par, "flows": {}}
    for name, acc in flow_acc.items():
        fm = {k: _nanmean(acc["mean"][k]) for k in METRIC_KEYS}
        fm["chamfer_spread_world"] = _nanmean(acc["chamfer_spread"])
        fm["t_spread_world"] = _nanmean(acc["t_spread"])
        fm["n_empty"] = acc["n_empty"]
        fb = {k: _nanmean(acc["best"][k]) for k in METRIC_KEYS}
        fb["n_empty"] = sum(1 for v in acc["best"]["chamfer_world"] if not np.isfinite(v))
        out["flows"][name] = {"mean": fm, "best": fb}
    return out


def _avg_summaries(items: list[dict]) -> dict:
    par_keys = items[0]["parallel"].keys()
    out = {
        "parallel": {
            k: (sum(it["parallel"][k] for it in items) if k == "n_empty"
                else _nanmean([it["parallel"][k] for it in items]))
            for k in par_keys
        },
        "flows": {},
    }
    for name in items[0]["flows"]:
        out["flows"][name] = {}
        for stat_key in ("mean", "best"):
            keys = items[0]["flows"][name][stat_key].keys()
            out["flows"][name][stat_key] = {
                k: (sum(it["flows"][name][stat_key][k] for it in items) if k == "n_empty"
                    else _nanmean([it["flows"][name][stat_key][k] for it in items]))
                for k in keys
            }
    return out


def _retarget_batch(base_tgt: dict, test_pool: ViewPool, chunk: list, gt_obj: GTSuperquadrics, P: int, device: str) -> dict:
    """Copy view-independent targets (images, s, geom) from base_tgt and populate SQ targets from gt_obj."""
    B, M = len(chunk), len(gt_obj)
    if M > P:
        raise ValueError(f"{M} GT superquadrics in {gt_obj} but only P={P} queries")
    out = {k: v for k, v in base_tgt.items() if k in ("images", "s", "surf", "vol", "occ")}
    R_t = torch.zeros(B, P, 3, 3)
    t_t = torch.zeros(B, P, 3)
    sc_t = torch.ones(B, P, 3)
    ep_t = torch.ones(B, P, 2)
    ex_t = torch.zeros(B, P)
    deform = gt_obj.deform
    if deform:
        ta_t = torch.zeros(B, P, 2)
        be_t = torch.zeros(B, P, 3)
        al_t = torch.zeros(B, P, 3)
    for b, smp in enumerate(chunk):
        R, t, scale = to_view_frame(gt_obj.R, gt_obj.t, gt_obj.scale, test_pool.w2c[smp.views[0]], smp.scale_s)
        R_t[b, :M] = torch.as_tensor(R, dtype=torch.float32)
        t_t[b, :M] = torch.as_tensor(t, dtype=torch.float32)
        sc_t[b, :M] = torch.as_tensor(scale, dtype=torch.float32)
        ep_t[b, :M] = torch.as_tensor(gt_obj.eps, dtype=torch.float32)
        ex_t[b, :M] = 1.0
        if deform:
            ta_t[b, :M] = torch.as_tensor(gt_obj.taper, dtype=torch.float32)
            be_t[b, :M] = torch.as_tensor(gt_obj.bend, dtype=torch.float32)
            al_t[b, :M] = torch.as_tensor(gt_obj.alpha, dtype=torch.float32)
    out.update(R=R_t.to(device), t=t_t.to(device), scale=sc_t.to(device), eps=ep_t.to(device), exist=ex_t.to(device))
    if deform:
        out.update(taper=ta_t.to(device), bend=be_t.to(device), alpha=al_t.to(device))
    return out


def _pairwise_chamfer_world(pts_k: list[torch.Tensor | None], scale_s: float) -> float:
    """Mean pairwise Chamfer distance across all pairs of non-empty draws (world units)."""
    valid = [p for p in pts_k if p is not None]
    if len(valid) < 2:
        return float("nan")
    dists = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dists.append(float(chamfer(valid[i], valid[j]) * scale_s))
    return float(np.mean(dists))


@torch.no_grad()
def _eval_batches(
    batch_items: list[tuple[dict, dict, dict[str, dict]]],
    occlude_fn,
    vggt: torch.nn.Module,
    dec_parallel: SQDecoder,
    flow_models: dict[str, FlowSQDenoiser],
    flow_steps_by_model: dict[str, int],
    P: int,
    k_samples: int,
    seed: int,
    device: str,
) -> dict:
    par_acc = {k: [] for k in METRIC_KEYS}
    par_acc["n_empty"] = 0
    flow_acc = {name: _empty_flow_acc() for name in flow_models}
    surf_gen_p = torch.Generator(device=device).manual_seed(seed)
    surf_gen_f = {name: torch.Generator(device=device).manual_seed(seed) for name in flow_models}

    for base_tgt, par_tgt, flow_tgts in batch_items:
        masked_imgs = occlude_fn(base_tgt["images"])
        tokens = vggt(masked_imgs)

        # 1. Parallel baseline (single deterministic forward pass)
        pred_p = dec_parallel(tokens)
        m_p, _ = _per_sample_metrics(pred_p, par_tgt, surf_gen_p)
        for k in METRIC_KEYS:
            par_acc[k].extend(m_p[k])
        par_acc["n_empty"] += sum(1 for v in m_p["chamfer_world"] if not np.isfinite(v))

        # 2. Flow models (K Euler draws on the exact same VGGT tokens)
        for name, dec_flow in flow_models.items():
            noise_gens = [torch.Generator(device=device).manual_seed(seed + k) for k in range(k_samples)]
            tgt_m = flow_tgts[name]
            n_steps = flow_steps_by_model[name]
            draw_metrics = []
            draw_pts: list[list[torch.Tensor | None]] = []
            draw_t_world = []
            s = tgt_m["s"][:, None, None]
            for kk in range(k_samples):
                x_k = sample_euler(dec_flow, tokens, n_queries=P, n_steps=n_steps, generator=noise_gens[kk])
                pred_k = decode_sq(
                    x_k,
                    t_mean=dec_flow.t_mean, t_std=dec_flow.t_std,
                    shift=dec_flow.flow_shift, scale=dec_flow.flow_scale,
                    deform=dec_flow.deform,
                )
                m_k, pts_k = _per_sample_metrics(pred_k, tgt_m, surf_gen_f[name], return_pts=True)
                draw_metrics.append(m_k)
                draw_pts.append(pts_k)
                draw_t_world.append(pred_k["t"] * s)

            t_stack = torch.stack(draw_t_world, dim=0)   # (K, B, P, 3)
            gt_mask = tgt_m["exist"] > 0.5
            B = gt_mask.shape[0]
            acc = flow_acc[name]
            for b in range(B):
                if any(np.isfinite(draw_metrics[kk]["chamfer_world"][b]) for kk in range(k_samples)):
                    best_k = min(
                        range(k_samples),
                        key=lambda kk: draw_metrics[kk]["chamfer_world"][b]
                        if np.isfinite(draw_metrics[kk]["chamfer_world"][b]) else float("inf"),
                    )
                else:
                    best_k = min(range(k_samples), key=lambda kk: draw_metrics[kk]["t_err_world"][b])
                for key in METRIC_KEYS:
                    vals_k = [draw_metrics[kk][key][b] for kk in range(k_samples)]
                    acc["mean"][key].append(_nanmean(vals_k))
                    acc["best"][key].append(draw_metrics[best_k][key][b])
                acc["n_empty"] += sum(1 for kk in range(k_samples) if not np.isfinite(draw_metrics[kk]["chamfer_world"][b]))
                acc["chamfer_spread"].append(_pairwise_chamfer_world([draw_pts[kk][b] for kk in range(k_samples)],
                                                                     float(tgt_m["s"][b])))
                spread_b = float(t_stack[:, b, gt_mask[b], :].std(dim=0, unbiased=False).norm(dim=-1).mean())
                acc["t_spread"].append(spread_b)

    return _summarize_bucket(par_acc, flow_acc)


def _format_table(title: str, row_label: str, rows: list[tuple[str, str, dict]], k_samples: int = 8) -> str:
    flow_names = list(rows[0][2]["flows"].keys())
    col_w = max(15, *(len(n) for n in flow_names))
    best_hdr = f"best-{k_samples} cd"
    lines = [
        f"\n=== {title} ===",
        f"{row_label:<7s} | {'N':<3s} | {'model':<{col_w}s} || "
        f"{'rot':>5s} {'t_err':>7s} {'chamf':>7s} {'c_gt':>7s} {'iou':>6s} || {best_hdr:>9s} || {'pair_cd':>7s} {'t_std':>7s}",
        "-" * (69 + col_w),
    ]
    for param, n_lbl, sm in rows:
        p = sm["parallel"]
        lines.append(
            f"{param:<7s} | {n_lbl:<3s} | {'parallel':<{col_w}s} || "
            f"{p['rot_deg']:5.1f} {p['t_err_world']:7.4f} {p['chamfer_world']:7.4f} {p['chamfer_gt']:7.4f} {p['iou']:6.3f} || "
            f"{'---':>9s} || {'---':>7s} {'---':>7s}"
        )
        for name in flow_names:
            fm, fb = sm["flows"][name]["mean"], sm["flows"][name]["best"]
            lines.append(
                f"{'':7s} | {'':3s} | {name:<{col_w}s} || "
                f"{fm['rot_deg']:5.1f} {fm['t_err_world']:7.4f} {fm['chamfer_world']:7.4f} {fm['chamfer_gt']:7.4f} {fm['iou']:6.3f} || "
                f"{fb['chamfer_world']:9.4f} || {fm['chamfer_spread_world']:7.4f} {fm['t_spread_world']:7.4f}"
            )
    return "\n".join(lines)


def _append_csv_rows(csv_rows: list[dict], mode: str, p_val: str, side: str, n_lbl: str, sm: dict,
                     k_samples: int = 8) -> None:
    par = sm["parallel"]
    par_row = {
        "mode": mode, "p": p_val, "side": side, "N": n_lbl,
        "model": "parallel", "stat": "single",
        "chamfer_spread_world": float("nan"), "t_spread_world": float("nan"), "n_empty": par["n_empty"],
    }
    for k in METRIC_KEYS:
        par_row[k] = par[k]
    csv_rows.append(par_row)

    for name, f_res in sm["flows"].items():
        for stat_lbl, grp_key in [("mean", "mean"), (f"best_of_{k_samples}", "best")]:
            g = f_res[grp_key]
            row = {
                "mode": mode, "p": p_val, "side": side, "N": n_lbl,
                "model": name, "stat": stat_lbl,
                "chamfer_spread_world": g.get("chamfer_spread_world", float("nan")),
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
    dec_flow: FlowSQDenoiser | None = None,
    test_pool: ViewPool = None,
    gt: GTSuperquadrics | None = None,
    geom: GeomTargets = None,
    P: int = 16,
    batch_size: int = 2,
    per_n: int = 8,
    out_csv: str | Path = "runs/occlusion_eval/results.csv",
    k_samples: int = 8,
    flow_steps: int = 20,
    seed: int = 1234,
    mask_seed: int = 1234,
    device: str = "cuda",
    flow_models: dict[str, FlowSQDenoiser] | None = None,
    parallel_gt: GTSuperquadrics | None = None,
    flow_gts: dict[str, GTSuperquadrics] | None = None,
    flow_steps_by_model: dict[str, int] | None = None,
) -> list[dict]:
    """Run box and half occlusion sweeps and return/save all CSV rows."""
    if flow_models is None:
        if dec_flow is None:
            raise ValueError("Provide either flow_models or dec_flow")
        flow_models = {"flow": dec_flow}
    par_gt = parallel_gt if parallel_gt is not None else gt
    f_gts = flow_gts if flow_gts is not None else {name: gt for name in flow_models}
    f_steps = flow_steps_by_model if flow_steps_by_model is not None else {name: flow_steps for name in flow_models}
    base_gt = par_gt if par_gt is not None else next(iter(f_gts.values()))

    samples_by_n = eval_samples(test_pool, base_gt, EVAL_NS, per_n, seed)
    rng_geom = np.random.default_rng(seed)
    batches_by_n: dict[int, list[tuple[dict, dict, dict[str, dict]]]] = {}
    for n in EVAL_NS:
        smps = samples_by_n[n]
        items = []
        for i in range(0, len(smps), batch_size):
            chunk = smps[i:i + batch_size]
            base_tgt = collate(test_pool, chunk, P, device, geom, rng_geom)
            par_tgt = base_tgt if par_gt is base_gt else _retarget_batch(base_tgt, test_pool, chunk, par_gt, P, device)
            model_tgts = {
                name: (base_tgt if f_gts[name] is base_gt else _retarget_batch(base_tgt, test_pool, chunk, f_gts[name], P, device))
                for name in flow_models
            }
            items.append((base_tgt, par_tgt, model_tgts))
        batches_by_n[n] = items

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
                vggt, dec_parallel, flow_models, f_steps, P, k_samples, seed, device,
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
                vggt, dec_parallel, flow_models, f_steps, P, k_samples, seed, device,
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
    fieldnames = ["mode", "p", "side", "N", "model", "stat", *METRIC_KEYS,
                  "chamfer_spread_world", "t_spread_world", "n_empty"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved {len(csv_rows)} rows -> {out_path}")
    return csv_rows


def _resolve_gt_sq_path(ckpt_args: dict, test_dir: str, deform: bool, override: str | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    cand = ckpt_args.get("gt_sq")
    if cand and Path(cand).expanduser().exists():
        return Path(cand).expanduser()
    fallback = Path(test_dir).parent / ("gt_sq_deform.npz" if deform else "gt_sq_rigid.npz")
    if fallback.exists():
        return fallback
    if cand:
        return Path(cand).expanduser()
    raise FileNotFoundError(f"Could not find gt_sq for deform={deform} near {test_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.eval_occlusion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parallel", required=True, help="path to parallel (regression) decoder.pt or run dir")
    p.add_argument("--flow", nargs="+", default=["runs/flow_geoC100XL/decoder.pt", "runs/flow_deform8L/decoder.pt"],
                   help="one or more flow-matching decoder.pt paths or run dirs")
    p.add_argument("--test", default=None, help="override test split folder (default: from checkpoint args)")
    p.add_argument("--gt-sq", default=None, help="override pseudo-GT .npz for all models (default: per checkpoint args)")
    p.add_argument("--gt-world", default=None, help="override gt_world.npz (default: <test>/../gt_world.npz)")
    p.add_argument("--batch", type=int, default=None, help="override batch size (default: from flow checkpoint)")
    p.add_argument("--per-n", type=int, default=None, help="fixed test samples per N (default: from checkpoint)")
    p.add_argument("--k", type=int, default=8, help="Euler draws per input for flow decoder (default 8)")
    p.add_argument("--flow-steps", type=int, default=None, help="override Euler ODE steps (default: per checkpoint)")
    p.add_argument("--seed", type=int, default=1234, help="seed for fixed test batches, surface samples, and flow noise")
    p.add_argument("--mask-seed", type=int, default=1234, help="seed for random occlusion boxes")
    p.add_argument("--out", default="runs/occlusion_eval/results.csv", help="output CSV path")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available -- run this on a GPU machine or via sbatch.")

    k_p, dec_parallel, args_p, ckpt_p = load_decoder(a.parallel, a.device)
    if k_p != "parallel":
        sys.exit(f"Expected a parallel checkpoint for --parallel, got {k_p!r}")

    used_names: set[str] = {"parallel"}
    flow_models: dict[str, FlowSQDenoiser] = {}
    flow_args: dict[str, dict] = {}
    flow_ckpts: dict[str, dict] = {}
    flow_steps_by_model: dict[str, int] = {}
    for fp in a.flow:
        kf, df, af, cf = load_decoder(fp, a.device)
        if kf != "flow":
            sys.exit(f"Expected a flow checkpoint in --flow, got {kf!r} ({fp})")
        if af["vggt_layer"] != args_p["vggt_layer"]:
            sys.exit(f"VGGT layer mismatch: parallel={args_p['vggt_layer']}, {fp}={af['vggt_layer']}")
        if af["queries"] != args_p["queries"]:
            sys.exit(f"Query count P mismatch: parallel={args_p['queries']}, {fp}={af['queries']}")
        name = _model_label(fp, used_names)
        flow_models[name] = df
        flow_args[name] = af
        flow_ckpts[name] = cf
        flow_steps_by_model[name] = a.flow_steps if a.flow_steps is not None else af.get("flow_steps", 20)

    print(f"parallel: {_resolve_ckpt_path(a.parallel)} (step {ckpt_p.get('step')}, deform={dec_parallel.deform})",
          file=sys.stderr)
    for name, df in flow_models.items():
        af, cf = flow_args[name], flow_ckpts[name]
        print(f"flow [{name}]: {_resolve_ckpt_path(a.flow[list(flow_models).index(name)])} "
              f"(step {cf.get('step')}, sq_dim={df.sq_dim}, deform={df.deform}, "
              f"slot_sinus={af.get('flow_slot_sinus', False)}, fourier={af.get('flow_fourier', 0)}, "
              f"geo={af.get('flow_geo', 'none')}, steps={flow_steps_by_model[name]})", file=sys.stderr)

    first_f_args = next(iter(flow_args.values()))
    test_dir = a.test
    if test_dir is None:
        for cand_args in [first_f_args, *flow_args.values(), args_p]:
            t_cand = cand_args.get("test")
            if t_cand and Path(t_cand).expanduser().exists():
                test_dir = str(Path(t_cand).expanduser())
                break
        if test_dir is None:
            test_dir = first_f_args.get("test") or args_p["test"]

    gt_world_path = a.gt_world
    if gt_world_path is None:
        w_cand = first_f_args.get("gt_world")
        gt_world_path = str(Path(w_cand).expanduser()) if (w_cand and Path(w_cand).expanduser().exists()) else str(Path(test_dir).parent / "gt_world.npz")

    P = args_p["queries"]
    batch_size = a.batch or first_f_args.get("batch", 2)
    per_n = a.per_n or first_f_args.get("eval_per_n", 8)

    test_pool = ViewPool(test_dir)
    gt_cache: dict[tuple[str, bool], GTSuperquadrics] = {}

    def _get_gt(path: Path, deform: bool) -> GTSuperquadrics:
        key = (str(path), deform)
        if key not in gt_cache:
            gt_cache[key] = GTSuperquadrics(path, deform=deform)
        return gt_cache[key]

    any_deform = dec_parallel.deform or any(df.deform for df in flow_models.values())
    par_gt_path = _resolve_gt_sq_path(args_p, test_dir, dec_parallel.deform, a.gt_sq)
    parallel_gt = _get_gt(par_gt_path, dec_parallel.deform or (a.gt_sq is not None and any_deform))
    flow_gts = {
        name: _get_gt(_resolve_gt_sq_path(flow_args[name], test_dir, df.deform, a.gt_sq),
                      df.deform or (a.gt_sq is not None and any_deform))
        for name, df in flow_models.items()
    }
    geom = GeomTargets(gt_world_path, Path(test_dir) / "cameras.json", seed=0)
    vggt = FrozenVGGT(args_p["vggt_layer"]).to(a.device)

    evaluate_occlusion(
        vggt, dec_parallel, test_pool=test_pool, geom=geom, P=P,
        batch_size=batch_size, per_n=per_n, out_csv=a.out,
        k_samples=a.k, seed=a.seed, mask_seed=a.mask_seed, device=a.device,
        flow_models=flow_models, parallel_gt=parallel_gt, flow_gts=flow_gts,
        flow_steps_by_model=flow_steps_by_model,
    )


if __name__ == "__main__":
    main()
