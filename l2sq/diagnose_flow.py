"""Diagnose a trained flow-matching SQ decoder (sampling steps, teacher forcing, 1-step x1, conditioning, raw ranges).

Run from the repo root (GPU):

    python -m l2sq.diagnose_flow runs/flow_3000/decoder.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from l2sq.data import GTSuperquadrics, ViewPool, collate
from l2sq.flow_matching_decoder import (
    GROUP_SLICES,
    FlowSQDenoiser,
    _shift_scale,
    decode_sq,
    pack_sq,
    sample_euler,
)
from l2sq.model import FrozenVGGT
from l2sq.sq import geodesic_deg
from l2sq.train import eval_samples


def _metrics(pred: dict, tgt: dict) -> dict[str, list[float]]:
    m = tgt["exist"] > 0.5
    s = tgt["s"][:, None]
    exist_pred = torch.sigmoid(pred["exist_logit"]) > 0.5
    return {
        "rot_deg": geodesic_deg(pred["R"], tgt["R"])[m].tolist(),
        "t_err_world": ((pred["t"] - tgt["t"]).norm(dim=-1) * s)[m].tolist(),
        "scale_rel_err": ((pred["scale"] - tgt["scale"]).abs() / tgt["scale"]).mean(-1)[m].tolist(),
        "exist_acc": (exist_pred == m).float().mean(-1).tolist(),
        "n_pred": exist_pred.sum(-1).float().tolist(),
    }


def _summarize(rows: list[dict[str, list[float]]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {k: float(np.nanmean([v for r in rows for v in r[k]])) for k in keys}


def _fmt(s: dict[str, float]) -> str:
    return (f"rot {s['rot_deg']:6.1f}deg  t {s['t_err_world']:.4f}  "
            f"scale {100 * s['scale_rel_err']:6.1f}%  exist-acc {s['exist_acc']:.3f}  #SQ {s['n_pred']:4.1f}")


@torch.no_grad()
def _euler_from_t(dec: FlowSQDenoiser, tokens: torch.Tensor, x: torch.Tensor, t0: float,
                  total_steps: int = 20) -> torch.Tensor:
    """Run the same Euler update starting from state x at time t0 up to t=1."""
    mem, n_views = dec.encode_tokens(tokens)
    steps = max(1, round((1.0 - t0) * total_steps))
    dt = (1.0 - t0) / steps
    for k in range(steps):
        t = torch.full((tokens.shape[0],), t0 + k * dt, device=tokens.device, dtype=tokens.dtype)
        x = x + dt * dec(x, t, mem=mem, n_views=n_views)
    return x


def _unnorm(x: torch.Tensor, t_mean: torch.Tensor | None = None, t_std: torch.Tensor | None = None,
            shift: torch.Tensor | None = None, scale: torch.Tensor | None = None) -> torch.Tensor:
    sh, sc = _shift_scale(x, shift, scale)
    out = x * sc + sh
    if t_mean is not None and t_std is not None:
        t_real = out[..., 0:3] * t_std.to(out).clamp_min(1e-6) + t_mean.to(out)
        out = torch.cat([t_real, out[..., 3:]], dim=-1)
    return out


def _minmax(t: torch.Tensor) -> tuple[float, float]:
    return float(t.min()), float(t.max())


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.diagnose_flow", description=__doc__)
    p.add_argument("ckpt", nargs="?", default="runs/flow_3000/decoder.pt", help="path to decoder.pt")
    p.add_argument("--seed", type=int, default=1234, help="seed for fixed test batches and eval noise")
    p.add_argument("--batch", type=int, default=None, help="override batch size (default: from checkpoint)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    ckpt = torch.load(a.ckpt, map_location="cpu")
    args = ckpt["args"]
    P = args["queries"]
    batch_size = a.batch or args.get("batch", 2)
    per_n = args.get("eval_per_n", 8)

    dec = FlowSQDenoiser(args["d_model"], P, args["layers"], args["heads"], args["ffn"]).to(a.device).eval()
    dec.load_state_dict(ckpt["decoder"])
    vggt = FrozenVGGT(args["vggt_layer"]).to(a.device)

    test_pool = ViewPool(args["test"])
    gt = GTSuperquadrics(args["gt_sq"], deform=False)
    samples_by_n = eval_samples(test_pool, gt, (4, 8), per_n, a.seed)

    # Precompute VGGT tokens once per fixed test batch.
    cached: dict[int, list[tuple[dict, torch.Tensor]]] = {}
    for n, smps in samples_by_n.items():
        cached[n] = []
        for i in range(0, len(smps), batch_size):
            tgt = collate(test_pool, smps[i:i + batch_size], P, a.device)
            cached[n].append((tgt, vggt(tgt["images"])))
    print(f"Loaded {a.ckpt} (step {ckpt.get('step')}), test N in (4, 8), {per_n} samples/N, batch={batch_size}")

    # 1. Sampling steps: 5, 20, 100
    print("\n1. Sampling steps (Euler from N(0,1), seed=1234):")
    samples_20: dict[int, list[tuple[dict, torch.Tensor, torch.Tensor]]] = {}
    for n_steps in (5, 20, 100):
        gen = torch.Generator(device=a.device).manual_seed(a.seed)
        all_rows = []
        for n in (4, 8):
            rows = []
            for tgt, tokens in cached[n]:
                x_smp = sample_euler(dec, tokens, n_queries=P, n_steps=n_steps, generator=gen)
                pred = decode_sq(x_smp, t_mean=dec.t_mean, t_std=dec.t_std, shift=dec.flow_shift, scale=dec.flow_scale)
                rows.append(_metrics(pred, tgt))
                if n_steps == 20:
                    samples_20.setdefault(n, []).append((tgt, x_smp, pred))
            all_rows.extend(rows)
            print(f"   steps={n_steps:3d} N={n}: {_fmt(_summarize(rows))}")
        print(f"   steps={n_steps:3d} all: {_fmt(_summarize(all_rows))}")

    # 2 & 3. Teacher-forcing ODE from t0 -> 1 and direct 1-step x1_hat = x_t + (1 - t0) * v_hat
    print("\n2. Teacher-forcing Euler (t0 -> 1) and 3. One-step x1_hat:")
    for t0 in (0.5, 0.8, 0.95):
        gen = torch.Generator(device=a.device).manual_seed(a.seed)
        tf_rows, one_rows = [], []
        for n in (4, 8):
            for tgt, tokens in cached[n]:
                x_1 = pack_sq(tgt, t_mean=dec.t_mean, t_std=dec.t_std, shift=dec.flow_shift, scale=dec.flow_scale)
                x_0 = torch.randn(x_1.shape, device=x_1.device, dtype=x_1.dtype, generator=gen)
                x_t = t0 * x_1 + (1.0 - t0) * x_0

                x_tf = _euler_from_t(dec, tokens, x_t, t0, total_steps=20)
                tf_rows.append(_metrics(decode_sq(x_tf, t_mean=dec.t_mean, t_std=dec.t_std,
                                                  shift=dec.flow_shift, scale=dec.flow_scale), tgt))

                t_vec = torch.full((x_t.shape[0],), t0, device=x_t.device, dtype=x_t.dtype)
                v_hat = dec(x_t, t_vec, tokens=tokens)
                x1_hat = x_t + (1.0 - t0) * v_hat
                one_rows.append(_metrics(decode_sq(x1_hat, t_mean=dec.t_mean, t_std=dec.t_std,
                                                   shift=dec.flow_shift, scale=dec.flow_scale), tgt))
        print(f"   t0={t0:.2f} [teacher-forced ODE]: {_fmt(_summarize(tf_rows))}")
        print(f"   t0={t0:.2f} [1-step x1_hat    ]: {_fmt(_summarize(one_rows))}")

    # 4. Conditioning check: shuffle VGGT tokens within batch (20 steps, same noise seed)
    print("\n4. Conditioning check (shuffled VGGT tokens vs correct tokens, 20 steps):")
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    dt_world, dt_flow, dexist, flips = [], [], [], []
    for n in (4, 8):
        for idx, (tgt, tokens) in enumerate(cached[n]):
            if tokens.shape[0] >= 2:
                tok_shuf = torch.roll(tokens, shifts=1, dims=0)
            else:
                other = cached[n][(idx + 1) % len(cached[n])][1]
                tok_shuf = other[:1]
            x_shuf = sample_euler(dec, tok_shuf, n_queries=P, n_steps=20, generator=gen)
            pred_shuf = decode_sq(x_shuf, t_mean=dec.t_mean, t_std=dec.t_std, shift=dec.flow_shift, scale=dec.flow_scale)
            _, x_corr, pred_corr = samples_20[n][idx]

            s = tgt["s"][:, None]
            dt_world.append(float(((pred_corr["t"] - pred_shuf["t"]).norm(dim=-1) * s).mean()))
            dt_flow.append(float((_unnorm(x_corr, dec.t_mean, dec.t_std, dec.flow_shift, dec.flow_scale)[..., :3] -
                                  _unnorm(x_shuf, dec.t_mean, dec.t_std, dec.flow_shift, dec.flow_scale)[..., :3]).abs().mean()))
            p_corr = torch.sigmoid(pred_corr["exist_logit"])
            p_shuf = torch.sigmoid(pred_shuf["exist_logit"])
            dexist.append(float((p_corr - p_shuf).abs().mean()))
            flips.append(float(((p_corr > 0.5) != (p_shuf > 0.5)).float().mean()))
    mean_dt, mean_de = float(np.mean(dt_world)), float(np.mean(dexist))
    print(f"   mean |t_corr - t_shuf| (world): {mean_dt:.4f}  (raw flow |dt|: {np.mean(dt_flow):.4f})")
    print(f"   mean |exist_corr - exist_shuf|: {mean_de:.4f}  (exist>0.5 flip rate: {100 * np.mean(flips):.1f}%)")
    if mean_dt < 0.02 and mean_de < 0.02:
        print("   NOTE: outputs barely change with shuffled VGGT tokens (conditioning is very weak).")

    # 5. Raw flow-space ranges (min/max) of sampled x_1 vs GT x_1
    print("\n5. Raw flow-space ranges (min, max) — GT vs Sampled (20 steps):")
    gt_raw = torch.cat([pack_sq(tgt, normalize=False) for n in (4, 8) for tgt, _ in cached[n]], dim=0)
    gt_mask = torch.cat([tgt["exist"] > 0.5 for n in (4, 8) for tgt, _ in cached[n]], dim=0)
    smp_raw = torch.cat([_unnorm(x_smp, dec.t_mean, dec.t_std, dec.flow_shift, dec.flow_scale)
                         for n in (4, 8) for _, x_smp, _ in samples_20[n]], dim=0)
    smp_mask = smp_raw[..., 14] > 0.5
    for name in ("t", "scale", "eps", "exist"):
        sl = GROUP_SLICES[name]
        g_vals = gt_raw[..., sl] if name == "exist" else gt_raw[..., sl][gt_mask]
        s_all = smp_raw[..., sl]
        s_act = s_all if (name == "exist" or not smp_mask.any()) else smp_raw[..., sl][smp_mask]
        label = "log-scale" if name == "scale" else name
        g_lo, g_hi = _minmax(g_vals)
        a_lo, a_hi = _minmax(s_all)
        k_lo, k_hi = _minmax(s_act)
        print(f"   {label:9s}  GT [{g_lo:7.3f}, {g_hi:7.3f}]  |  "
              f"sampled(all) [{a_lo:7.3f}, {a_hi:7.3f}]  |  sampled(exist>0.5) [{k_lo:7.3f}, {k_hi:7.3f}]")

    # 6. Translation (t) breakdown: per-slot table, mean offset, centered error, Hungarian matching, std
    print("\n6. Translation (t) diagnosis in world-scale units (20-step samples vs GT):")

    def _t_stats(t_gt: torch.Tensor, t_pred: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> dict:
        g_act, p_on_gt = t_gt[gt_mask], t_pred[gt_mask]
        diff = p_on_gt - g_act
        mean_diff = diff.mean(dim=0)
        err_idx = float(diff.norm(dim=-1).mean())
        err_cent = float((diff - mean_diff).norm(dim=-1).mean())
        p_act = t_pred[pred_mask]
        if len(p_act) > 0 and len(g_act) > 0:
            cost = torch.cdist(g_act, p_act).cpu().numpy()
            ri, ci = linear_sum_assignment(cost)
            err_hung = float(cost[ri, ci].mean())
        else:
            err_hung = float("nan")
        gt_std = g_act.std(dim=0, unbiased=False)
        pred_std = p_act.std(dim=0, unbiased=False) if len(p_act) > 1 else torch.zeros(3, device=t_pred.device)
        return {
            "diff": diff, "mean_diff": mean_diff, "mean_norm": float(mean_diff.norm()),
            "err_idx": err_idx, "err_cent": err_cent, "err_hung": err_hung,
            "gt_std": gt_std, "pred_std": pred_std, "n_pred_act": int(pred_mask.sum()),
        }

    def _print_sample_t(label: str, tgt: dict, pred: dict, b: int) -> None:
        s = float(tgt["s"][b])
        t_gt, t_pred = tgt["t"][b] * s, pred["t"][b] * s
        gt_mask = tgt["exist"][b] > 0.5
        pred_mask = torch.sigmoid(pred["exist_logit"][b]) > 0.5
        st = _t_stats(t_gt, t_pred, gt_mask, pred_mask)
        print(f"   [{label}] active GT={int(gt_mask.sum())}, pred={st['n_pred_act']}")
        print("   slot |         GT t (x, y, z)         |        pred t (x, y, z)        |     diff (pred - GT)")
        for slot in torch.nonzero(gt_mask, as_tuple=True)[0].tolist():
            g, pr, d = t_gt[slot], t_pred[slot], t_pred[slot] - t_gt[slot]
            print(f"   {slot:4d} | ({g[0]:6.3f}, {g[1]:6.3f}, {g[2]:6.3f}) | "
                  f"({pr[0]:6.3f}, {pr[1]:6.3f}, {pr[2]:6.3f}) | ({d[0]:+6.3f}, {d[1]:+6.3f}, {d[2]:+6.3f})")
        md, gs, ps = st["mean_diff"], st["gt_std"], st["pred_std"]
        print(f"   mean(pred-GT)=({md[0]:+.3f}, {md[1]:+.3f}, {md[2]:+.3f}) norm={st['mean_norm']:.4f} | "
              f"err: idx={st['err_idx']:.4f}, centered={st['err_cent']:.4f}, hung={st['err_hung']:.4f}")
        print(f"   std(t) GT=({gs[0]:.3f}, {gs[1]:.3f}, {gs[2]:.3f})  "
              f"pred=({ps[0]:.3f}, {ps[1]:.3f}, {ps[2]:.3f})")

    n8_flat = [(tgt, pred, b) for tgt, _, pred in samples_20[8] for b in range(tgt["t"].shape[0])]
    _print_sample_t("N=8 sample 0", *n8_flat[0])
    if len(n8_flat) > 1:
        _print_sample_t("N=8 sample 1 (diff views)", *n8_flat[1])

    all_st = [
        _t_stats(tgt["t"][b] * float(tgt["s"][b]), pred["t"][b] * float(tgt["s"][b]),
                 tgt["exist"][b] > 0.5, torch.sigmoid(pred["exist_logit"][b]) > 0.5)
        for n in (4, 8) for tgt, _, pred in samples_20[n] for b in range(tgt["t"].shape[0])
    ]
    print(f"   [all N=4,8 test set ({len(all_st)} samples)] mean_offset_norm={np.mean([r['mean_norm'] for r in all_st]):.4f} | "
          f"err: idx={np.mean([r['err_idx'] for r in all_st]):.4f}, "
          f"centered={np.mean([r['err_cent'] for r in all_st]):.4f}, "
          f"hung={np.nanmean([r['err_hung'] for r in all_st]):.4f}")


if __name__ == "__main__":
    main()
