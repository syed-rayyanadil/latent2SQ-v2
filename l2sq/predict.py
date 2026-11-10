"""Run a trained decoder on views 0..N-1 of a split and export the superquadrics for the viewer.

Run from the repo root:

    python -m l2sq.predict runs/chair_L1_logscale dataset/chair/test -n 8 24

View 0 is the reference, so the prediction is in that split's camera-0 frame (metric) and lines
up with the split viewer (GT points, aligned VGGT cloud, pseudo-GT superquadrics):

    python -m recon.view dataset/chair/test --mesh runs/chair_L1_logscale/predict/test_N8_0/pred_sq.ply

Writes <run>/predict/<split>_N<n>_0/{pred_sq.ply, gt_sq.ply, sample.json, metrics.json}.
Training used N <= --n-max (8 by default); larger N is outside what the decoder has seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from l2sq.data import GTSuperquadrics, ViewPool, make_sample
from l2sq.model import FrozenVGGT, SQDecoder
from l2sq.train import evaluate, export_examples


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m l2sq.predict", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="training run folder with decoder.pt, e.g. runs/chair_L1_logscale")
    p.add_argument("split", help="split folder, e.g. dataset/chair/test")
    p.add_argument("-n", type=int, nargs="+", default=[8], help="numbers of views (first n of the split)")
    p.add_argument("--gt-sq", help="pseudo-GT superquadrics (default: the run's --gt-sq)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    run, split = Path(a.run), Path(a.split)
    ckpt = torch.load(run / "decoder.pt", map_location="cpu")
    args = ckpt["args"]
    dec = SQDecoder(args["d_model"], args["queries"], args["layers"], args["heads"], args["ffn"])
    dec.load_state_dict(ckpt["decoder"])
    dec.to(a.device).eval()
    vggt = FrozenVGGT(args["vggt_layer"]).to(a.device)

    pool = ViewPool(split)
    gt = GTSuperquadrics(a.gt_sq or args["gt_sq"])
    w = {"t": args["w_t"], "R": args["w_R"], "scale": args["w_scale"], "eps": args["w_eps"], "exist": args["w_exist"]}
    out_dir = run / "predict"
    print(f"decoder from {run} (step {ckpt['step']}), trained on N in [{args['n_min']}, {args['n_max']}]",
          file=sys.stderr)

    for n in a.n:
        if n > len(pool):
            sys.exit(f"-n {n} > {len(pool)} views in {split}")
        smp = make_sample(pool, gt, list(range(n)))
        export_examples(vggt, dec, pool, [smp], split.name, args["queries"], a.device, out_dir)
        r = evaluate(vggt, dec, pool, {n: [smp]}, args["queries"], 1, a.device, w,
                     args.get("scale_loss", "linear"))[f"N{n}"]
        d = out_dir / f"{split.name}_N{n}_0"
        (d / "metrics.json").write_text(json.dumps(r, indent=2))
        peak = torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else 0.0
        print(f"N={n:2d}: rot {r['rot_deg']:.1f}deg  t {r['t_err_world']:.4f}  scale {100 * r['scale_rel_err']:.1f}%  "
              f"chamfer {r['chamfer_world']:.4f}  #SQ {r['n_pred']:.0f}  (peak GPU {peak:.1f} GB)\n"
              f"  view: python -m recon.view {split} --mesh {d / 'pred_sq.ply'}", file=sys.stderr)


if __name__ == "__main__":
    main()
