"""CPU-only tests for FlowSQDenoiser and train.py --decoder flow (no GPU, no VGGT)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from l2sq.flow_matching_decoder import FlowSQDenoiser, decode_sq, flow_group_stats, flow_loss, pack_sq, sample_euler
from l2sq.model import TOKEN_DIM
from l2sq.sq import EPS_MAX, EPS_MIN, SCALE_MIN, rot6d_to_matrix


def _fake_tgt(B: int = 2, P: int = 16, M: int = 7, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    R = rot6d_to_matrix(torch.randn(B, P, 6, generator=g))
    t = torch.randn(B, P, 3, generator=g) * 0.4 + torch.tensor([0.0, 0.0, 1.0])
    scale = torch.rand(B, P, 3, generator=g) * 0.35 + 0.03
    eps = torch.rand(B, P, 2, generator=g) * 1.4 + 0.25
    exist = torch.zeros(B, P)
    exist[:, :M] = 1.0
    return {"t": t, "R": R, "scale": scale, "eps": eps, "exist": exist}


def _assert_valid_sq(sq: dict, B: int, P: int) -> None:
    assert sq["t"].shape == (B, P, 3) and torch.isfinite(sq["t"]).all()
    eye = torch.eye(3).expand(B, P, 3, 3)
    assert torch.allclose(sq["R"].transpose(-1, -2) @ sq["R"], eye, atol=1e-4)
    assert torch.allclose(torch.linalg.det(sq["R"]), torch.ones(B, P), atol=1e-4)
    assert (sq["scale"] >= SCALE_MIN).all() and torch.isfinite(sq["scale"]).all()
    assert (sq["eps"] >= EPS_MIN).all() and (sq["eps"] <= EPS_MAX).all()
    assert sq["exist_logit"].shape == (B, P) and torch.isfinite(sq["exist_logit"]).all()


def test_overfit_one_batch() -> None:
    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    denoiser = FlowSQDenoiser(d_model=128, n_queries=P, n_layers=2, n_heads=4, ffn_dim=256)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    mem, _ = denoiser.encode_tokens(tokens)
    assert denoiser(torch.randn(B, P, 15), torch.rand(B), mem=mem, n_views=2).shape == (B, P, 15)

    print("=== Overfit 1 fixed batch (500 steps) ===")
    for step in range(1, 501):
        loss, parts = flow_loss(denoiser, tokens, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % 100 == 0:
            print(f"  step {step:3d}: loss={loss.item():.4f}  exist={parts['exist']:.4f}  "
                  f"t={parts['t']:.4f}  R={parts['R']:.4f}  scale={parts['scale']:.4f}  eps={parts['eps']:.4f}")

    denoiser.eval()
    # Check deterministic mid-path t=0.5 velocity error (excludes t->1 unobservable-x0 variance)
    with torch.no_grad():
        eval_gen = torch.Generator().manual_seed(7)
        x_1 = pack_sq(tgt)
        x_0 = torch.randn(x_1.shape, generator=eval_gen)
        v_pred = denoiser(0.5 * x_1 + 0.5 * x_0, torch.full((B,), 0.5), tokens=tokens)
        err2 = (v_pred - (x_1 - x_0)) ** 2
        m = tgt["exist"]
        exist_mid = float(err2[..., 14].mean())
        total_mid = sum(float((err2[..., s].mean(-1) * m).sum() / m.sum()) for s in
                        [slice(0, 3), slice(3, 9), slice(9, 12), slice(12, 14)]) + exist_mid
    print(f"  eval@t=0.5: loss={total_mid:.4f}  exist={exist_mid:.4f}")
    assert loss.item() < 0.25 and total_mid < 0.25, f"overfit loss too high: step500={loss.item():.4f}, mid={total_mid:.4f}"
    assert parts["exist"] < 0.05 and exist_mid < 0.05, f"existence loss too high: step500={parts['exist']:.4f}, mid={exist_mid:.4f}"

    smp = sample_euler(denoiser, tokens, n_queries=P, n_steps=20, generator=torch.Generator().manual_seed(42))
    assert smp.shape == (B, P, 15) and torch.isfinite(smp).all()
    _assert_valid_sq(decode_sq(smp), B, P)


def test_train_py_flow_wiring() -> None:
    import l2sq.train as train_mod

    class FakeVGGT(torch.nn.Module):
        def __init__(self, layer: int = 23):
            super().__init__()
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            B, N = images.shape[:2]
            return torch.randn(B, N, 8, TOKEN_DIM)

    class FakePool:
        def __init__(self, folder, stride: int = 4):
            self.folder = Path(folder)
            self.w2c = np.repeat(np.eye(3, 4)[None], 8, axis=0)
            self.points_world = [np.array([[0.0, 0.0, 2.0]], dtype=np.float32) for _ in range(8)]
            self.images = torch.zeros(8, 3, 518, 518)
        def __len__(self) -> int:
            return 8
        def normalization(self, views: list[int]) -> float:
            return 2.0

    class FakeGT:
        def __init__(self, path, deform: bool = False):
            t = _fake_tgt(1, 16, 7)
            self.deform = deform
            self.R = t["R"][0, :7].numpy()
            self.t = t["t"][0, :7].numpy()
            self.scale = t["scale"][0, :7].numpy()
            self.eps = t["eps"][0, :7].numpy()
            if deform:
                self.taper = np.full((7, 2), 0.25, dtype=np.float64)
                self.bend = np.full((7, 3), 0.15, dtype=np.float64)
                self.alpha = np.full((7, 3), 0.5, dtype=np.float64)
            else:
                self.taper = self.bend = self.alpha = None
        def __len__(self) -> int:
            return 7

    class FakeGeom:
        def __init__(self, *a, **kw):
            self.surf = np.random.randn(256, 3).astype(np.float32) * 0.3
            self.vol = np.random.randn(256, 3).astype(np.float32) * 0.5
            self.occ = (np.linalg.norm(self.vol, axis=1) < 0.4).astype(np.float32)
        def draw(self, rng, n_surf: int, n_vol: int):
            return self.surf[:min(n_surf, len(self.surf))], self.vol[:min(n_vol, len(self.vol))], self.occ[:min(n_vol, len(self.vol))]

    class FakeMesh:
        def export(self, path):
            Path(path).write_text("ply\n")

    print("\n=== train.py --decoder flow CPU wiring test (1 train step + eval) ===")
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(train_mod, "FrozenVGGT", FakeVGGT), \
         patch.object(train_mod, "ViewPool", FakePool), \
         patch.object(train_mod, "GTSuperquadrics", FakeGT), \
         patch.object(train_mod, "GeomTargets", FakeGeom), \
         patch.object(train_mod, "sq_mesh", lambda *a, **kw: FakeMesh()):
        out = Path(tmp) / "run"
        train_mod.main([
            "--train", "fake_train", "--test", "fake_test", "--gt-sq", "fake.npz",
            "--out", str(out), "--decoder", "flow", "--layers", "1", "--d-model", "64",
            "--heads", "4", "--ffn", "128", "--steps", "1", "--log-every", "1",
            "--eval-every", "1", "--eval-per-n", "1", "--examples", "1", "--batch", "2",
            "--n-surf", "64", "--n-vol", "64", "--flow-steps", "4", "--device", "cpu", "--no-tb",
        ])
        final = json.loads((out / "eval_final.json").read_text())
        assert "n_empty" in final["test"]["all"]
        assert (out / "decoder.pt").is_file()

        out_tn = Path(tmp) / "run_tnorm"
        train_mod.main([
            "--train", "fake_train", "--test", "fake_test", "--gt-sq", "fake.npz",
            "--out", str(out_tn), "--decoder", "flow", "--flow-t-norm", "--layers", "1", "--d-model", "64",
            "--heads", "4", "--ffn", "128", "--steps", "1", "--log-every", "1",
            "--eval-every", "1", "--eval-per-n", "1", "--examples", "1", "--batch", "2",
            "--n-surf", "64", "--n-vol", "64", "--flow-steps", "4", "--device", "cpu", "--no-tb",
        ])
        ckpt_tn = torch.load(out_tn / "decoder.pt", map_location="cpu", weights_only=False)
        assert not torch.allclose(ckpt_tn["decoder"]["t_mean"], torch.zeros(3))

        out_all = Path(tmp) / "run_norm_all"
        train_mod.main([
            "--train", "fake_train", "--test", "fake_test", "--gt-sq", "fake.npz",
            "--out", str(out_all), "--decoder", "flow", "--flow-norm-all", "--layers", "1", "--d-model", "64",
            "--heads", "4", "--ffn", "128", "--steps", "1", "--log-every", "1",
            "--eval-every", "1", "--eval-per-n", "1", "--examples", "1", "--batch", "2",
            "--n-surf", "64", "--n-vol", "64", "--flow-steps", "4", "--device", "cpu", "--no-tb",
        ])
        ckpt_all = torch.load(out_all / "decoder.pt", map_location="cpu", weights_only=False)
        assert torch.allclose(ckpt_all["decoder"]["t_mean"], torch.zeros(3))
        assert ckpt_all["decoder"]["flow_shift"][14].item() == 0.5 and ckpt_all["decoder"]["flow_scale"][14].item() == 0.5
        assert (ckpt_all["decoder"]["flow_scale"][:14] >= 0.05).all()

        out_def = Path(tmp) / "run_flow_deform"
        train_mod.main([
            "--train", "fake_train", "--test", "fake_test", "--gt-sq", "fake_deform.npz",
            "--out", str(out_def), "--decoder", "flow", "--flow-deform", "--flow-norm-all",
            "--flow-loss", "mix", "--flow-vel-loss", "l2", "--flow-slot-sinus", "--flow-fourier", "2",
            "--flow-geo", "points", "--w-geo", "0.5",
            "--layers", "1", "--d-model", "64", "--heads", "4", "--ffn", "128",
            "--steps", "1", "--log-every", "1", "--eval-every", "1", "--eval-per-n", "1",
            "--examples", "1", "--batch", "2", "--n-surf", "64", "--n-vol", "64",
            "--flow-steps", "4", "--device", "cpu", "--no-tb",
        ])
        args_def = json.loads((out_def / "args.json").read_text())
        assert (args_def["flow_deform"] is True and args_def["flow_fourier"] == 2
                and args_def["flow_vel_loss"] == "l2" and args_def["flow_geo"] == "points" and args_def["w_geo"] == 0.5)
        final_def = json.loads((out_def / "eval_final.json").read_text())
        assert "taper_err" in final_def["test"]["all"] and "bend_err" in final_def["test"]["all"]
        ckpt_def = torch.load(out_def / "decoder.pt", map_location="cpu", weights_only=False)
        assert ckpt_def["decoder"]["flow_shift"].shape == (26,) and ckpt_def["decoder"]["flow_scale"].shape == (26,)
        assert ckpt_def["decoder"]["flow_shift"][14].item() == 0.5 and ckpt_def["decoder"]["flow_scale"][14].item() == 0.5

        # Verify evaluate() counts empty predictions and uses nanmean safely
        class EmptyDec(torch.nn.Module):
            deform = False
            def forward(self, tokens):
                B = tokens.shape[0]
                return {"t": torch.zeros(B, 16, 3), "R": torch.eye(3).expand(B, 16, 3, 3),
                        "scale": torch.ones(B, 16, 3), "eps": torch.ones(B, 16, 2),
                        "exist_logit": torch.full((B, 16), -10.0)}

        pool, gt, geom = FakePool("x"), FakeGT("x"), FakeGeom()
        samples = train_mod.eval_samples(pool, gt, (1, 2), 1, seed=0)
        r_empty = train_mod.evaluate(FakeVGGT(), EmptyDec(), pool, samples, 16, 1, "cpu",
                                     {"loss": "mse", "w": {"t": 1, "R": 1, "scale": 1, "eps": 1, "exist": 1},
                                      "scale_loss": "log", "decoder": "parallel"}, geom)
        assert r_empty["all"]["n_empty"] == 2 and np.isnan(r_empty["all"]["chamfer_world"])
        print(f"empty-pred eval check: n_empty={r_empty['all']['n_empty']}, chamfer={r_empty['all']['chamfer_world']}")

        # Verify --grad-clip 0 (default off) matches today's behavior identically, and --grad-clip > 0 skips bad steps
        out_gc_default = Path(tmp) / "run_gc_default"
        out_gc_zero = Path(tmp) / "run_gc_zero"
        common_args = [
            "--train", "fake_train", "--test", "fake_test", "--gt-sq", "fake.npz",
            "--decoder", "flow", "--layers", "1", "--d-model", "64", "--heads", "4", "--ffn", "128",
            "--steps", "2", "--log-every", "2", "--eval-every", "2", "--eval-per-n", "1",
            "--examples", "1", "--batch", "2", "--n-surf", "64", "--n-vol", "64",
            "--flow-steps", "4", "--device", "cpu", "--seed", "0", "--no-tb",
        ]
        train_mod.main([*common_args, "--out", str(out_gc_default)])
        train_mod.main([*common_args, "--out", str(out_gc_zero), "--grad-clip", "0"])
        args_gcd = json.loads((out_gc_default / "args.json").read_text())
        assert args_gcd["grad_clip"] == 0.0
        sd_def = torch.load(out_gc_default / "decoder.pt", map_location="cpu", weights_only=False)["decoder"]
        sd_zero = torch.load(out_gc_zero / "decoder.pt", map_location="cpu", weights_only=False)["decoder"]
        assert all(torch.equal(sd_def[k], sd_zero[k]) for k in sd_def)

        # With --grad-clip > 0, non-finite loss skips opt.step() with a warning and keeps weights finite
        import io
        from contextlib import redirect_stderr
        out_gc_skip = Path(tmp) / "run_gc_skip"
        orig_fl = train_mod.flow_loss

        def nan_flow_loss(dec, *args, **kwargs):
            l, p = orig_fl(dec, *args, **kwargs)
            return l + float("nan"), p

        buf = io.StringIO()
        with patch.object(train_mod, "flow_loss", side_effect=nan_flow_loss), redirect_stderr(buf):
            train_mod.main([*common_args, "--out", str(out_gc_skip), "--grad-clip", "1.0"])
        assert "skipping optimizer step" in buf.getvalue()
        args_gcs = json.loads((out_gc_skip / "args.json").read_text())
        assert args_gcs["grad_clip"] == 1.0
        sd_skip = torch.load(out_gc_skip / "decoder.pt", map_location="cpu", weights_only=False)["decoder"]
        assert all(torch.isfinite(v).all() for v in sd_skip.values())


def test_flow_repeat() -> None:
    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    denoiser = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)

    # 1. K=1 matches reference unrepeated flow_loss bit-for-bit with fixed seed
    g1 = torch.Generator().manual_seed(99)
    loss_k1, parts_k1 = flow_loss(denoiser, tokens, tgt, generator=g1, repeat=1)
    g_ref = torch.Generator().manual_seed(99)
    x_1 = pack_sq(tgt)
    x_0 = torch.randn(x_1.shape, generator=g_ref)
    t_ref = torch.rand(B, generator=g_ref)
    v_ref = denoiser(t_ref[:, None, None] * x_1 + (1.0 - t_ref[:, None, None]) * x_0, t_ref, tokens=tokens)
    err2 = (v_ref - (x_1 - x_0)) ** 2
    m = tgt["exist"]
    loss_ref = sum((err2[..., sl].mean(-1) * m).sum() / m.sum() for sl in
                   [slice(0, 3), slice(3, 9), slice(9, 12), slice(12, 14)]) + err2[..., 14].mean()
    assert torch.allclose(loss_k1, loss_ref, atol=1e-6), f"{loss_k1} vs {loss_ref}"

    # 2. K=4 encodes tokens once, expands batch to B*4 with independent t/x_0, finite loss
    seen = []
    orig_encode = denoiser.encode_tokens
    orig_forward = denoiser.forward
    encode_calls = 0

    def spy_encode(tok):
        nonlocal encode_calls
        encode_calls += 1
        return orig_encode(tok)

    def spy_forward(x_t, t, tokens=None, mem=None, n_views=None):
        seen.append((x_t.shape, t.shape, mem.shape, t.detach().clone(), x_t.detach().clone()))
        return orig_forward(x_t, t, tokens=tokens, mem=mem, n_views=n_views)

    with patch.object(denoiser, "encode_tokens", side_effect=spy_encode), \
         patch.object(denoiser, "forward", side_effect=spy_forward):
        loss_k4, parts_k4 = flow_loss(denoiser, tokens, tgt, generator=torch.Generator().manual_seed(123), repeat=4)
    assert encode_calls == 1
    x_shape, t_shape, mem_shape, t_seen, x_t_seen = seen[0]
    assert x_shape == (B * 4, P, 15) and t_shape == (B * 4,) and mem_shape == (B * 4, 2 * 8, 64)
    assert len(torch.unique(t_seen)) == B * 4
    assert not torch.allclose(x_t_seen[0], x_t_seen[1])
    assert torch.isfinite(loss_k4) and all(np.isfinite(v) for v in parts_k4.values())

    # 3. Short single-batch overfit with K=4 converges
    print("\n=== Single-batch overfit with K=4 (200 steps) ===")
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)
    hist = []
    for step in range(1, 201):
        loss, parts = flow_loss(denoiser, tokens, tgt, repeat=4)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d}: loss={loss.item():.4f}  exist={parts['exist']:.4f}")
    assert sum(hist[-10:]) / 10 < sum(hist[:10]) / 10 * 0.25


def test_flow_loss_mix() -> None:
    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    denoiser = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)

    # 1. Default mode="velocity" matches explicit velocity loss bit-for-bit
    l_def, p_def = flow_loss(denoiser, tokens, tgt, generator=torch.Generator().manual_seed(77))
    l_vel, p_vel = flow_loss(denoiser, tokens, tgt, generator=torch.Generator().manual_seed(77), mode="velocity")
    assert torch.allclose(l_def, l_vel, atol=1e-7) and p_def == p_vel

    # 2. mode="mix" is finite, logs v_loss & x1_loss separately, and backprops finite gradients
    denoiser.zero_grad()
    l_mix, p_mix = flow_loss(denoiser, tokens, tgt, generator=torch.Generator().manual_seed(77),
                             repeat=2, mode="mix", w_x1=1.0)
    assert torch.isfinite(l_mix) and "v_loss" in p_mix and "x1_loss" in p_mix
    assert np.isfinite(p_mix["v_loss"]) and np.isfinite(p_mix["x1_loss"])
    l_mix.backward()
    grads = [p.grad for p in denoiser.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

    # 3. Short single-batch overfit with mode="mix" converges
    print("\n=== Single-batch overfit with --flow-loss mix (200 steps) ===")
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)
    hist = []
    for step in range(1, 201):
        loss, parts = flow_loss(denoiser, tokens, tgt, repeat=2, mode="mix", w_x1=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        hist.append((loss.item(), parts["v_loss"], parts["x1_loss"]))
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d}: total={loss.item():.4f}  v_loss={parts['v_loss']:.4f}  x1_loss={parts['x1_loss']:.4f}")
    assert sum(x[0] for x in hist[-10:]) / 10 < sum(x[0] for x in hist[:10]) / 10 * 0.25


def test_flow_t_norm() -> None:
    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    t_mean = torch.tensor([0.1, -0.2, 1.0])
    t_std = torch.tensor([0.3, 0.4, 0.5])

    # 1. Default off (zeros/ones) is bit-for-bit identical to None
    p_none = pack_sq(tgt)
    p_def = pack_sq(tgt, t_mean=torch.zeros(3), t_std=torch.ones(3))
    assert torch.equal(p_none, p_def)
    d_none = decode_sq(p_none)
    d_def = decode_sq(p_def, t_mean=torch.zeros(3), t_std=torch.ones(3))
    assert all(torch.equal(d_none[k], d_def[k]) for k in d_none)

    # 2. Pack/decode round trip with t_norm on (only t normalized; rest unchanged)
    p_norm = pack_sq(tgt, t_mean=t_mean, t_std=t_std)
    assert torch.allclose(p_norm[..., :3], (tgt["t"] - t_mean) / t_std, atol=1e-6)
    assert torch.equal(p_norm[..., 3:], p_none[..., 3:])
    rt = decode_sq(p_norm, t_mean=t_mean, t_std=t_std)
    _assert_valid_sq(rt, B, P)
    assert torch.allclose(rt["t"][:, :M], tgt["t"][:, :M], atol=1e-5)

    # 3. Single-batch overfit with t_norm on (velocity + mix) & checkpoint buffer reload
    denoiser = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    denoiser.set_t_norm(t_mean, t_std)
    print("\n=== Single-batch overfit with --flow-t-norm on (200 steps) ===")
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)
    hist = []
    for step in range(1, 201):
        loss, parts = flow_loss(denoiser, tokens, tgt, repeat=2, mode="mix", w_x1=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d}: total={loss.item():.4f}  v_loss={parts['v_loss']:.4f}  x1_loss={parts['x1_loss']:.4f}")
    assert sum(hist[-10:]) / 10 < sum(hist[:10]) / 10 * 0.25

    reloaded = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    reloaded.load_state_dict(denoiser.state_dict())
    assert torch.allclose(reloaded.t_mean, t_mean) and torch.allclose(reloaded.t_std, t_std)


def test_flow_norm_all() -> None:
    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    raw = pack_sq(tgt, normalize=False)[tgt["exist"] > 0.5]
    sh = torch.cat([raw.mean(0)[:14], torch.tensor([0.5])])
    sc = torch.cat([raw.std(0, unbiased=False).clamp_min(0.05)[:14], torch.tensor([0.5])])

    # 1. Default off and --flow-t-norm alone keep exact behavior
    d_def = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    l_def, _ = flow_loss(d_def, tokens, tgt, generator=torch.Generator().manual_seed(55))
    assert torch.allclose(pack_sq(tgt), pack_sq(tgt, t_mean=d_def.t_mean, t_std=d_def.t_std,
                                                shift=d_def.flow_shift, scale=d_def.flow_scale))
    t_mean, t_std = torch.tensor([0.1, -0.2, 0.9]), torch.tensor([0.3, 0.4, 0.5])
    d_tn = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    d_tn.set_t_norm(t_mean, t_std)
    assert torch.equal(pack_sq(tgt, t_mean=t_mean, t_std=t_std),
                       pack_sq(tgt, t_mean=d_tn.t_mean, t_std=d_tn.t_std,
                               shift=d_tn.flow_shift, scale=d_tn.flow_scale))

    # 2. Round trip with --flow-norm-all on
    x_norm = pack_sq(tgt, shift=sh, scale=sc)
    assert torch.allclose(x_norm[..., :14], (pack_sq(tgt, normalize=False)[..., :14] - sh[:14]) / sc[:14], atol=1e-6)
    assert torch.allclose(x_norm[..., 14], (tgt["exist"] - 0.5) / 0.5, atol=1e-6)
    rt = decode_sq(x_norm, shift=sh, scale=sc)
    _assert_valid_sq(rt, B, P)
    assert torch.allclose(rt["t"][:, :M], tgt["t"][:, :M], atol=1e-5)
    assert torch.allclose(rt["R"][:, :M], tgt["R"][:, :M], atol=1e-5)
    assert torch.allclose(rt["scale"][:, :M], tgt["scale"][:, :M], atol=1e-4)
    assert torch.allclose(rt["eps"][:, :M], tgt["eps"][:, :M], atol=1e-4)

    # 3. Single-batch overfit with --flow-norm-all (mix loss -> real-space x1 comparison) + buffer reload
    denoiser = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    denoiser.set_flow_norm(sh, sc)
    print("\n=== Single-batch overfit with --flow-norm-all (200 steps) ===")
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)
    hist = []
    for step in range(1, 201):
        loss, parts = flow_loss(denoiser, tokens, tgt, repeat=2, mode="mix", w_x1=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d}: total={loss.item():.4f}  v_loss={parts['v_loss']:.4f}  x1_loss={parts['x1_loss']:.4f}")
    assert sum(hist[-10:]) / 10 < sum(hist[:10]) / 10 * 0.25

    reloaded = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=2, n_heads=4, ffn_dim=128)
    reloaded.load_state_dict(denoiser.state_dict())
    assert torch.allclose(reloaded.flow_shift, sh) and torch.allclose(reloaded.flow_scale, sc)
    assert torch.isfinite(l_def)


def test_slot_init_sinus() -> None:
    from l2sq.model import sinusoidal

    B = 2
    denoiser = FlowSQDenoiser(d_model=256, n_queries=16, n_layers=1, n_heads=8, ffn_dim=256, slot_init="sinus")
    expected = sinusoidal(16, 256)
    assert denoiser.slot_emb.shape == (1, 16, 256)
    assert torch.equal(denoiser.slot_emb.squeeze(0), expected)
    out = denoiser(torch.randn(B, 16, 15), torch.rand(B), tokens=torch.randn(B, 2, 8, TOKEN_DIM))
    assert out.shape == (B, 16, 15) and torch.isfinite(out).all()


def test_fourier_and_l2_vel_loss() -> None:
    from l2sq.eval_occlusion import load_decoder

    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M, seed=11)
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)

    # 1. Fourier input embedding (fourier_freqs=4 -> in_features = 15 * (1 + 2 * 4) = 135)
    den_f = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128,
                           slot_init="sinus", fourier_freqs=4)
    assert den_f.x_proj.in_features == 15 * (1 + 2 * 4)
    x_t = torch.randn(B, P, 15)
    v_out = den_f(x_t, torch.rand(B), tokens=tokens)
    assert v_out.shape == (B, P, 15) and torch.isfinite(v_out).all()

    # Checkpoint reload with new options AND legacy checkpoint without new keys
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        new_ckpt = tmp_dir / "new.pt"
        old_ckpt = tmp_dir / "old.pt"
        base_args = {"d_model": 64, "queries": P, "layers": 1, "heads": 4, "ffn": 128, "decoder": "flow"}
        torch.save({"decoder": den_f.state_dict(),
                    "args": {**base_args, "flow_slot_sinus": True, "flow_fourier": 4, "flow_vel_loss": "l2"}},
                   new_ckpt)
        den_old = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128)
        torch.save({"decoder": den_old.state_dict(), "args": base_args}, old_ckpt)

        kind_n, rel_new, _, _ = load_decoder(new_ckpt, "cpu")
        kind_o, rel_old, _, _ = load_decoder(old_ckpt, "cpu")
        assert kind_n == "flow" and rel_new.fourier_freqs == 4 and rel_new.x_proj.in_features == 135
        assert torch.equal(rel_new.slot_emb, den_f.slot_emb)
        assert kind_o == "flow" and rel_old.fourier_freqs == 0 and rel_old.x_proj.in_features == 15

    # 2. Euclidean velocity loss (vel_loss="l2"): finite, finite grads, != mse, exact group structure
    loss_mse, parts_mse = flow_loss(den_f, tokens, tgt, generator=torch.Generator().manual_seed(99), vel_loss="mse")
    loss_l2, parts_l2 = flow_loss(den_f, tokens, tgt, generator=torch.Generator().manual_seed(99), vel_loss="l2")
    assert set(parts_l2.keys()) == set(parts_mse.keys()) == {"t", "R", "scale", "eps", "exist"}
    assert torch.isfinite(loss_l2) and loss_l2.item() != loss_mse.item()
    loss_l2.backward()
    grads = [p.grad for p in den_f.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)

    # Also verify mix mode works with vel_loss="l2"
    den_f.zero_grad()
    loss_mix_l2, parts_mix = flow_loss(den_f, tokens, tgt, generator=torch.Generator().manual_seed(99),
                                       mode="mix", w_x1=1.0, vel_loss="l2")
    assert torch.isfinite(loss_mix_l2) and "x1_loss" in parts_mix
    loss_mix_l2.backward()
    assert all(torch.isfinite(p.grad).all() for p in den_f.parameters() if p.grad is not None)


def test_flow_deform() -> None:
    from l2sq.eval_occlusion import load_decoder

    torch.manual_seed(0)
    B, P, M = 2, 16, 7
    tgt = _fake_tgt(B, P, M, seed=21)
    g = torch.Generator().manual_seed(22)
    tgt["taper"] = torch.rand(B, P, 2, generator=g) * 1.8 - 0.9
    tgt["bend"] = torch.rand(B, P, 3, generator=g) * 0.95 + 0.01
    tgt["alpha"] = torch.rand(B, P, 3, generator=g) * (2.0 * np.pi) - np.pi

    # 1. 26-dim pack/decode round trip (angle -> (cos, sin) -> atan2 angle, normalized and unnormalized)
    raw = pack_sq(tgt, normalize=False, deform=True)
    assert raw.shape == (B, P, 26)
    expected_cs = torch.stack([tgt["alpha"].cos(), tgt["alpha"].sin()], dim=-1).flatten(-2)
    assert torch.allclose(raw[..., 20:26], expected_cs, atol=1e-6)

    x_norm = pack_sq(tgt, normalize=True, deform=True)
    assert x_norm.shape == (B, P, 26)
    rt = decode_sq(x_norm, normalized=True, deform=True)
    _assert_valid_sq(rt, B, P)
    assert torch.allclose(rt["taper"][:, :M], tgt["taper"][:, :M], atol=1e-5)
    assert torch.allclose(rt["bend"][:, :M], tgt["bend"][:, :M], atol=1e-4)
    # Compare circular angle modulo 2*pi via (cos, sin)
    assert torch.allclose(rt["alpha"][:, :M].cos(), tgt["alpha"][:, :M].cos(), atol=1e-5)
    assert torch.allclose(rt["alpha"][:, :M].sin(), tgt["alpha"][:, :M].sin(), atol=1e-5)

    # 2. Forward shape, sample_euler, and checkpoint reload with slot_init="sinus" & fourier_freqs=4
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)
    den = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128,
                         slot_init="sinus", fourier_freqs=4, deform=True)
    assert den.sq_dim == 26 and den.deform is True
    assert den.x_proj.in_features == 26 * (1 + 2 * 4)
    assert den.out_proj.out_features == 26
    v_out = den(torch.randn(B, P, 26), torch.rand(B), tokens=tokens)
    assert v_out.shape == (B, P, 26) and torch.isfinite(v_out).all()

    smp = sample_euler(den, tokens, n_queries=P, n_steps=3, generator=torch.Generator().manual_seed(5))
    assert smp.shape == (B, P, 26) and torch.isfinite(smp).all()
    dec_out = decode_sq(smp, t_mean=den.t_mean, t_std=den.t_std, shift=den.flow_shift, scale=den.flow_scale, deform=True)
    assert set(("taper", "bend", "alpha", "alpha_vec")).issubset(dec_out.keys())

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_p = Path(tmp) / "deform_flow.pt"
        args = {"d_model": 64, "queries": P, "layers": 1, "heads": 4, "ffn": 128,
                "decoder": "flow", "flow_slot_sinus": True, "flow_fourier": 4, "flow_deform": True}
        torch.save({"decoder": den.state_dict(), "args": args}, ckpt_p)
        kind, rel, _, _ = load_decoder(ckpt_p, "cpu")
        assert kind == "flow" and rel.deform and rel.sq_dim == 26 and rel.fourier_freqs == 4
        assert torch.allclose(rel.flow_shift, den.flow_shift) and torch.allclose(rel.flow_scale, den.flow_scale)

    # 3. flow_loss (velocity + mix, mse + l2) is finite with finite grads and reports new parts
    for vel_mode in ("mse", "l2"):
        den.zero_grad()
        loss, parts = flow_loss(den, tokens, tgt, generator=torch.Generator().manual_seed(7),
                                mode="mix", w_x1=1.0, vel_loss=vel_mode)
        assert torch.isfinite(loss)
        for key in ("taper", "bend", "alpha", "x1_taper", "x1_bend", "x1_alpha"):
            assert key in parts and np.isfinite(parts[key])
        loss.backward()
        grads = [p.grad for p in den.parameters() if p.grad is not None]
        assert len(grads) > 0 and all(torch.isfinite(gr).all() for gr in grads)


def test_flow_geo_loss() -> None:
    from l2sq.flow_matching_decoder import _geo_surface_loss

    torch.manual_seed(0)
    B, P, M = 2, 8, 4
    tgt = _fake_tgt(B, P, M, seed=31)
    tgt["s"] = torch.tensor([1.5, 2.0])
    tgt["surf"] = torch.randn(B, 128, 3) * 0.2
    tokens = torch.randn(B, 2, 8, TOKEN_DIM)

    # 1. points geo loss: 0 when pred == tgt, strictly positive when rotated 20 deg or shifted
    m = tgt["exist"]
    n = m.sum().clamp_min(1.0)
    wt = torch.ones(B, 1)
    rep = lambda a: a
    # Make slot 0 an elongated thin rod so 20-deg rotation clearly moves surface points
    tgt_rod = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt.items()}
    tgt_rod["scale"][:, 0] = torch.tensor([0.3, 0.02, 0.02])
    tgt_rod["R"][:, 0] = torch.eye(3)

    g_zero = _geo_surface_loss(tgt_rod, tgt_rod, wt, m, n, rep, False, "points", 1)
    assert g_zero.item() == 0.0

    ang = np.deg2rad(20.0)
    Rz = torch.tensor([[np.cos(ang), -np.sin(ang), 0.0],
                       [np.sin(ang),  np.cos(ang), 0.0],
                       [0.0,          0.0,         1.0]], dtype=torch.float32)
    pred_rot = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt_rod.items()}
    pred_rot["R"][0, 0] = Rz @ pred_rot["R"][0, 0]
    g_rot = _geo_surface_loss(pred_rot, tgt_rod, wt, m, n, rep, False, "points", 1)
    assert g_rot.item() > 1e-4, f"expected rotation loss > 0, got {g_rot.item()}"

    pred_shift = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt_rod.items()}
    pred_shift["t"][0, 0, 0] += 0.1
    g_shift = _geo_surface_loss(pred_shift, tgt_rod, wt, m, n, rep, False, "points", 1)
    assert g_shift.item() > 1e-4, f"expected shift loss > 0, got {g_shift.item()}"

    # 2. Direct rot6d -> decode_sq -> _geo_surface_loss gradient check
    x_raw = pack_sq(tgt_rod, normalize=True, deform=False).requires_grad_(True)
    dec_x = decode_sq(x_raw, normalized=True, smooth_exist=True)
    g_diff = _geo_surface_loss(dec_x, pred_rot, wt, m, n, rep, False, "points", 1)
    g_diff.backward()
    assert x_raw.grad is not None and torch.isfinite(x_raw.grad).all()
    assert x_raw.grad[:, :M, 3:9].abs().sum() > 0    # rot6d gets gradient
    assert x_raw.grad[:, :M, 0:3].abs().sum() > 0    # t gets gradient

    # 3. flow_loss with geo_loss in ("points", "chamfer"), both rigid and deform + all flags
    for deform in (False, True):
        t_cur = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt.items()}
        if deform:
            t_cur["taper"] = torch.zeros(B, P, 2) + 0.1
            t_cur["bend"] = torch.zeros(B, P, 3) + 0.2
            t_cur["alpha"] = torch.zeros(B, P, 3) + 0.4
        den = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128,
                             slot_init="sinus", fourier_freqs=2, deform=deform)
        # Default geo_loss="none" keeps parts without "geo"
        l_none, p_none = flow_loss(den, tokens, t_cur, generator=torch.Generator().manual_seed(55),
                                   mode="mix", w_x1=1.0, vel_loss="l2")
        assert "geo" not in p_none and torch.isfinite(l_none)

        for geo_mode in ("points", "chamfer"):
            den.zero_grad()
            loss, parts = flow_loss(den, tokens, t_cur, generator=torch.Generator().manual_seed(55),
                                    mode="mix", w_x1=1.0, vel_loss="l2", geo_loss=geo_mode, w_geo=0.5)
            assert "geo" in parts and np.isfinite(parts["geo"]) and torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in den.parameters() if p.grad is not None]
            assert len(grads) > 0 and all(torch.isfinite(gr).all() for gr in grads)


    # 4. Geo chamfer robustness against zero-scale, NaN-prone, and empty-active-slot x1_hat
    x_bad = torch.zeros(B, P, 15)
    x_bad[:, 0, 9:12] = -1e6                               # tiny/zero scale after exp clamp
    x_bad[:, 1, 9:12] = float("nan")                       # NaN-prone scale
    x_bad[:, 1, 12:14] = float("inf")                      # NaN-prone eps
    x_bad[:, 1, 3:9] = 0.0                                 # zero rot6d -> zero cross area
    x_bad.requires_grad_(True)
    pred_bad = decode_sq(x_bad, normalized=False, smooth_exist=True)
    # Also directly set scale=0 on slot 0 to verify zero-scale tolerance without decode clamp
    pred_bad_direct = {**pred_bad, "scale": pred_bad["scale"].clone()}
    pred_bad_direct["scale"][:, 0] = 0.0
    g_bad = _geo_surface_loss(pred_bad_direct, tgt, wt, m, n, rep, False, "chamfer", 1)
    assert torch.isfinite(g_bad)
    g_bad.backward()
    assert x_bad.grad is not None and torch.isfinite(x_bad.grad).all()
    # Finite active slot (slot 2) still receives non-zero gradient through chamfer
    assert x_bad.grad[:, 2, 0:3].abs().sum() > 0

    # Sample with zero active teacher slots skips cleanly (0 loss, finite)
    m_partial = torch.zeros(B, P)
    m_partial[0, :M] = 1.0
    g_part = _geo_surface_loss(tgt, tgt, wt, m_partial, m_partial.sum().clamp_min(1.0), rep, False, "chamfer", 1)
    assert torch.isfinite(g_part)
    m_none = torch.zeros(B, P)
    g_empty = _geo_surface_loss(tgt, tgt, wt, m_none, m_none.sum().clamp_min(1.0), rep, False, "chamfer", 1)
    assert g_empty.item() == 0.0

    # 5. Geo points robustness in deform mode: zero-scale, eps at clamp limits, non-finite SQ, and multi-step stability
    tgt_def = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt.items()}
    tgt_def["taper"] = torch.zeros(B, P, 2) + 0.1
    tgt_def["bend"] = torch.zeros(B, P, 3) + 0.2
    tgt_def["alpha"] = torch.zeros(B, P, 3) + 0.4
    tgt_def_nan_pad = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tgt_def.items()}
    tgt_def_nan_pad["scale"][:, -1] = float("nan")     # non-finite inactive teacher slot

    x_bad_def = torch.zeros(B, P, 26)
    x_bad_def[:, 0, 9:12] = -1e6                       # tiny/zero scale
    x_bad_def[:, 1, 12:14] = torch.tensor([-1e6, 1e6]) # eps at EPS_MIN and EPS_MAX clamp limits
    x_bad_def[:, 1, 3:9] = 0.0                         # zero rot6d
    x_bad_def[:, 1, 20:26] = 0.0                       # zero bend direction
    x_bad_def[:, 2, 3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])
    x_bad_def[:, 2, 9:12] = -1.5                       # finite active slot
    x_bad_def[:, 3, :] = float("nan")                  # non-finite SQ (inactive slot)
    x_bad_def[:, 0, 15:17] = float("inf")              # non-finite taper on active slot
    x_bad_def.requires_grad_(True)
    pred_bad_def = decode_sq(x_bad_def, normalized=False, smooth_exist=True, deform=True)
    pred_bad_def_direct = {**pred_bad_def, "scale": pred_bad_def["scale"].clone()}
    pred_bad_def_direct["scale"][:, 0] = 0.0           # literal zero scale
    g_pts_bad = _geo_surface_loss(pred_bad_def_direct, tgt_def_nan_pad, wt, m, n, rep, True, "points", 1)
    assert torch.isfinite(g_pts_bad)
    g_pts_bad.backward()
    assert x_bad_def.grad is not None and torch.isfinite(x_bad_def.grad).all()
    assert x_bad_def.grad[:, 2, 0:3].abs().sum() > 0

    # End-to-end flow_loss(geo_loss="points") in deform mode with extreme x1_hat (zero-scale, eps limits, overflow-prone)
    class _ExtremeFlowWrapper(torch.nn.Module):
        def __init__(self, base: FlowSQDenoiser):
            super().__init__()
            self.base = base
            for attr in ("deform", "sq_dim", "fourier_freqs", "t_mean", "t_std", "flow_shift", "flow_scale"):
                setattr(self, attr, getattr(base, attr))

        def encode_tokens(self, toks):
            return self.base.encode_tokens(toks)

        def forward(self, x_t, t_in, mem=None, n_views=None):
            v = self.base(x_t, t_in, mem=mem, n_views=n_views)
            v = v.clone()
            v[:, 0, 9:12] = v[:, 0, 9:12] - 1e6                # zero-scale SQ in x1_hat
            v[:, 1, 12:14] = v[:, 1, 12:14] + torch.tensor([-1e6, 1e6])  # eps at clamp limits
            v[:, 2, 3:9] = v[:, 2, 3:9] * 0.0                  # zero rot6d
            v[:, 2, 20:26] = v[:, 2, 20:26] * 0.0              # zero bend direction
            v[:, 3, 9:12] = v[:, 3, 9:12] + 1e6                # overflow/huge scale exponent
            return v

    den_def = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128, deform=True)
    wrap_def = _ExtremeFlowWrapper(den_def)
    wrap_def.zero_grad()
    loss_ex, parts_ex = flow_loss(wrap_def, tokens, tgt_def, generator=torch.Generator().manual_seed(9),
                                  mode="mix", w_x1=1.0, geo_loss="points", w_geo=20.0)
    assert torch.isfinite(loss_ex) and np.isfinite(parts_ex["geo"])
    loss_ex.backward()
    grads_ex = [p.grad for p in den_def.parameters() if p.grad is not None]
    assert len(grads_ex) > 0 and all(torch.isfinite(gr).all() for gr in grads_ex)

    # Also test flow_loss(geo_loss="points") with x1_hat injected with NaN directly into decode_sq
    import l2sq.flow_matching_decoder as fmd_mod
    real_decode = fmd_mod.decode_sq

    def _decode_with_nan_slot(x_hat, *args, **kwargs):
        x_hat = x_hat.clone()
        x_hat[:, 3, :] = float("nan")                  # non-finite SQ in x1_hat (active slot 3)
        return real_decode(x_hat, *args, **kwargs)

    den_def.zero_grad()
    with patch.object(fmd_mod, "decode_sq", side_effect=_decode_with_nan_slot):
        loss_nan_x1, parts_nan_x1 = flow_loss(wrap_def, tokens, tgt_def, generator=torch.Generator().manual_seed(11),
                                              mode="mix", w_x1=1.0, geo_loss="points", w_geo=20.0)
    assert torch.isfinite(loss_nan_x1) and np.isfinite(parts_nan_x1["geo"])
    loss_nan_x1.backward()
    grads_nan = [p.grad for p in den_def.parameters() if p.grad is not None]
    assert len(grads_nan) > 0 and all(torch.isfinite(gr).all() for gr in grads_nan)

    # Multi-step optimization stability at w_geo=20.0 in deform mode (regression for step-2 pole atan2 NaN)
    den_stab = FlowSQDenoiser(d_model=64, n_queries=P, n_layers=1, n_heads=4, ffn_dim=128, deform=True)
    opt_stab = torch.optim.AdamW(den_stab.parameters(), lr=2e-4)
    for _ in range(10):
        l_s, p_s = flow_loss(den_stab, tokens, tgt_def, mode="mix", w_x1=1.0, geo_loss="points", w_geo=20.0)
        opt_stab.zero_grad()
        l_s.backward()
        gn_s = torch.nn.utils.clip_grad_norm_(den_stab.parameters(), 1.0)
        assert torch.isfinite(l_s) and np.isfinite(p_s["geo"]) and torch.isfinite(gn_s)
        opt_stab.step()


if __name__ == "__main__":
    test_overfit_one_batch()
    test_flow_repeat()
    test_flow_loss_mix()
    test_flow_t_norm()
    test_flow_norm_all()
    test_slot_init_sinus()
    test_fourier_and_l2_vel_loss()
    test_flow_deform()
    test_flow_geo_loss()
    test_train_py_flow_wiring()
    print("\nALL CPU FLOW TESTS PASSED")
