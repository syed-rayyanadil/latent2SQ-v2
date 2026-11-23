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
        deform = False
        def __init__(self, path, deform: bool = False):
            t = _fake_tgt(1, 16, 7)
            self.R = t["R"][0, :7].numpy()
            self.t = t["t"][0, :7].numpy()
            self.scale = t["scale"][0, :7].numpy()
            self.eps = t["eps"][0, :7].numpy()
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


if __name__ == "__main__":
    test_overfit_one_batch()
    test_flow_repeat()
    test_train_py_flow_wiring()
    print("\nALL CPU FLOW TESTS PASSED")
