"""CPU-only smoke test for l2sq/eval_occlusion.py with dummy tokens and checkpoints."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from l2sq.eval_occlusion import (
    BOX_PS,
    EVAL_NS,
    HALF_SIDES,
    apply_box_occlusion,
    apply_half_occlusion,
    evaluate_occlusion,
    load_decoder,
)
from l2sq.flow_matching_decoder import FlowSQDenoiser
from l2sq.model import SQDecoder, TOKEN_DIM


class DummyVGGT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        B, N = images.shape[:2]
        # Deterministic feature signal derived from the occluded images.
        pooled = images.mean(dim=(-2, -1))                      # (B, N, 3)
        tokens = torch.zeros(B, N, 4, TOKEN_DIM, device=images.device)
        tokens[..., :3] = pooled[:, :, None, :]
        return tokens


class DummyPool:
    def __init__(self, n_views: int = 10) -> None:
        self.w2c = np.repeat(np.eye(3, 4, dtype=np.float32)[None], n_views, axis=0)
        self.points_world = [np.array([[0.0, 0.0, 2.0]], dtype=np.float32) for _ in range(n_views)]
        self.images = torch.rand(n_views, 3, 64, 64)

    def __len__(self) -> int:
        return len(self.points_world)

    def normalization(self, views: list[int]) -> float:
        return 1.5


class DummyGT:
    deform = False

    def __init__(self, M: int = 5) -> None:
        rng = np.random.default_rng(0)
        self.R = np.repeat(np.eye(3, dtype=np.float64)[None], M, axis=0)
        self.t = rng.normal(0.0, 0.2, size=(M, 3))
        self.scale = rng.uniform(0.05, 0.2, size=(M, 3))
        self.eps = rng.uniform(0.3, 1.5, size=(M, 2))

    def __len__(self) -> int:
        return len(self.R)


class DummyGeom:
    def draw(self, rng: np.random.Generator, n_surf: int, n_vol: int):
        surf = rng.normal(0.0, 0.3, size=(128, 3)).astype(np.float32)
        vol = rng.uniform(-0.5, 0.5, size=(256, 3)).astype(np.float32)
        occ = (np.linalg.norm(vol, axis=-1) < 0.3).astype(np.float32)
        return surf, vol, occ


def test_occlusion_masks() -> None:
    imgs = torch.ones(2, 4, 3, 64, 64)
    rng = np.random.default_rng(0)
    b0 = apply_box_occlusion(imgs, 0.0, rng)
    assert torch.equal(b0, imgs)
    b25 = apply_box_occlusion(imgs, 0.25, rng)
    frac = (b25[:, :, 0] == 0.5).float().mean(dim=(-2, -1))
    assert ((frac - 0.25).abs() < 0.02).all(), f"unexpected box area fraction: {frac}"

    for side in HALF_SIDES:
        h = apply_half_occlusion(imgs, side)
        assert abs(float((h == 0.5).float().mean()) - 0.5) < 1e-5


def test_load_and_eval_smoke() -> None:
    P, D, L, H, FFN = 8, 32, 1, 4, 64
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        p_ckpt_path = tmp_dir / "parallel_decoder.pt"
        f_ckpt_path = tmp_dir / "flow_decoder.pt"
        csv_path = tmp_dir / "results.csv"

        base_args = {
            "d_model": D, "queries": P, "layers": L, "heads": H, "ffn": FFN,
            "vggt_layer": 23, "test": "dummy/test", "gt_sq": "dummy/gt_sq.npz",
            "batch": 2, "eval_per_n": 2,
        }
        par_dec = SQDecoder(D, P, L, H, FFN, deform=False)
        torch.save({"decoder": par_dec.state_dict(), "args": {**base_args}, "step": 100}, p_ckpt_path)

        flow_dec = FlowSQDenoiser(D, P, L, H, FFN)
        flow_dec.set_t_norm(torch.tensor([0.1, -0.2, 0.8]), torch.tensor([0.2, 0.3, 0.4]))
        torch.save({"decoder": flow_dec.state_dict(),
                    "args": {**base_args, "decoder": "flow", "flow_t_norm": True},
                    "step": 100}, f_ckpt_path)

        k1, dec1, _, _ = load_decoder(p_ckpt_path, "cpu")
        k2, dec2, _, _ = load_decoder(f_ckpt_path, "cpu")
        assert k1 == "parallel" and k2 == "flow"
        assert torch.allclose(dec2.t_mean, torch.tensor([0.1, -0.2, 0.8]))
        assert torch.allclose(dec2.t_std, torch.tensor([0.2, 0.3, 0.4]))

        vggt = DummyVGGT()
        rows = evaluate_occlusion(
            vggt=vggt, dec_parallel=dec1, dec_flow=dec2,
            test_pool=DummyPool(), gt=DummyGT(), geom=DummyGeom(), P=P,
            batch_size=2, per_n=2, out_csv=csv_path,
            k_samples=5, flow_steps=3, seed=1234, mask_seed=1234, device="cpu",
        )
        # VGGT runs once per batch: 4 box ps * 3 Ns * 1 batch + 4 half sides * 3 Ns * 1 batch = 24 calls
        assert vggt.calls == (len(BOX_PS) + len(HALF_SIDES)) * len(EVAL_NS)
        assert csv_path.exists()
        with csv_path.open() as f:
            parsed = list(csv.DictReader(f))
        assert len(parsed) == len(rows) > 0
        # box: 4 p * 4 N (1,4,8,all) * 3 (parallel, flow mean, flow best) = 48
        # half: (4 sides * 4 N + 1 avg * 4 N) * 3 = 60  -> total 108 rows
        assert len(parsed) == 108, f"expected 108 rows, got {len(parsed)}"
        for r in parsed:
            if r["stat"] == "mean":
                assert np.isfinite(float(r["t_spread_world"]))


if __name__ == "__main__":
    test_occlusion_masks()
    test_load_and_eval_smoke()
    print("ALL OCCLUSION EVAL SMOKE TESTS PASSED")
