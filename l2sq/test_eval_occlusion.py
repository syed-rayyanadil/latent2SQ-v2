"""CPU-only smoke test for l2sq/eval_occlusion.py with dummy tokens and checkpoints."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from l2sq.data import augment_occlusion
from l2sq.eval_occlusion import (
    BOX_PS,
    EVAL_NS,
    HALF_SIDES,
    apply_box_occlusion,
    apply_half_occlusion,
    evaluate_occlusion,
    load_decoder,
    main as eval_main,
)
from l2sq.flow_matching_decoder import FlowSQDenoiser
from l2sq.model import SQDecoder, TOKEN_DIM, sinusoidal


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
                assert np.isfinite(float(r["chamfer_spread_world"]))


def test_current_flow_checkpoints_and_multi_eval() -> None:
    import l2sq.eval_occlusion as eo_mod

    P, D, L, H, FFN = 8, 32, 1, 4, 64

    class DummyDeformGT(DummyGT):
        deform = True

        def __init__(self, M: int = 5) -> None:
            super().__init__(M)
            self.taper = np.full((M, 2), 0.15, dtype=np.float64)
            self.bend = np.full((M, 3), 0.2, dtype=np.float64)
            self.alpha = np.full((M, 3), 0.4, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        par_dir = tmp_dir / "par_base"
        xl_dir = tmp_dir / "flow_geoC100XL"
        def_dir = tmp_dir / "flow_deform8L"
        for d in (par_dir, xl_dir, def_dir):
            d.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "multi_results.csv"

        base_args = {
            "d_model": D, "queries": P, "layers": L, "heads": H, "ffn": FFN,
            "vggt_layer": 23, "test": "dummy/test", "gt_sq": "dummy/gt_sq_rigid.npz",
            "batch": 2, "eval_per_n": 2, "flow_steps": 3,
        }
        par_dec = SQDecoder(D, P, L, H, FFN, deform=False)
        torch.save({"decoder": par_dec.state_dict(), "args": {**base_args}, "step": 100}, par_dir / "decoder.pt")

        # Flow XL-style rigid checkpoint with sinus slot_init, fourier=4, flow_geo="chamfer"
        xl_dec = FlowSQDenoiser(D, P, L, H, FFN, slot_init="sinus", fourier_freqs=4, deform=False)
        xl_dec.set_t_norm(torch.tensor([0.05, -0.1, 0.7]), torch.tensor([0.2, 0.25, 0.3]))
        torch.save({
            "decoder": xl_dec.state_dict(),
            "args": {**base_args, "decoder": "flow", "flow_slot_sinus": True, "flow_fourier": 4,
                     "flow_deform": False, "flow_geo": "chamfer"},
            "step": 200,
        }, xl_dir / "decoder.pt")

        # Flow deform checkpoint with sq_dim=26, deform=True, sinus slot_init, fourier=2, flow_geo="points"
        def_dec = FlowSQDenoiser(D, P, L, H, FFN, slot_init="sinus", fourier_freqs=2, deform=True)
        def_dec.set_flow_norm(torch.zeros(26), torch.full((26,), 0.5))
        torch.save({
            "decoder": def_dec.state_dict(),
            "args": {**base_args, "gt_sq": "dummy/gt_sq_deform.npz", "decoder": "flow",
                     "flow_slot_sinus": True, "flow_fourier": 2, "flow_deform": True, "flow_geo": "points"},
            "step": 300,
        }, def_dir / "decoder.pt")

        # load_decoder supports run directories and rebuilds exact architectures like train.py
        _, rel_xl, args_xl, _ = load_decoder(xl_dir, "cpu")
        assert rel_xl.sq_dim == 15 and not rel_xl.deform and rel_xl.fourier_freqs == 4
        assert torch.allclose(rel_xl.slot_emb[0], sinusoidal(P, D))
        assert args_xl["flow_geo"] == "chamfer"

        _, rel_def, args_def, _ = load_decoder(def_dir, "cpu")
        assert rel_def.sq_dim == 26 and rel_def.deform and rel_def.fourier_freqs == 2
        assert rel_def.flow_shift.shape == (26,) and torch.allclose(rel_def.flow_scale, torch.full((26,), 0.5))
        assert args_def["flow_geo"] == "points"

        # Test multi-model evaluate_occlusion + K=8 mean / best-of-K chamfer / pairwise chamfer spread
        vggt = DummyVGGT()
        gt_rigid = DummyGT()
        gt_def = DummyDeformGT()
        with patch.object(eo_mod, "FrozenVGGT", lambda layer: vggt), \
             patch.object(eo_mod, "ViewPool", lambda p: DummyPool()), \
             patch.object(eo_mod, "GeomTargets", lambda w, c, seed=0: DummyGeom()), \
             patch.object(eo_mod, "GTSuperquadrics", lambda p, deform=False: gt_def if deform else gt_rigid):
            eval_main([
                "--parallel", str(par_dir),
                "--flow", str(xl_dir), str(def_dir),
                "--batch", "2", "--per-n", "2", "--device", "cpu",
                "--out", str(csv_path),
            ])

        assert vggt.calls == (len(BOX_PS) + len(HALF_SIDES)) * len(EVAL_NS)
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        # 5 rows per bucket (parallel + 2*(mean + best_of_8)) * 36 buckets = 180
        assert len(rows) == 180, f"expected 180 rows, got {len(rows)}"
        by_key = {(r["mode"], r["p"], r["side"], r["N"], r["model"], r["stat"]): r for r in rows}
        for model_name in ("flow_geoC100XL", "flow_deform8L"):
            rm = by_key[("box", "0.5", "", "all", model_name, "mean")]
            rb = by_key[("box", "0.5", "", "all", model_name, "best_of_8")]
            assert np.isfinite(float(rm["chamfer_spread_world"])) and float(rm["chamfer_spread_world"]) >= 0.0
            assert np.isfinite(float(rm["chamfer_gt"]))
            assert float(rb["chamfer_world"]) <= float(rm["chamfer_world"]) + 1e-6


def test_augment_occlusion() -> None:
    imgs = torch.zeros(2, 6, 3, 64, 64)
    rng = np.random.default_rng(7)
    # prob=0 leaves tensor unchanged
    clean = augment_occlusion(imgs, 0.0, rng)
    assert torch.equal(clean, imgs)
    # prob=1 masks every view with gray 0.5 and area fraction in [0.25, 0.75]
    occ = augment_occlusion(imgs, 1.0, rng)
    fracs = (occ[:, :, 0] == 0.5).float().mean(dim=(-2, -1))
    assert (fracs >= 0.23).all() and (fracs <= 0.77).all(), f"unexpected box fracs: {fracs}"


if __name__ == "__main__":
    test_occlusion_masks()
    test_load_and_eval_smoke()
    test_current_flow_checkpoints_and_multi_eval()
    test_augment_occlusion()
    print("ALL OCCLUSION EVAL SMOKE TESTS PASSED")
