"""Unit tests for optical-flow helpers (no GPU, no full Phi-4 stack)."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"


def _load(rel_path: str, name: str):
    path = _SRC / "drivevlms" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_flow_io = _load(Path("utils") / "flow_io.py", "flow_io_mod")
_siglip = _load(Path("utils") / "siglip_expand.py", "siglip_expand_mod")
flow_npz_path_for_image = _flow_io.flow_npz_path_for_image
load_flow_uv_tensor = _flow_io.load_flow_uv_tensor
expand_siglip_vision_patch_in_channels = _siglip.expand_siglip_vision_patch_in_channels
sys.path.insert(0, str(_SRC))
from drivevlms.collate_fn.drivelm_nus_phi4 import format_prompt_phi4  # noqa: E402


class TestFlowIO(unittest.TestCase):
    def test_flow_npz_path_for_image(self) -> None:
        p = "/data/nuscenes/samples/CAM_FRONT/foo.jpg"
        out = flow_npz_path_for_image(p, "/flow/root")
        self.assertTrue(out.endswith(os.path.join("CAM_FRONT", "foo.npz")))

    def test_load_flow_scales_uv(self) -> None:
        h, w = 4, 4
        u = np.full((h, w), 10.0, dtype=np.float32)
        v = np.full((h, w), 24.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "flow")
            os.makedirs(os.path.join(root, "CAM_FRONT"))
            img_path = "/dummy/samples/CAM_FRONT/bar.jpg"
            npz = os.path.join(root, "CAM_FRONT", "bar.npz")
            np.savez_compressed(npz, u=u, v=v, valid=np.array(True))
            t = load_flow_uv_tensor(
                img_path, root, h, w, 2.0, torch.float32, torch.device("cpu")
            )
        self.assertEqual(tuple(t.shape), (2, h, w))
        self.assertTrue(torch.allclose(t[0], torch.full((h, w), 5.0)))
        self.assertTrue(torch.allclose(t[1], torch.full((h, w), 12.0)))

    def test_missing_npz_is_zero_flow(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "flow")
            os.makedirs(os.path.join(root, "CAM_FRONT"))
            img_path = "/x/samples/CAM_FRONT/nope.jpg"
            t = load_flow_uv_tensor(
                img_path, root, 3, 3, 32.0, torch.float32, torch.device("cpu")
            )
        self.assertTrue(torch.all(t == 0))

    def test_valid_false_ignores_stored_uv(self) -> None:
        h, w = 2, 2
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "flow")
            os.makedirs(os.path.join(root, "CAM_BACK"))
            img_path = "/x/samples/CAM_BACK/x.jpg"
            npz = os.path.join(root, "CAM_BACK", "x.npz")
            np.savez_compressed(
                npz,
                u=np.ones((h, w), np.float32) * 99.0,
                v=np.ones((h, w), np.float32) * 99.0,
                valid=np.array(False),
            )
            t = load_flow_uv_tensor(
                img_path, root, h, w, 1.0, torch.float32, torch.device("cpu")
            )
        self.assertTrue(torch.all(t == 0))

    def test_zero_flow_scale_errors(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "flow")
            os.makedirs(os.path.join(root, "CAM_FRONT"))
            img_path = "/x/samples/CAM_FRONT/x.jpg"
            np.savez_compressed(
                os.path.join(root, "CAM_FRONT", "x.npz"),
                u=np.zeros((1, 1), np.float32),
                v=np.zeros((1, 1), np.float32),
                valid=True,
            )
            with self.assertRaises(ValueError):
                load_flow_uv_tensor(
                    img_path, root, 1, 1, 0.0, torch.float32, torch.device("cpu")
                )


class TestFlowInjectBroadcast(unittest.TestCase):
    """Regression: rgb can be [N_crop, 3, H, W] with N_crop>1 (matches Phi-4 processor)."""

    def test_cat_rgb_uv_matches_multi_crop(self) -> None:
        rgb = torch.zeros(2, 3, 10, 10)
        uv = torch.ones(2, 10, 10)
        nc = rgb.shape[0]
        uv_b = uv.unsqueeze(0).expand(nc, -1, -1, -1).contiguous()
        merged = torch.cat([rgb, uv_b], dim=1)
        self.assertEqual(tuple(merged.shape), (2, 5, 10, 10))


class TestSiglipExpand(unittest.TestCase):
    def test_expand_five_channels(self) -> None:
        import torch.nn as nn

        class FakeVision(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.Module()
                self.embeddings.patch_embedding = nn.Conv2d(3, 8, kernel_size=14, stride=14)
                self.config = type("C", (), {"num_channels": 3})()

        class FakeImageEmbed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.img_processor = FakeVision()

        class FakeExtend(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_embed = FakeImageEmbed()

        class FakeInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens_extend = FakeExtend()

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = FakeInner()

        m = FakeModel()
        expand_siglip_vision_patch_in_channels(m, 5)
        conv = m.model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding
        self.assertEqual(conv.in_channels, 5)
        self.assertEqual(conv.weight.shape[1], 5)
        self.assertTrue(torch.all(conv.weight[:, 3:] == 0))
        expand_siglip_vision_patch_in_channels(m, 5)
        self.assertEqual(conv.in_channels, 5)


class TestFlowImagePrompt(unittest.TestCase):
    def test_prompt_expands_to_12_image_slots_when_flow_enabled(self) -> None:
        p = format_prompt_phi4("x", include_flow_images=True)
        self.assertIn("<|image_12|>", p)
        self.assertIn("<|image_7|>", p)

    def test_prompt_keeps_6_image_slots_without_flow(self) -> None:
        p = format_prompt_phi4("x", include_flow_images=False)
        self.assertIn("<|image_6|>", p)
        self.assertNotIn("<|image_7|>", p)


if __name__ == "__main__":
    unittest.main()
