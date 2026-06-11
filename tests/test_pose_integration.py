# Copyright (c) 2026 webAI, Inc.
"""Integration tests for the pose estimation pipeline.

Tests model building from YAML, forward passes, the YOLO API, CLI parsing,
dataset/model config validity, converter weight-name coverage, the MLX
version pin, and the no-AI-name policy.

Run:
    python -m pytest tests/test_pose_integration.py -v
"""

import re
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon")

from yolo26mlx.nn.modules.head import Pose26
from yolo26mlx.nn.tasks import build_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_DIR / "src" / "yolo26mlx"
POSE_YAML = PACKAGE_DIR / "cfg" / "models" / "26" / "yolo26-pose.yaml"

NK = 17 * 3


# ---------------------------------------------------------------------------
# Model building from YAML
# ---------------------------------------------------------------------------


class TestModelBuild:
    """Verify the pose model builds from YAML across scales."""

    def test_build_nano_pose(self):
        """Build yolo26n-pose; last layer must be Pose26 with kpt_shape (17, 3)."""
        model = build_model(str(POSE_YAML), ch=3, nc=1, verbose=False, scale="n")
        mx.eval(model.parameters())
        head = model.model.layers[-1]
        assert isinstance(head, Pose26), f"Last layer is {type(head)}, expected Pose26"
        assert tuple(head.kpt_shape) == (17, 3)
        assert head.nk == NK

    def test_build_all_scales(self):
        """All five scales should build and end in a Pose26 head."""
        for scale in ["n", "s", "m", "l", "x"]:
            model = build_model(str(POSE_YAML), ch=3, nc=1, verbose=False, scale=scale)
            mx.eval(model.parameters())
            head = model.model.layers[-1]
            assert isinstance(head, Pose26), f"scale={scale}: last layer not Pose26"


# ---------------------------------------------------------------------------
# Forward pass with random weights
# ---------------------------------------------------------------------------


class TestModelForward:
    """Verify forward pass output formats."""

    @pytest.fixture()
    def model(self):
        m = build_model(str(POSE_YAML), ch=3, nc=1, verbose=False, scale="n")
        mx.eval(m.parameters())
        return m

    def test_inference_output_shape(self, model):
        """Inference returns (B, max_det, 6 + nk)."""
        model.eval()
        img = mx.random.normal((1, 640, 640, 3))
        out = model(img)
        mx.eval(out)
        assert out.shape == (1, 300, 6 + NK)

    def test_training_output_dict(self, model):
        """Training returns nested one2many/one2one dicts with keypoints."""
        model.train()
        img = mx.random.normal((2, 640, 640, 3))
        out = model(img)
        assert isinstance(out, dict)
        assert "one2many" in out and "one2one" in out
        for branch in ("one2many", "one2one"):
            assert "keypoints" in out[branch]
            assert "kpts_sigma" in out[branch]
            mx.eval(out[branch]["keypoints"])
            assert out[branch]["keypoints"].shape[0] == 2


# ---------------------------------------------------------------------------
# YOLO API
# ---------------------------------------------------------------------------


class TestYOLOAPI:
    """Verify the high-level YOLO class with pose."""

    def test_build_from_yaml(self):
        from yolo26mlx.engine.model import YOLO

        model = YOLO(str(POSE_YAML), task="pose", verbose=False)
        assert model.task == "pose"
        assert model.model is not None

    def test_auto_detect_task(self):
        """Task should be auto-detected from -pose in the filename."""
        from yolo26mlx.engine.model import YOLO

        model = YOLO(str(POSE_YAML), verbose=False)
        assert model.task == "pose"

    def test_predict_with_random_weights(self, tmp_path):
        """predict() should return Results (keypoints populated when confident)."""
        from PIL import Image

        from yolo26mlx.engine.model import YOLO

        model = YOLO(str(POSE_YAML), task="pose", verbose=False)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / "test_img.jpg"
        Image.fromarray(img).save(str(img_path))

        results = model.predict(str(img_path), conf=0.001)
        assert results is not None and len(results) > 0
        assert results[0].boxes is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Verify the CLI parses pose-related flags."""

    def test_predict_with_task_pose(self):
        from yolo26mlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "predict",
                "--model",
                "models/yolo26n-pose.npz",
                "--source",
                "images/test.jpg",
                "--task",
                "pose",
            ]
        )
        assert args.task == "pose"
        assert args.model == "models/yolo26n-pose.npz"

    def test_train_with_task_pose(self):
        from yolo26mlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "--model",
                "models/yolo26n-pose.npz",
                "--data",
                "coco8-pose",
                "--task",
                "pose",
                "--epochs",
                "2",
            ]
        )
        assert args.task == "pose"
        assert args.data == "coco8-pose"


# ---------------------------------------------------------------------------
# Dataset / model config validity
# ---------------------------------------------------------------------------


class TestConfigs:
    """Verify pose model + dataset configs exist and are valid."""

    def test_pose_model_yaml_content(self):
        """yolo26-pose.yaml: nc=1, kpt_shape, end2end, Pose26 head."""
        import yaml

        assert POSE_YAML.exists(), f"Missing: {POSE_YAML}"
        with open(POSE_YAML) as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 1
        assert list(cfg["kpt_shape"]) == [17, 3]
        head_layers = cfg.get("head", [])
        assert any("Pose26" in str(layer) for layer in head_layers), "No Pose26 in head"

    def test_package_coco8_pose_yaml(self):
        """Package cfg/datasets/coco8-pose.yaml: nc=1, person, kpt_shape, flip_idx."""
        import yaml

        cfg_path = PACKAGE_DIR / "cfg" / "datasets" / "coco8-pose.yaml"
        assert cfg_path.exists(), f"Missing: {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 1
        assert list(cfg["kpt_shape"]) == [17, 3]
        assert "flip_idx" in cfg and len(cfg["flip_idx"]) == 17
        assert cfg["names"][0] == "person"

    def test_script_coco8_pose_yaml(self):
        """Script-side configs/coco8-pose.yaml mirrors the package config."""
        import yaml

        cfg_path = PROJECT_DIR / "configs" / "coco8-pose.yaml"
        assert cfg_path.exists(), f"Missing: {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 1
        assert list(cfg["kpt_shape"]) == [17, 3]
        assert len(cfg["flip_idx"]) == 17


# ---------------------------------------------------------------------------
# Weight converter patterns
# ---------------------------------------------------------------------------


class TestConverterPatterns:
    """Verify the converter recognizes Pose26 keypoint-head weight names."""

    def test_cv4_feature_extractor(self):
        from yolo26mlx.converters.convert import is_conv_weight

        assert is_conv_weight("model.23.cv4.0.0.conv.weight", (64, 64, 3, 3))
        assert is_conv_weight("model.23.one2one_cv4.0.0.conv.weight", (64, 64, 3, 3))

    def test_cv4_kpts_and_sigma(self):
        from yolo26mlx.converters.convert import is_conv_weight

        assert is_conv_weight("model.23.cv4_kpts.0.weight", (51, 64, 1, 1))
        assert is_conv_weight("model.23.cv4_sigma.0.weight", (34, 64, 1, 1))
        assert is_conv_weight("model.23.one2one_cv4_kpts.1.weight", (51, 64, 1, 1))
        assert is_conv_weight("model.23.one2one_cv4_sigma.2.weight", (34, 64, 1, 1))


# ---------------------------------------------------------------------------
# MLX version pin (regression guard)
# ---------------------------------------------------------------------------


class TestMLXPin:
    """Ensure the mlx upper bound is never silently dropped."""

    def test_pyproject_pins_mlx(self):
        text = (PROJECT_DIR / "pyproject.toml").read_text()
        assert "mlx>=0.30.3,<0.31" in text, "mlx pin >=0.30.3,<0.31 missing from pyproject.toml"


# ---------------------------------------------------------------------------
# No AI / tool-name references
# ---------------------------------------------------------------------------


class TestNoAINames:
    """Pose deliverables must not reference AI assistants/tools or imply AI authorship."""

    BANNED = re.compile(
        r"\b(claude|anthropic|openai|chatgpt|copilot|codeium|gpt-?[0-9]|"
        r"llama|gemini|cursor|midjourney)\b"
        r"|ai[-\s]generated|ai[-\s]assisted|generated by ai|written by an ai|co-authored",
        re.IGNORECASE,
    )

    def _files(self):
        files = [
            PROJECT_DIR / "GUIDE_POSE.md",
            PROJECT_DIR / "configs" / "coco8-pose.yaml",
            PACKAGE_DIR / "cfg" / "datasets" / "coco8-pose.yaml",
            POSE_YAML,
            PROJECT_DIR / "tests" / "test_pose.py",
        ]
        # NOTE: this file is intentionally excluded — it defines the banned-token
        # detection regex, so it necessarily contains those tokens as literals.
        files += sorted((PROJECT_DIR / "scripts").glob("benchmark_yolo26_pose_*.py"))
        files += sorted((PROJECT_DIR / "scripts").glob("*_pose_*.py"))
        return [f for f in dict.fromkeys(files) if f.exists()]

    def test_no_banned_terms(self):
        offenders = []
        for f in self._files():
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                m = self.BANNED.search(line)
                if m:
                    offenders.append(f"{f.name}:{i}: {m.group(0)!r}")
        assert not offenders, "Banned AI/tool references found:\n" + "\n".join(offenders)
