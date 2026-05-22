"""Tests for YOLO26 pose inference support."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon")

from yolo26mlx.engine.model import YOLO  # noqa: E402
from yolo26mlx.engine.predictor import Predictor  # noqa: E402
from yolo26mlx.nn.modules.head import Pose26  # noqa: E402
from yolo26mlx.nn.tasks import build_model  # noqa: E402


def test_pose26_config_builds_nano_head():
    """The pose YAML should build a nano Pose26 model."""
    model = build_model("yolo26-pose.yaml", verbose=False, scale="n")
    head = model.model[-1]
    assert isinstance(head, Pose26)
    assert model.nc == 1
    assert model.kpt_shape == (17, 3)
    assert head.nk == 51


def test_pose26_forward_shape():
    """Pose26 inference should return boxes, score, class, and keypoints."""
    head = Pose26(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=(64, 128, 256))
    head.eval()
    x = [
        mx.random.normal((1, 80, 80, 64)),
        mx.random.normal((1, 40, 40, 128)),
        mx.random.normal((1, 20, 20, 256)),
    ]
    out = head(x)
    mx.eval(out)
    assert out.shape == (1, 300, 57)


def test_pose26_weight_name_mapping():
    """Pose26-specific PyTorch names should map to MLX parameter names."""
    yolo = YOLO.__new__(YOLO)
    assert (
        yolo._map_pytorch_to_mlx_name("model.23.cv4_kpts.0.weight")
        == "model.layers.23.cv4_kpts.layer0.weight"
    )
    assert (
        yolo._map_pytorch_to_mlx_name("model.23.cv4_sigma.2.bias")
        == "model.layers.23.cv4_sigma.layer2.bias"
    )
    assert (
        yolo._map_pytorch_to_mlx_name("model.23.one2one_cv4_kpts.1.weight")
        == "model.layers.23.one2one_cv4_kpts.layer1.weight"
    )


def test_pose_postprocess_returns_scaled_keypoints():
    """Pose postprocess should filter detections and undo letterbox scaling."""

    class DummyModel:
        kpt_shape = (17, 3)

        def eval(self):
            return None

    predictor = Predictor(model=DummyModel(), task="pose")
    pred = np.zeros((2, 57), dtype=np.float32)
    pred[0, :6] = [50, 60, 20, 20, 0.9, 0]
    pred[0, 6:] = np.tile([20, 30, 0.8], 17)
    pred[1, :6] = [50, 60, 20, 20, 0.1, 0]
    pred[1, 6:] = np.tile([5, 5, 0.2], 17)

    boxes, keypoints = predictor._postprocess_pose(
        pred,
        orig_shape=(100, 200),
        letterbox_info={"ratio": 0.5, "dw": 10.0, "dh": 5.0},
        conf=0.25,
    )

    assert boxes.data.shape == (1, 6)
    assert keypoints.data.shape == (1, 17, 3)
    np.testing.assert_allclose(keypoints.data[0, 0], [20, 50, 0.8], atol=1e-5)
