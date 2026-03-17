# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Results Class - Pure MLX Implementation

Results container for inference outputs.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw


class Boxes:
    """Container for detection bounding boxes.

    Supports YOLO26 end-to-end detection outputs.
    """

    def __init__(self, boxes: mx.array | np.ndarray, orig_shape: tuple[int, int]):
        """Initialize Boxes.

        Args:
            boxes: Bounding boxes array with shape (N, 6) containing
                   [x1, y1, x2, y2, conf, cls]
            orig_shape: Original image shape (height, width)
        """
        if isinstance(boxes, mx.array):
            boxes = np.array(boxes)

        self.data = boxes  # (N, 6) - x1, y1, x2, y2, conf, cls
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> np.ndarray:
        """Bounding boxes in xyxy format (x1, y1, x2, y2)."""
        return self.data[:, :4] if len(self.data) > 0 else np.empty((0, 4))

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes in xywh format (cx, cy, w, h)."""
        if len(self.data) == 0:
            return np.empty((0, 4))
        boxes = self.xyxy
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def conf(self) -> np.ndarray:
        """Detection confidence scores for each box."""
        return self.data[:, 4] if len(self.data) > 0 else np.empty((0,))

    @property
    def cls(self) -> np.ndarray:
        """Class index for each detected box."""
        return self.data[:, 5] if len(self.data) > 0 else np.empty((0,))

    def __len__(self) -> int:
        """Return the number of detected boxes."""
        return len(self.data)

    def __repr__(self) -> str:
        """Return string representation of Boxes with count and shape info."""
        return f"Boxes(n={len(self)}, orig_shape={self.orig_shape})"


class Masks:
    """Container for segmentation masks."""

    def __init__(self, masks: mx.array | np.ndarray, orig_shape: tuple[int, int]):
        """Initialize Masks.

        Args:
            masks: Segmentation masks with shape (N, H, W)
            orig_shape: Original image shape (height, width)
        """
        if isinstance(masks, mx.array):
            masks = np.array(masks)

        self.data = masks
        self.orig_shape = orig_shape

    @property
    def masks(self) -> np.ndarray:
        """Segmentation mask array with shape (N, H, W)."""
        return self.data

    def __len__(self) -> int:
        """Return the number of segmentation masks."""
        return len(self.data) if self.data is not None else 0

    def __repr__(self) -> str:
        """Return string representation of Masks with count and shape info."""
        return f"Masks(n={len(self)}, shape={self.data.shape if self.data is not None else None})"


class Keypoints:
    """Container for pose keypoints."""

    def __init__(self, keypoints: mx.array | np.ndarray, orig_shape: tuple[int, int]):
        """Initialize Keypoints.

        Args:
            keypoints: Keypoints array with shape (N, K, 3) where K is number of keypoints
                       and 3 is (x, y, visibility)
            orig_shape: Original image shape (height, width)
        """
        if isinstance(keypoints, mx.array):
            keypoints = np.array(keypoints)

        self.data = keypoints
        self.orig_shape = orig_shape

    @property
    def xy(self) -> np.ndarray:
        """Keypoint x, y coordinates with shape (N, K, 2)."""
        return self.data[..., :2] if self.data is not None else None

    @property
    def conf(self) -> np.ndarray:
        """Keypoint visibility/confidence scores with shape (N, K)."""
        return self.data[..., 2] if self.data is not None else None

    def __len__(self) -> int:
        """Return the number of detected persons (keypoint sets)."""
        return len(self.data) if self.data is not None else 0

    def __repr__(self) -> str:
        """Return string representation of Keypoints with count and shape info."""
        return (
            f"Keypoints(n={len(self)}, shape={self.data.shape if self.data is not None else None})"
        )


class OBB:
    """Container for oriented bounding boxes."""

    def __init__(self, obb: mx.array | np.ndarray, orig_shape: tuple[int, int]):
        """Initialize OBB.

        Args:
            obb: OBB array with shape (N, 7) containing
                 [cx, cy, w, h, angle, conf, cls]
            orig_shape: Original image shape (height, width)
        """
        if isinstance(obb, mx.array):
            obb = np.array(obb)

        self.data = obb
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> np.ndarray:
        """OBB in xywhr format (cx, cy, w, h, rotation)."""
        return self.data[:, :5] if len(self.data) > 0 else np.empty((0, 5))

    @property
    def conf(self) -> np.ndarray:
        """Confidence score for each oriented bounding box."""
        return self.data[:, 5] if len(self.data) > 0 else np.empty((0,))

    @property
    def cls(self) -> np.ndarray:
        """Class index for each oriented bounding box."""
        return self.data[:, 6] if len(self.data) > 0 else np.empty((0,))

    def __len__(self) -> int:
        """Return the number of oriented bounding boxes."""
        return len(self.data)

    def __repr__(self) -> str:
        """Return string representation of OBB with count and shape info."""
        return f"OBB(n={len(self)}, orig_shape={self.orig_shape})"


class Results:
    """Results container for YOLO26 inference outputs.

    Unified container for detection, segmentation, pose, and OBB results.
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str = "",
        names: dict[int, str] | None = None,
        boxes: Boxes | None = None,
        masks: Masks | None = None,
        keypoints: Keypoints | None = None,
        obb: OBB | None = None,
    ):
        """Initialize Results.

        Args:
            orig_img: Original input image
            path: Image path
            names: Class names dict {id: name}
            boxes: Detection boxes
            masks: Segmentation masks
            keypoints: Pose keypoints
            obb: Oriented bounding boxes
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names or {}
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.obb = obb

    def __len__(self) -> int:
        """Return the number of detections (boxes or OBBs)."""
        if self.boxes is not None:
            return len(self.boxes)
        elif self.obb is not None:
            return len(self.obb)
        return 0

    def __repr__(self) -> str:
        """Return string representation of Results with detection counts."""
        return (
            f"Results(path='{self.path}', "
            f"boxes={len(self.boxes) if self.boxes else 0}, "
            f"masks={len(self.masks) if self.masks else 0}, "
            f"keypoints={len(self.keypoints) if self.keypoints else 0}, "
            f"obb={len(self.obb) if self.obb else 0})"
        )

    def plot(self) -> np.ndarray:
        """Plot detections on image.

        Draws axis-aligned boxes and labels for detection outputs.

        Returns:
            Annotated image as numpy array.
        """
        pil_img = Image.fromarray(self.orig_img.copy())
        draw = ImageDraw.Draw(pil_img)

        # Fallback COCO names to keep labels readable even if metadata is missing.
        coco80 = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        def class_name_for(cls_id: int) -> str:
            def is_generic(name: str) -> bool:
                s = str(name).strip().lower()
                return s.startswith("class") or s.startswith("cls")

            if isinstance(self.names, dict):
                # Some models store keys as strings; support both.
                if cls_id in self.names:
                    candidate = str(self.names[cls_id])
                    if not is_generic(candidate):
                        return candidate
                if str(cls_id) in self.names:
                    candidate = str(self.names[str(cls_id)])
                    if not is_generic(candidate):
                        return candidate
            elif isinstance(self.names, (list, tuple)):
                if 0 <= cls_id < len(self.names):
                    candidate = str(self.names[cls_id])
                    if not is_generic(candidate):
                        return candidate
            if 0 <= cls_id < len(coco80):
                return coco80[cls_id]
            return f"class{cls_id}"

        if self.boxes is not None and len(self.boxes) > 0:
            for i, box in enumerate(self.boxes.xyxy):
                x1, y1, x2, y2 = [int(v) for v in box]
                conf = float(self.boxes.conf[i])
                cls_id = int(self.boxes.cls[i])
                cls_name = class_name_for(cls_id)
                label = f"{cls_name}: {conf:.2f}"

                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
                text_bbox = draw.textbbox((x1, y1), label)
                tw = max(1, text_bbox[2] - text_bbox[0])
                th = max(1, text_bbox[3] - text_bbox[1])
                text_x = max(0, x1)
                # Place label above the box; fallback inside image bounds if needed.
                text_y = y1 - th - 4
                if text_y < 0:
                    text_y = 0
                draw.rectangle(
                    [(text_x, text_y), (text_x + tw + 6, text_y + th + 4)],
                    fill=(255, 0, 0),
                )
                draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255))

        return np.array(pil_img)

    def save(self, filename: str | None = None) -> str:
        """Save annotated image to file and return output path.

        Args:
            filename: Output file path. If not provided, writes to
                `results/<input_stem>_result.jpg` (or `results/result.jpg`).

        Returns:
            Saved image path as a string.
        """
        img = self.plot()
        if filename is None:
            stem = Path(self.path).stem if self.path else "result"
            out_path = Path("results") / f"{stem}_result.jpg"
        else:
            out_path = Path(filename)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_path)
        return str(out_path)

    def to_json(self) -> dict[str, Any]:
        """Convert results to JSON-serializable dict.

        Returns:
            Dictionary with path, shape, boxes, masks, keypoints, and OBB data.
        """
        result = {
            "path": self.path,
            "orig_shape": self.orig_shape,
        }

        if self.boxes is not None and len(self.boxes) > 0:
            result["boxes"] = {
                "xyxy": self.boxes.xyxy.tolist(),
                "conf": self.boxes.conf.tolist(),
                "cls": self.boxes.cls.tolist(),
            }

        if self.masks is not None and len(self.masks) > 0:
            result["masks_shape"] = self.masks.data.shape

        if self.keypoints is not None and len(self.keypoints) > 0:
            result["keypoints"] = {
                "xy": self.keypoints.xy.tolist(),
                "conf": self.keypoints.conf.tolist(),
            }

        if self.obb is not None and len(self.obb) > 0:
            result["obb"] = {
                "xywhr": self.obb.xywhr.tolist(),
                "conf": self.obb.conf.tolist(),
                "cls": self.obb.cls.tolist(),
            }

        return result

    def verbose(self) -> str:
        """Return verbose string with image info and top detection details."""
        s = f"Image: {self.path or 'N/A'} ({self.orig_shape[1]}x{self.orig_shape[0]})\n"

        if self.boxes is not None and len(self.boxes) > 0:
            s += f"  Detections: {len(self.boxes)}\n"
            for i in range(min(len(self.boxes), 5)):
                cls_id = int(self.boxes.cls[i])
                cls_name = self.names.get(cls_id, f"class{cls_id}")
                conf = self.boxes.conf[i]
                s += f"    - {cls_name}: {conf:.2f}\n"
            if len(self.boxes) > 5:
                s += f"    ... and {len(self.boxes) - 5} more\n"

        return s
