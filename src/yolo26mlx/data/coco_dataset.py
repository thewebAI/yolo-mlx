# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
COCO Dataset Loader for YOLO26 MLX

Loads COCO val2017 images and annotations for evaluation.
"""

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class COCODataset:
    """COCO dataset loader for YOLO26 MLX validation.

    Attributes:
        root: Path to COCO dataset root
        split: Dataset split ('val2017', 'train2017')
        images: List of image info dicts
        annotations: Dict mapping image_id to list of annotations
        categories: Dict mapping category_id to category info
        class_names: List of class names (80 for COCO)
    """

    # COCO category IDs to contiguous 0-79 mapping
    # COCO has 91 categories but only 80 are used
    COCO_IDS = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    def __init__(
        self,
        root: str,
        split: str = "val2017",
        img_size: int = 640,
        augment: bool = False,
    ):
        """Initialize COCO dataset.

        Args:
            root: Path to COCO dataset root directory
            split: Dataset split ('val2017' or 'train2017')
            img_size: Target image size for preprocessing
            augment: Apply training augmentations (HSV jitter, horizontal flip)
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # Paths
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        self.annotations_file = self.root / "annotations" / f"instances_{split}.json"

        # Build COCO ID to index mapping
        self.coco_id_to_idx = {cid: idx for idx, cid in enumerate(self.COCO_IDS)}

        # Load annotations
        self.images = []
        self.annotations = {}
        self.categories = {}
        self.class_names = []

        if self.annotations_file.exists():
            self._load_annotations()
        else:
            # Fall back to image directory listing
            self._load_from_directory()

    def _load_annotations(self):
        """Load COCO JSON annotations file and populate self.images, self.annotations, and self.categories."""
        logger.info(f"Loading COCO annotations from {self.annotations_file}...")

        with open(self.annotations_file) as f:
            data = json.load(f)

        # Store images
        self.images = data["images"]

        # Build image_id to annotations mapping
        self.annotations = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Store categories
        self.categories = {cat["id"]: cat for cat in data["categories"]}

        # Build class names - only for categories that exist
        self.class_names = []
        for cid in self.COCO_IDS:
            if cid in self.categories:
                self.class_names.append(self.categories[cid]["name"])
            else:
                self.class_names.append(f"class_{cid}")

        logger.info(
            f"  Loaded {len(self.images)} images with {len(data['annotations'])} annotations"
        )
        logger.info(f"  {len(self.categories)} categories")

    def _load_from_directory(self):
        """Scan image directory and load YOLO-format labels."""
        logger.info(f"Loading images from {self.images_dir}...")

        # Support multiple image formats
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_files.extend(self.images_dir.glob(ext))
            image_files.extend(self.images_dir.glob(ext.upper()))
        image_files = sorted(set(image_files))  # Remove duplicates and sort

        for img_path in image_files:
            # Try to extract image ID from filename
            try:
                img_id = int(img_path.stem)
            except ValueError:
                # Use hash of filename as ID if not numeric
                img_id = hash(img_path.stem) & 0x7FFFFFFF

            self.images.append({"id": img_id, "file_name": img_path.name, "width": 0, "height": 0})

            # Load YOLO format labels if available
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self._load_yolo_labels(img_id, label_path)

        logger.info(f"  Found {len(self.images)} images")
        logger.info(f"  Found {len(self.annotations)} images with annotations")

    def _load_yolo_labels(self, img_id: int, label_path: Path):
        """Load YOLO format labels for an image.

        YOLO format: class_id x_center y_center width height (normalized 0-1)

        Args:
            img_id: Image ID
            label_path: Path to .txt label file
        """
        annotations = []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Store as normalized [x, y, w, h] for top-left corner
                    # These are NORMALIZED to [0,1], not pixel coordinates
                    # We'll convert to proper pixel coords in __getitem__ using actual image size
                    x = x_center - width / 2
                    y = y_center - height / 2
                    w = width
                    h = height

                    # For YOLO format, class_id is already the 0-79 index.
                    # Mark as yolo_format so __getitem__ won't remap through COCO IDs.
                    annotations.append(
                        {
                            "bbox": [x, y, w, h],  # Normalized [0-1]
                            "bbox_normalized": True,  # Flag to indicate normalized coords
                            "yolo_format": True,  # class_id is already 0-based index
                            "category_id": class_id,
                            "area": w * h,  # Normalized area (will be scaled later)
                            "iscrowd": 0,
                            "image_id": img_id,
                        }
                    )

        if annotations:
            self.annotations[img_id] = annotations

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[mx.array, dict]:
        """Get image and annotations by index.

        Args:
            idx: Image index

        Returns:
            Tuple of (image_array, annotation_dict)
            - image_array: MLX array (H, W, C) normalized to [0, 1]
            - annotation_dict: Contains 'boxes', 'labels', 'image_id', etc.
        """
        img_info = self.images[idx]
        img_id = img_info["id"]

        # Load image
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize with letterboxing to maintain aspect ratio
        image_resized, ratio, pad = self._letterbox(image, self.img_size)

        # Convert to numpy array (uint8 for augmentation, then float32)
        img_np = np.array(image_resized, dtype=np.uint8)

        # Get annotations (before augmentation so we can flip boxes)
        anns = self.annotations.get(img_id, [])

        # Convert annotations to array format
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Check if coordinates are already normalized (from YOLO format)
            if ann.get("bbox_normalized", False):
                # YOLO format: coordinates are normalized to [0,1]
                # Convert to xyxy (still normalized)
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Apply letterbox transformation
                # For normalized coords, we need to:
                # 1. Scale from normalized original → pixel original
                # 2. Apply letterbox (scale by ratio, add pad)
                # 3. Normalize back to [0,1] in letterboxed space

                # Scale to original pixel coordinates
                x1_px = x1 * orig_w
                y1_px = y1 * orig_h
                x2_px = x2 * orig_w
                y2_px = y2 * orig_h

                # Apply letterbox transformation
                x1 = (x1_px * ratio + pad[0]) / self.img_size
                y1 = (y1_px * ratio + pad[1]) / self.img_size
                x2 = (x2_px * ratio + pad[0]) / self.img_size
                y2 = (y2_px * ratio + pad[1]) / self.img_size
            else:
                # COCO format: pixel coordinates
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Scale to resized image coordinates
                x1 = (x1 * ratio + pad[0]) / self.img_size
                y1 = (y1 * ratio + pad[1]) / self.img_size
                x2 = (x2 * ratio + pad[0]) / self.img_size
                y2 = (y2 * ratio + pad[1]) / self.img_size

            # Clip to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

            boxes.append([x1, y1, x2, y2])

            # For COCO dataset, convert category ID to 0-79 index
            # For YOLO format labels, use ID directly (already 0-based)
            cat_id = ann["category_id"]
            if ann.get("yolo_format", False):
                # YOLO labels: class_id is already the 0-79 index
                labels.append(cat_id)
            elif cat_id in self.coco_id_to_idx:
                labels.append(self.coco_id_to_idx[cat_id])
            else:
                # Custom dataset - use category_id directly as label
                labels.append(cat_id)

            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        # Create annotation dict
        annotation = {
            "image_id": img_id,
            "boxes": (
                np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            ),
            "labels": (
                np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
            ),
            "areas": (
                np.array(areas, dtype=np.float32) if areas else np.zeros((0,), dtype=np.float32)
            ),
            "iscrowd": (
                np.array(iscrowd, dtype=np.int64) if iscrowd else np.zeros((0,), dtype=np.int64)
            ),
            "orig_size": (orig_h, orig_w),
            "ratio": ratio,
            "pad": pad,
        }

        # Apply augmentations (training only)
        # Ported from ultralytics RandomPerspective + RandomHSV + RandomFlip
        if self.augment:
            img_np, annotation = self._random_affine(img_np, annotation, scale=0.5, translate=0.1)
            img_np = self._augment_hsv(img_np, hgain=0.015, sgain=0.7, vgain=0.4)
            img_np, annotation = self._random_fliplr(img_np, annotation, p=0.5)

        # Convert to MLX array (float32, normalized to [0, 1])
        img_array = mx.array(img_np.astype(np.float32) / 255.0)

        return img_array, annotation

    def _random_affine(
        self, img: np.ndarray, annotation: dict, scale: float = 0.5, translate: float = 0.1
    ) -> tuple:
        """Random scale and translate augmentation.

        Ported from ultralytics/data/augment.py RandomPerspective.
        With default config: degrees=0, shear=0, perspective=0, scale=0.5, translate=0.1

        Scale=0.5 means random scale factor in [0.5, 1.5].
        Translate=0.1 means shift by up to 10% of image size.

        Args:
            img: numpy array (H, W, 3) uint8
            annotation: Dict with 'boxes' in xyxy normalized format [0,1]
            scale: Scale factor range (0.5 means 50%-150%)
            translate: Translation fraction (0.1 means ±10%)

        Returns:
            Tuple of (augmented_img, updated_annotation)
        """
        if not _HAS_CV2:
            return img, annotation

        h, w = img.shape[:2]

        # Random scale in [1-scale, 1+scale]
        s = random.uniform(1 - scale, 1 + scale)

        # Center → Scale → Translate
        # C: translate to center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        # R: rotation (0) + scale
        R = np.eye(3, dtype=np.float32)
        R[0, 0] = s
        R[1, 1] = s

        # T: translate to new center
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

        # Combined matrix: T @ R @ C
        M = T @ R @ C

        # Apply affine transform
        img = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=(114, 114, 114))

        # Transform bounding boxes
        boxes = annotation["boxes"]
        if len(boxes) > 0:
            # boxes are in normalized [0,1] xyxy format
            # Convert to pixel coords
            bboxes = boxes.copy()
            bboxes[:, [0, 2]] *= w  # x coords
            bboxes[:, [1, 3]] *= h  # y coords

            n = len(bboxes)
            # Get 4 corners of each box: x1y1, x2y2, x1y2, x2y1
            xy = np.ones((n * 4, 3), dtype=np.float32)
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2]  # transform
            xy = xy.reshape(n, 8)

            # Find new bounding boxes from transformed corners
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_bboxes = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)

            # Clip to image bounds
            new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, w)
            new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, h)

            # Filter: keep boxes with min width/height and reasonable aspect ratio
            # Matches PyTorch box_candidates(wh_thr=2, ar_thr=100, area_thr=0.10)
            orig_w_box = bboxes[:, 2] - bboxes[:, 0]
            orig_h_box = bboxes[:, 3] - bboxes[:, 1]
            new_w = new_bboxes[:, 2] - new_bboxes[:, 0]
            new_h = new_bboxes[:, 3] - new_bboxes[:, 1]
            eps = 1e-16
            orig_area = orig_w_box * orig_h_box * s  # scaled original area
            new_area = new_w * new_h
            ar = np.maximum(new_w / (new_h + eps), new_h / (new_w + eps))
            keep = (new_w > 2) & (new_h > 2) & (new_area / (orig_area + eps) > 0.1) & (ar < 100)

            # Normalize back to [0,1]
            new_bboxes[:, [0, 2]] /= w
            new_bboxes[:, [1, 3]] /= h

            annotation = dict(annotation)  # shallow copy
            annotation["boxes"] = new_bboxes[keep].astype(np.float32)
            annotation["labels"] = annotation["labels"][keep]
            if "areas" in annotation:
                annotation["areas"] = annotation["areas"][keep]
            if "iscrowd" in annotation:
                annotation["iscrowd"] = annotation["iscrowd"][keep]

        return img, annotation

    def _augment_hsv(
        self, img: np.ndarray, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4
    ) -> np.ndarray:
        """Random HSV augmentation. Ported from ultralytics/data/augment.py RandomHSV.

        Args:
            img: numpy array (H, W, 3) uint8 RGB
            hgain: Hue gain fraction
            sgain: Saturation gain fraction
            vgain: Value gain fraction

        Returns:
            Augmented image (H, W, 3) uint8
        """
        if not _HAS_CV2:
            return img
        if not (hgain or sgain or vgain):
            return img

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(img.dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(img.dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(img.dtype)
        lut_sat[0] = 0  # prevent pure white changing color

        # Convert RGB → BGR for cv2, then BGR → HSV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hue, sat, val = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _random_fliplr(
        self, img: np.ndarray, annotation: dict, p: float = 0.5
    ) -> tuple[np.ndarray, dict]:
        """Random horizontal flip. Ported from ultralytics/data/augment.py RandomFlip.

        Args:
            img: numpy array (H, W, 3) uint8
            annotation: Dict with 'boxes' in xyxy normalized format
            p: Flip probability

        Returns:
            Tuple of (flipped_img, updated_annotation)
        """
        if random.random() < p:
            img = np.ascontiguousarray(np.fliplr(img))
            boxes = annotation["boxes"]
            if len(boxes) > 0:
                # Flip xyxy boxes horizontally: new_x1 = 1 - old_x2, new_x2 = 1 - old_x1
                boxes_flipped = boxes.copy()
                boxes_flipped[:, 0] = 1.0 - boxes[:, 2]  # new x1 = 1 - old x2
                boxes_flipped[:, 2] = 1.0 - boxes[:, 0]  # new x2 = 1 - old x1
                annotation["boxes"] = boxes_flipped
        return img, annotation

    def _letterbox(
        self, image: Image.Image, target_size: int, color: tuple[int, int, int] = (114, 114, 114)
    ) -> tuple[Image.Image, float, tuple[float, float]]:
        """Resize image with letterboxing to maintain aspect ratio.

        Args:
            image: PIL Image
            target_size: Target size (square)
            color: Padding color

        Returns:
            Tuple of (resized_image, scale_ratio, (pad_x, pad_y))
        """
        w, h = image.size

        # Compute scale factor
        ratio = min(target_size / h, target_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # Compute padding
        pad_w = (target_size - new_w) / 2
        pad_h = (target_size - new_h) / 2

        # Resize image
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)

        # Create padded image
        padded = Image.new("RGB", (target_size, target_size), color)
        padded.paste(image_resized, (int(pad_w), int(pad_h)))

        return padded, ratio, (pad_w, pad_h)

    def get_dataloader(
        self, batch_size: int = 16, shuffle: bool = False
    ) -> Iterator[tuple[mx.array, list[dict]]]:
        """Create a batch iterator.

        Args:
            batch_size: Number of images per batch.
            shuffle: Whether to randomly order the images each epoch.

        Returns:
            Iterator yielding (image_batch, annotation_list) tuples.
        """
        indices = list(range(len(self)))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            images = []
            annotations = []

            for idx in batch_indices:
                img, ann = self[idx]
                images.append(img)
                annotations.append(ann)

            # Stack images into batch
            batch_images = mx.stack(images, axis=0)

            yield batch_images, annotations


class COCOResultsWriter:
    """Write detection results in COCO JSON format for evaluation."""

    def __init__(self):
        """Initialize an empty results writer."""
        self.results = []

    def add_predictions(
        self,
        image_id: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        orig_size: tuple[int, int],
        ratio: float,
        pad: tuple[float, float],
        img_size: int = 640,
    ):
        """Add predictions for an image.

        Args:
            image_id: COCO image ID
            boxes: Detection boxes in xyxy format, normalized [0, 1]
            scores: Confidence scores
            labels: Class indices (0-79)
            orig_size: Original image size (height, width)
            ratio: Letterbox ratio
            pad: Letterbox padding (pad_x, pad_y)
            img_size: Input image size
        """
        orig_h, orig_w = orig_size

        for i in range(len(boxes)):
            # Convert normalized coords to pixel coords in letterboxed image
            x1, y1, x2, y2 = boxes[i] * img_size

            # Remove letterbox padding and scale back to original
            x1 = (x1 - pad[0]) / ratio
            y1 = (y1 - pad[1]) / ratio
            x2 = (x2 - pad[0]) / ratio
            y2 = (y2 - pad[1]) / ratio

            # Clip to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            # Convert to COCO format [x, y, width, height]
            w = x2 - x1
            h = y2 - y1

            # Skip boxes with non-positive dimensions (completely outside image)
            if w <= 0 or h <= 0:
                continue

            # Convert class index to COCO category ID
            category_id = COCODataset.COCO_IDS[int(labels[i])]

            self.results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i]),
                }
            )

    def save(self, output_path: str):
        """Save results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
            f.write("\n")
        logger.info(f"Saved {len(self.results)} detections to {output_path}")

    def get_results(self) -> list[dict]:
        """Return list of detection result dicts in COCO JSON format."""
        return self.results
