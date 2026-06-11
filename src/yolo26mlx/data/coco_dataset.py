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


def polygon2mask(
    imgsz: tuple[int, int],
    polygon: np.ndarray | list[np.ndarray],
    downsample_ratio: int = 1,
) -> np.ndarray:
    """Rasterize polygon contour(s) to a binary mask.

    Reference: ultralytics data/utils.py polygon2mask

    Args:
        imgsz: Image size as (height, width).
        polygon: Single polygon (K, 2) or list of polygon arrays for
                 multi-part annotations (e.g. occluded objects).
        downsample_ratio: Downsample factor for the output mask.

    Returns:
        Binary mask of shape (H // downsample_ratio, W // downsample_ratio).
    """
    if not _HAS_CV2:
        h, w = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros(imgsz, dtype=np.uint8)
    if isinstance(polygon, list):
        contours = [np.array(p, dtype=np.int32) for p in polygon]
    else:
        contours = [np.array(polygon, dtype=np.int32)]
    cv2.fillPoly(mask, contours, color=1)

    if downsample_ratio > 1:
        h, w = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def polygons2masks_overlap(
    imgsz: tuple[int, int],
    segments: list[np.ndarray],
    downsample_ratio: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an overlap mask from multiple polygon segments.

    Segments must be pre-sorted by descending area by the caller.
    Pixels are painted with 1-based instance IDs; later (smaller) instances
    overwrite earlier (larger) ones in overlap regions.

    Reference: ultralytics data/utils.py polygons2masks_overlap

    Args:
        imgsz: Image size as (height, width).
        segments: List of polygon arrays, each (K, 2) in pixel coords,
                  pre-sorted by descending area.
        downsample_ratio: Downsample factor for the output mask.

    Returns:
        Tuple of:
        - Overlap mask (H // ds, W // ds) with 1-based instance IDs.
        - Sort index array mapping output instance ID-1 to input index
          (identity when pre-sorted).
    """
    h, w = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
    masks = np.zeros((h, w), dtype=np.int32)

    areas = []
    ms = []
    for seg in segments:
        m = polygon2mask(imgsz, seg, downsample_ratio)
        ms.append(m)
        areas.append(float(m.sum()))

    areas = np.array(areas)
    index = np.argsort(-areas)
    ms_sorted = [ms[i] for i in index]

    for i, m in enumerate(ms_sorted):
        mask = m * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)

    return masks, index


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
        task: str = "detect",
        mask_ratio: int = 4,
        kpt_shape: tuple[int, int] = (17, 3),
        flip_idx: list[int] | None = None,
        rect: bool = False,
        stride: int = 32,
    ):
        """Initialize COCO dataset.

        Args:
            root: Path to COCO dataset root directory
            split: Dataset split ('val2017' or 'train2017')
            img_size: Target image size for preprocessing
            augment: Apply training augmentations (HSV jitter, horizontal flip)
            task: Task type ('detect', 'segment', or 'pose')
            mask_ratio: Downsample ratio for GT masks (default 4, giving 160x160 for 640 input)
            kpt_shape: Keypoint shape (num_keypoints, dims) for pose tasks
            flip_idx: Left/right keypoint swap indices applied on horizontal
                flip. When None for a pose task, horizontal flipping is disabled
                (matches Ultralytics gating ``fliplr=0`` without ``flip_idx``).
            rect: Rectangular letterbox to a per-image stride-aligned canvas
                instead of a square ``img_size`` canvas. Mirrors the Ultralytics
                ``rect=True`` validation protocol (less padding, longest side =
                ``img_size``). When True the dataloader yields one image per batch
                because shapes vary per image.
            stride: Model max stride used to align the rectangular canvas.
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.task = task
        self.mask_ratio = mask_ratio
        self.kpt_shape = tuple(kpt_shape)
        self.flip_idx = list(flip_idx) if flip_idx is not None else None
        self.rect = bool(rect)
        self.stride = int(stride)

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

        Supports both detection and segmentation formats:
        - Detection: cls cx cy w h (5 values)
        - Segmentation: cls x1 y1 x2 y2 x3 y3 ... (>5 values, polygon vertices)

        Args:
            img_id: Image ID
            label_path: Path to .txt label file
        """
        annotations = []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                if self.task == "pose":
                    # Pose format: cls cx cy w h px1 py1 v1 px2 py2 v2 ...
                    nkpt, ndim = self.kpt_shape
                    expected = 5 + nkpt * ndim
                    if len(parts) < expected:
                        continue
                    x_center, y_center, width, height = (float(v) for v in parts[1:5])
                    x = x_center - width / 2
                    y = y_center - height / 2
                    kpt_vals = np.array(
                        [float(v) for v in parts[5 : 5 + nkpt * ndim]], dtype=np.float32
                    ).reshape(nkpt, ndim)
                    if ndim == 2:
                        # No visibility channel in labels — treat all as visible.
                        kpts = np.concatenate(
                            [kpt_vals, np.ones((nkpt, 1), dtype=np.float32)], axis=1
                        )
                    else:
                        kpts = kpt_vals
                    annotations.append(
                        {
                            "bbox": [x, y, width, height],
                            "bbox_normalized": True,
                            "yolo_format": True,
                            "category_id": class_id,
                            "area": width * height,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "keypoints_raw": kpts,
                        }
                    )
                    continue

                if len(parts) == 5:
                    # Detection format: cls cx cy w h
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    x = x_center - width / 2
                    y = y_center - height / 2
                    w = width
                    h = height

                    annotations.append(
                        {
                            "bbox": [x, y, w, h],
                            "bbox_normalized": True,
                            "yolo_format": True,
                            "category_id": class_id,
                            "area": w * h,
                            "iscrowd": 0,
                            "image_id": img_id,
                        }
                    )
                else:
                    # Segmentation format: cls x1 y1 x2 y2 x3 y3 ...
                    coords = [float(v) for v in parts[1:]]
                    if len(coords) < 6:
                        continue
                    segments = np.array(coords, dtype=np.float32).reshape(-1, 2)

                    x_min, y_min = segments.min(axis=0)
                    x_max, y_max = segments.max(axis=0)
                    w = x_max - x_min
                    h = y_max - y_min

                    annotations.append(
                        {
                            "bbox": [x_min, y_min, w, h],
                            "bbox_normalized": True,
                            "yolo_format": True,
                            "category_id": class_id,
                            "area": w * h,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "segments": coords,
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

        # For segmentation, sort annotations by area (descending) so that
        # overlap map instance IDs match annotation indices after re-sorting.
        # Matches Ultralytics: instances = instances[sorted_idx]
        if self.task == "segment" and len(anns) > 1:
            anns = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)

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

        # Collect segments (polygon vertices) for segmentation task
        seg_list = []
        for ann in anns:
            seg_coords = self._extract_segments(ann, orig_w, orig_h, ratio, pad)
            seg_list.append(seg_coords)

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

        if self.task == "segment":
            annotation["segments"] = seg_list

        if self.task == "pose":
            nkpt = self.kpt_shape[0]
            kpts_list = [self._extract_keypoints(ann, orig_w, orig_h, ratio, pad) for ann in anns]
            annotation["keypoints"] = (
                np.stack(kpts_list).astype(np.float32)
                if kpts_list
                else np.zeros((0, nkpt, 3), dtype=np.float32)
            )

        # Apply augmentations (training only)
        # Ported from ultralytics RandomPerspective + RandomHSV + RandomFlip
        if self.augment:
            img_np, annotation = self._random_affine(img_np, annotation, scale=0.5, translate=0.1)
            img_np = self._augment_hsv(img_np, hgain=0.015, sgain=0.7, vgain=0.4)
            img_np, annotation = self._random_fliplr(img_np, annotation, p=0.5)

        # Rasterize masks + sem_masks and re-sort annotations (for segmentation task)
        if self.task == "segment":
            self._rasterize_masks(annotation)

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

            # Transform polygon vertices with same affine matrix
            if "segments" in annotation:
                new_segments = []
                for idx, seg in enumerate(annotation["segments"]):
                    if not keep[idx]:
                        continue
                    if seg is None:
                        new_segments.append(seg)
                        continue
                    new_segments.append(self._transform_seg_affine(seg, w, h, M))
                annotation["segments"] = new_segments

            # Transform keypoints with the same affine matrix; keypoints pushed
            # out of frame have their visibility zeroed (matches Ultralytics
            # RandomPerspective.apply_keypoints).
            if "keypoints" in annotation and len(annotation["keypoints"]) > 0:
                kpts = annotation["keypoints"]  # (n, K, 3) normalized letterboxed
                nk = kpts.shape[1]
                xy = kpts[..., :2].reshape(-1, 2).astype(np.float32).copy()
                vis = kpts[..., 2].reshape(-1).astype(np.float32).copy()
                xy[:, 0] *= w
                xy[:, 1] *= h
                pts_h = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)
                xy_t = (pts_h @ M.T)[:, :2]
                inside = (xy_t[:, 0] >= 0) & (xy_t[:, 0] < w) & (xy_t[:, 1] >= 0) & (xy_t[:, 1] < h)
                vis_new = np.where(inside & (vis != 0), vis, 0.0)
                xy_t[:, 0] /= w
                xy_t[:, 1] /= h
                kpts_new = np.concatenate([xy_t, vis_new[:, None]], axis=1).reshape(-1, nk, 3)
                annotation["keypoints"] = kpts_new[keep].astype(np.float32)

        return img, annotation

    @staticmethod
    def _transform_seg_affine(
        seg: list[np.ndarray] | np.ndarray,
        w: int,
        h: int,
        M: np.ndarray,
    ) -> list[np.ndarray] | np.ndarray:
        """Apply affine transform to segment polygon(s)."""
        if isinstance(seg, list):
            return [COCODataset._transform_seg_affine(part, w, h, M) for part in seg]
        if len(seg) < 3:
            return seg
        pts = seg.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])
        transformed = (pts_h @ M.T)[:, :2]
        transformed[:, 0] = np.clip(transformed[:, 0], 0, w) / w
        transformed[:, 1] = np.clip(transformed[:, 1], 0, h) / h
        return transformed

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
        # Ultralytics disables horizontal flipping for pose data without a
        # left/right swap map (fliplr=0 when flip_idx is absent).
        if self.task == "pose" and self.flip_idx is None:
            return img, annotation

        if random.random() < p:
            img = np.ascontiguousarray(np.fliplr(img))
            boxes = annotation["boxes"]
            if len(boxes) > 0:
                boxes_flipped = boxes.copy()
                boxes_flipped[:, 0] = 1.0 - boxes[:, 2]
                boxes_flipped[:, 2] = 1.0 - boxes[:, 0]
                annotation["boxes"] = boxes_flipped

            # Flip polygon vertices horizontally
            if "segments" in annotation:
                annotation["segments"] = [
                    self._flip_segments_lr(seg) for seg in annotation["segments"]
                ]

            # Flip keypoints horizontally and swap left/right joints via flip_idx.
            if "keypoints" in annotation and len(annotation["keypoints"]) > 0:
                kpts = annotation["keypoints"].copy()
                kpts[..., 0] = 1.0 - kpts[..., 0]
                if self.flip_idx is not None and len(self.flip_idx) == kpts.shape[1]:
                    kpts = kpts[:, self.flip_idx, :]
                annotation["keypoints"] = kpts
        return img, annotation

    @staticmethod
    def _flip_segments_lr(
        seg: list[np.ndarray] | np.ndarray | None,
    ) -> list[np.ndarray] | np.ndarray | None:
        """Flip polygon vertices horizontally (x = 1 - x) for normalized coords."""
        if seg is None:
            return seg
        if isinstance(seg, list):
            return [COCODataset._flip_segments_lr(part) for part in seg]
        if len(seg) < 3:
            return seg
        flipped = seg.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        return flipped

    def _extract_keypoints(
        self, ann: dict, orig_w: int, orig_h: int, ratio: float, pad: tuple[float, float]
    ) -> np.ndarray:
        """Extract per-object keypoints transformed to letterboxed normalized space.

        Handles YOLO-pose format (normalized, stored under "keypoints_raw")
        and COCO ``person_keypoints`` JSON (pixel coords, flat list under
        "keypoints"). Returns an array of shape (K, 3) with (x, y, v) where
        x, y are normalized to [0, 1] in the letterboxed image.

        Args:
            ann: Single annotation dict.
            orig_w: Original image width in pixels.
            orig_h: Original image height in pixels.
            ratio: Letterbox scale ratio.
            pad: Letterbox padding (pad_x, pad_y).

        Returns:
            Keypoints array (K, 3) in letterboxed normalized coords.
        """
        nkpt = self.kpt_shape[0]
        out = np.zeros((nkpt, 3), dtype=np.float32)

        if "keypoints_raw" in ann:
            kp = np.asarray(ann["keypoints_raw"], dtype=np.float32)
            x_px = kp[:, 0] * orig_w
            y_px = kp[:, 1] * orig_h
            vis = kp[:, 2]
        elif "keypoints" in ann:
            flat = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            x_px = flat[:, 0]
            y_px = flat[:, 1]
            vis = flat[:, 2]
        else:
            return out

        out[:, 0] = (x_px * ratio + pad[0]) / self.img_size
        out[:, 1] = (y_px * ratio + pad[1]) / self.img_size
        out[:, 2] = vis
        # Keypoints marked not-labeled (v == 0) carry no coordinate.
        invisible = vis == 0
        out[invisible, 0] = 0.0
        out[invisible, 1] = 0.0
        return out

    def _extract_segments(
        self, ann: dict, orig_w: int, orig_h: int, ratio: float, pad: tuple[float, float]
    ) -> list[np.ndarray] | None:
        """Extract polygon vertices from an annotation and transform to letterboxed space.

        Handles YOLO-seg format (normalized, stored in "segments" key),
        COCO JSON polygon format (pixel coords, stored in "segmentation" key),
        and COCO RLE format (decoded via pycocotools if available).

        Returns a list of polygon arrays so that multi-part annotations
        (e.g. occluded objects with separate polygon regions) are fully
        represented.

        Args:
            ann: Single annotation dict.
            orig_w: Original image width.
            orig_h: Original image height.
            ratio: Letterbox scale ratio.
            pad: Letterbox padding (pad_x, pad_y).

        Returns:
            List of (K_i, 2) polygon arrays in letterboxed normalized [0, 1]
            coords, or None if no segmentation is available.
        """
        if "segments" in ann:
            coords = ann["segments"]
            if isinstance(coords, list) and len(coords) >= 6:
                segments = np.array(coords, dtype=np.float32).reshape(-1, 2)
            else:
                return None
            px = segments[:, 0] * orig_w
            py = segments[:, 1] * orig_h
            segments[:, 0] = (px * ratio + pad[0]) / self.img_size
            segments[:, 1] = (py * ratio + pad[1]) / self.img_size
            return [np.clip(segments, 0, 1)]

        if "segmentation" in ann:
            seg = ann["segmentation"]

            if isinstance(seg, dict):
                # RLE format — decode to binary mask, then extract contours
                try:
                    from pycocotools import mask as mask_util

                    rle = seg
                    if isinstance(rle.get("counts"), list):
                        # Uncompressed RLE → compress first
                        rle = mask_util.frPyObjects(rle, rle["size"][0], rle["size"][1])
                    rle_mask = mask_util.decode(rle)  # (H, W) uint8
                    if _HAS_CV2 and rle_mask.any():
                        contours, _ = cv2.findContours(
                            rle_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        parts = []
                        for c in contours:
                            pts = c.squeeze().astype(np.float32)
                            if pts.ndim < 2 or len(pts) < 3:
                                continue
                            pts[:, 0] = (pts[:, 0] * ratio + pad[0]) / self.img_size
                            pts[:, 1] = (pts[:, 1] * ratio + pad[1]) / self.img_size
                            parts.append(np.clip(pts, 0, 1))
                        return parts if parts else None
                except ImportError:
                    pass
                return None

            if isinstance(seg, list) and len(seg) > 0:
                # COCO polygon format: list of flat coord lists (pixel coords).
                # Each element is one polygon part — include ALL of them.
                parts = []
                for poly_flat in seg:
                    if not isinstance(poly_flat, list) or len(poly_flat) < 6:
                        continue
                    poly = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
                    poly[:, 0] = (poly[:, 0] * ratio + pad[0]) / self.img_size
                    poly[:, 1] = (poly[:, 1] * ratio + pad[1]) / self.img_size
                    parts.append(np.clip(poly, 0, 1))
                return parts if parts else None

        return None

    def _rasterize_masks(self, annotation: dict) -> None:
        """Rasterize polygon segments into an overlap mask and semantic mask.

        Sets annotation["masks"] (overlap map) and annotation["sem_masks"]
        (per-pixel class index) at reduced resolution.

        After rasterization, annotations (boxes, labels, segments, etc.)
        are re-sorted to match the overlap map's instance ID order, following
        the Ultralytics convention: instances = instances[sorted_idx].

        Args:
            annotation: Annotation dict with "segments" and "labels".
                        Modified in place.
        """
        seg_list = annotation.get("segments", [])
        labels = annotation.get("labels", np.zeros(0, dtype=np.int64))
        mask_h = self.img_size // self.mask_ratio
        mask_w = self.img_size // self.mask_ratio

        # Build pixel-coord polygons, tracking which annotations have valid segments.
        # Each entry in valid_segs is a list of (K_i, 2) arrays (one or more
        # polygon parts per annotation, supporting multi-part COCO objects).
        valid_indices = []
        valid_segs = []
        for i, seg in enumerate(seg_list):
            if seg is None:
                continue
            if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], np.ndarray):
                parts = []
                for part in seg:
                    if part is not None and len(part) >= 3:
                        pp = part.copy()
                        pp[:, 0] *= mask_w
                        pp[:, 1] *= mask_h
                        parts.append(pp)
                if parts:
                    valid_segs.append(parts)
                    valid_indices.append(i)
            elif isinstance(seg, np.ndarray) and len(seg) >= 3:
                poly_px = seg.copy()
                poly_px[:, 0] *= mask_w
                poly_px[:, 1] *= mask_h
                valid_segs.append([poly_px])
                valid_indices.append(i)

        if not valid_segs:
            annotation["masks"] = np.zeros((mask_h, mask_w), dtype=np.int32)
            annotation["sem_masks"] = np.zeros((mask_h, mask_w), dtype=np.int64)
            return

        masks, sort_idx = polygons2masks_overlap((mask_h, mask_w), valid_segs, downsample_ratio=1)

        # Re-sort annotations to match overlap map instance ID order.
        # sort_idx maps from sorted position → original valid_segs index.
        # full_order maps from sorted position → original annotation index.
        full_order = np.array([valid_indices[j] for j in sort_idx])

        # Rebuild annotation arrays in sorted order, appending any
        # annotations without segments at the end.
        n_total = len(seg_list) if len(seg_list) == len(labels) else len(labels)
        has_seg = set(full_order.tolist())
        no_seg = [i for i in range(n_total) if i not in has_seg]
        reorder = np.concatenate([full_order, np.array(no_seg, dtype=np.int64)])

        if len(reorder) == len(labels):
            annotation["labels"] = labels[reorder]
            annotation["boxes"] = annotation["boxes"][reorder]
            if "areas" in annotation:
                annotation["areas"] = annotation["areas"][reorder]
            if "iscrowd" in annotation:
                annotation["iscrowd"] = annotation["iscrowd"][reorder]
            new_segs = [seg_list[j] if j < len(seg_list) else None for j in reorder]
            annotation["segments"] = new_segs

        # Build semantic mask: per-pixel class index
        sem_mask = np.zeros((mask_h, mask_w), dtype=np.int64)
        sorted_labels = annotation["labels"]
        for inst_id in range(1, int(masks.max()) + 1):
            if inst_id - 1 < len(sorted_labels):
                sem_mask[masks == inst_id] = int(sorted_labels[inst_id - 1])

        annotation["masks"] = masks
        annotation["sem_masks"] = sem_mask

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
        img = np.asarray(image)
        h, w = img.shape[:2]

        # Compute scale factor (matches Ultralytics LetterBox: round, not floor).
        ratio = min(target_size / h, target_size / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

        # Resize with cv2 INTER_LINEAR to match the Ultralytics reference pipeline
        # bit-for-bit (PIL BILINEAR uses a different kernel and shifts the input).
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Total padding needed on each axis. Square: pad to a target_size canvas.
        # Rectangular (Ultralytics rect=True val): pad only to the next stride
        # multiple of the resized side, so the longest side equals target_size
        # and the other side carries minimal padding.
        if self.rect:
            pad_w = (self.stride - new_w % self.stride) % self.stride
            pad_h = (self.stride - new_h % self.stride) % self.stride
        else:
            pad_w = target_size - new_w
            pad_h = target_size - new_h

        # Symmetric padding with the same round(dh +/- 0.1) split Ultralytics uses,
        # so top + bottom (and left + right) always sum to the required padding.
        dw = pad_w / 2
        dh = pad_h / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        # Return the actual integer pad offsets so forward (GT) and inverse
        # (prediction) coordinate mapping stay mutually consistent.
        return Image.fromarray(padded), ratio, (float(left), float(top))

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

        # Rectangular letterbox produces a per-image canvas shape, so images
        # cannot be stacked into a uniform batch; emit one image per batch.
        if self.rect:
            batch_size = 1

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
