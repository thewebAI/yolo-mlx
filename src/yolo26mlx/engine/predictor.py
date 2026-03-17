# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Predictor - Pure MLX Implementation

Inference pipeline for YOLO26 models with mx.compile optimization.
"""

from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
import numpy as np
from PIL import Image

from yolo26mlx.engine.results import OBB, Boxes, Keypoints, Masks, Results


class Predictor:
    """YOLO26 Prediction class - Pure MLX.

    Optimized inference with mx.compile for maximum performance on Apple Silicon.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "detect",
        names: dict[int, str] | None = None,
        stride: mx.array | None = None,
    ):
        """Initialize predictor.

        Args:
            model: YOLO26 model for inference
            task: Task type - 'detect', 'segment', 'pose', or 'obb'
            names: Class names dictionary
            stride: Model stride tensor
        """
        self.model = model
        self.task = task
        self.names = names or {}
        self.stride = stride if stride is not None else mx.array([8.0, 16.0, 32.0])

        # Compiled inference function
        self._compiled_predict = None

        # Set model to eval mode
        if self.model is not None:
            self.model.eval()

    def __call__(
        self,
        source: str | Path | list | np.ndarray | Image.Image,
        conf: float = 0.25,
        imgsz: int = 640,
        save: bool = False,
        stream: bool = False,
        rect: bool = True,
    ) -> list[Results] | Generator[Results, None, None]:
        """Run inference.

        Args:
            source: Image path, directory, list, numpy array, or PIL Image
            conf: Confidence threshold
            imgsz: Input image size
            save: Save results to disk
            stream: Return generator for memory efficiency
            rect: Rectangular inference (pad to stride-aligned dims, not square).
                  Matches ultralytics predict default. Faster for non-square images.

        Returns:
            List of Results objects (or generator if stream=True)
        """
        # Process source
        images, paths, orig_imgs, letterbox_infos = self._load_source(source, imgsz, rect=rect)

        if stream:
            return self._stream_predict(images, paths, orig_imgs, letterbox_infos, conf, save)
        else:
            return self._batch_predict(images, paths, orig_imgs, letterbox_infos, conf, save)

    def _load_source(
        self, source: str | Path | list | np.ndarray | Image.Image, imgsz: int, rect: bool = True
    ) -> tuple:
        """Load and preprocess source images.

        Args:
            source: Image source(s)
            imgsz: Target image size
            rect: Rectangular letterbox (stride-aligned, non-square)

        Returns:
            Tuple of (preprocessed image batch, paths, original images, letterbox info dicts)
        """
        if isinstance(source, str | Path):
            source_path = Path(source)
            if source_path.is_dir():
                # Directory of images
                sources = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            else:
                sources = [source_path]
        elif isinstance(source, np.ndarray):
            # Single numpy array
            sources = [source]
        elif isinstance(source, Image.Image):
            # Single PIL Image
            sources = [source]
        elif isinstance(source, list):
            sources = source
        else:
            sources = [source]

        images = []
        paths = []
        orig_imgs = []
        letterbox_infos = []

        for src in sources:
            img, path, orig, lb_info = self._preprocess_single(src, imgsz, rect=rect)
            images.append(img)
            paths.append(path)
            orig_imgs.append(orig)
            letterbox_infos.append(lb_info)

        # Stack into batch
        batch = mx.stack(images, axis=0) if len(images) > 1 else mx.expand_dims(images[0], axis=0)

        return batch, paths, orig_imgs, letterbox_infos

    def _preprocess_single(
        self, source: str | Path | np.ndarray | Image.Image, imgsz: int, rect: bool = True
    ) -> tuple:
        """Preprocess a single image.

        Uses OpenCV (C-backed) for fast image loading and resizing when available,
        falling back to PIL for compatibility.

        Args:
            source: Image source
            imgsz: Target size
            rect: Rectangular letterbox (stride-aligned, non-square)

        Returns:
            Tuple of (preprocessed tensor, path, original image, letterbox info dict)
        """
        if _HAS_CV2:
            return self._preprocess_cv2(source, imgsz, rect=rect)
        return self._preprocess_pil(source, imgsz, rect=rect)

    def _preprocess_cv2(
        self, source: str | Path | np.ndarray | Image.Image, imgsz: int, rect: bool = True
    ) -> tuple:
        """Fast preprocessing using OpenCV (C-backed, ~5x faster than PIL).

        Args:
            source: Image path, PIL Image, or numpy array to preprocess.
            imgsz: Target image size for letterbox resize.
            rect: If True, pad to stride-aligned dims instead of square.

        Returns:
            Tuple of (MLX tensor, path string, original numpy image, letterbox info dict).
        """
        if isinstance(source, str | Path):
            path = str(source)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(source, Image.Image):
            path = ""
            img = np.array(source.convert("RGB"))
        elif isinstance(source, np.ndarray):
            path = ""
            img = source if source.ndim == 3 else source
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Store original (already numpy RGB)
        orig_img = img.copy()

        # Letterbox resize using OpenCV
        img_resized, ratio, (dw, dh) = self._letterbox_cv2(img, imgsz, auto=rect)

        # Convert to float32 and normalize (in-place for speed)
        img_array = img_resized.astype(np.float32, copy=False)
        img_array *= 1.0 / 255.0

        # Store letterbox params for postprocess coordinate rescaling
        letterbox_info = {"ratio": ratio, "dw": dw, "dh": dh}

        return mx.array(img_array), path, orig_img, letterbox_info

    def _preprocess_pil(
        self, source: str | Path | np.ndarray | Image.Image, imgsz: int, rect: bool = True
    ) -> tuple:
        """Fallback preprocessing using PIL when OpenCV is unavailable.

        Args:
            source: Image path, PIL Image, or numpy array to preprocess.
            imgsz: Target image size for letterbox resize.
            rect: If True, pad to stride-aligned dims instead of square.

        Returns:
            Tuple of (MLX tensor, path string, original numpy image, letterbox info dict).
        """
        if isinstance(source, str | Path):
            path = str(source)
            img = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            path = ""
            img = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            path = ""
            img = Image.fromarray(source.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        orig_img = np.array(img)
        img_resized, ratio, (dw, dh) = self._letterbox_pil(img, imgsz, auto=rect)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Store letterbox params for postprocess coordinate rescaling
        letterbox_info = {"ratio": ratio, "dw": dw, "dh": dh}

        return mx.array(img_array), path, orig_img, letterbox_info

    def _letterbox_cv2(
        self,
        img: np.ndarray,
        new_size: int,
        color: tuple = (114, 114, 114),
        auto: bool = False,
        stride: int = 32,
    ) -> tuple:
        """Fast letterbox resize using OpenCV.

        Args:
            img: numpy array (H, W, C) in RGB
            new_size: Target size
            color: Padding color
            auto: If True, pad to stride-aligned dims (non-square, faster).
                  If False, pad to square new_size x new_size.
            stride: Stride for auto-alignment (default 32)

        Returns:
            Tuple of (resized image, scale ratio, padding)
        """
        h, w = img.shape[:2]

        # Compute scale
        ratio = min(new_size / h, new_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # Resize with OpenCV (much faster than PIL)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        if auto:
            # Pad to nearest stride multiple (non-square, fewer pixels)
            dw = (stride - new_w % stride) % stride
            dh = (stride - new_h % stride) % stride
        else:
            # Pad to square
            dw = new_size - new_w
            dh = new_size - new_h

        dw /= 2
        dh /= 2

        # Pad with cv2.copyMakeBorder (faster than creating new image + paste)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return padded, ratio, (dw, dh)

    def _letterbox_pil(
        self,
        img: Image.Image,
        new_size: int,
        color: tuple = (114, 114, 114),
        auto: bool = False,
        stride: int = 32,
    ) -> tuple:
        """Fallback letterbox resize using PIL.

        Args:
            img: PIL Image in RGB mode.
            new_size: Target size (longest edge).
            color: Padding fill color (R, G, B).
            auto: If True, pad to stride-aligned dims (non-square, faster).
                  If False, pad to square new_size x new_size.
            stride: Stride for auto-alignment (default 32).

        Returns:
            Tuple of (padded PIL Image, scale ratio, (dw, dh) padding).
        """
        w, h = img.size
        ratio = min(new_size / h, new_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        if auto:
            pad_w = (stride - new_w % stride) % stride
            pad_h = (stride - new_h % stride) % stride
        else:
            pad_w = new_size - new_w
            pad_h = new_size - new_h

        dw = pad_w / 2
        dh = pad_h / 2

        canvas_w = new_w + int(round(dw - 0.1)) + int(round(dw + 0.1))
        canvas_h = new_h + int(round(dh - 0.1)) + int(round(dh + 0.1))
        padded = Image.new("RGB", (canvas_w, canvas_h), color)
        padded.paste(img_resized, (int(round(dw - 0.1)), int(round(dh - 0.1))))
        return padded, ratio, (dw, dh)

    def _batch_predict(
        self,
        images: mx.array,
        paths: list[str],
        orig_imgs: list[np.ndarray],
        letterbox_infos: list[dict],
        conf: float,
        save: bool,
    ) -> list[Results]:
        """Run batch prediction.

        Args:
            images: Preprocessed image batch (B, H, W, C)
            paths: Image paths
            orig_imgs: Original images
            letterbox_infos: Letterbox ratio/padding per image
            conf: Confidence threshold
            save: Save results

        Returns:
            List of Results objects
        """
        # Run inference with compiled model
        preds = self._predict(images)

        # Force evaluation
        mx.eval(preds)

        # Post-process predictions
        results = []
        for i in range(len(paths)):
            result = self._postprocess(
                preds=preds,
                batch_idx=i,
                orig_img=orig_imgs[i],
                path=paths[i],
                letterbox_info=letterbox_infos[i],
                conf=conf,
            )
            results.append(result)

            if save:
                result.save()

        return results

    def _stream_predict(
        self,
        images: mx.array,
        paths: list[str],
        orig_imgs: list[np.ndarray],
        letterbox_infos: list[dict],
        conf: float,
        save: bool,
    ) -> Generator[Results, None, None]:
        """Yield Results one image at a time for memory-efficient streaming.

        Args:
            images: Preprocessed image batch (B, H, W, C).
            paths: Image file paths corresponding to each image.
            orig_imgs: Original unprocessed images as numpy arrays.
            letterbox_infos: Letterbox ratio/padding dicts for coordinate rescaling.
            conf: Confidence threshold for filtering detections.
            save: If True, save each annotated result to disk.

        Yields:
            Results object for each image in the batch.
        """
        for i in range(images.shape[0]):
            # Process single image
            img = mx.expand_dims(images[i], axis=0)
            preds = self._predict(img)
            mx.eval(preds)

            result = self._postprocess(
                preds=preds,
                batch_idx=0,
                orig_img=orig_imgs[i],
                path=paths[i],
                letterbox_info=letterbox_infos[i],
                conf=conf,
            )

            if save:
                result.save()

            yield result

    def _predict(self, images: mx.array) -> mx.array:
        """Run model inference.

        Uses mx.compile for optimized inference on Apple Silicon.
        Automatically enables compile_for_inference() on the model for
        27-50% faster inference via JIT compilation.

        Args:
            images: Preprocessed image tensor (B, H, W, C)

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Enable compiled inference on first call (27-50% faster)
        if self._compiled_predict is None:
            if hasattr(self.model, "compile_for_inference"):
                self.model.compile_for_inference()
            self._compiled_predict = True  # Mark as initialized

        # Run inference (uses compiled path transparently via model.__call__)
        return self.model(images)

    def _postprocess(
        self,
        preds: mx.array,
        batch_idx: int,
        orig_img: np.ndarray,
        path: str,
        letterbox_info: dict,
        conf: float,
    ) -> Results:
        """Post-process model predictions.

        Args:
            preds: Raw model outputs
            batch_idx: Batch index
            orig_img: Original image
            path: Image path
            letterbox_info: Letterbox params {ratio, dw, dh} for coordinate rescaling
            conf: Confidence threshold

        Returns:
            Results object
        """
        # Get predictions for this batch item
        if isinstance(preds, list | tuple):
            pred = preds[batch_idx] if len(preds) > 1 else preds[0]
        elif isinstance(preds, mx.array):
            pred = preds[batch_idx] if preds.shape[0] > 1 else preds[0]
        else:
            pred = preds

        # Convert to numpy for processing
        if isinstance(pred, mx.array):
            pred = np.array(pred)

        # Apply task-specific post-processing
        if self.task == "detect":
            boxes = self._postprocess_detect(pred, orig_img.shape[:2], letterbox_info, conf)
            return Results(orig_img=orig_img, path=path, names=self.names, boxes=boxes)
        elif self.task == "segment":
            boxes, masks = self._postprocess_segment(pred, orig_img.shape[:2], letterbox_info, conf)
            return Results(orig_img=orig_img, path=path, names=self.names, boxes=boxes, masks=masks)
        elif self.task == "pose":
            boxes, keypoints = self._postprocess_pose(
                pred, orig_img.shape[:2], letterbox_info, conf
            )
            return Results(
                orig_img=orig_img, path=path, names=self.names, boxes=boxes, keypoints=keypoints
            )
        elif self.task == "obb":
            obb_results = self._postprocess_obb(orig_img.shape[:2])
            return Results(orig_img=orig_img, path=path, names=self.names, obb=obb_results)
        else:
            return Results(orig_img=orig_img, path=path, names=self.names)

    def _postprocess_detect(
        self, pred: np.ndarray, orig_shape: tuple, letterbox_info: dict, conf: float
    ) -> Boxes:
        """Post-process detection predictions.

        YOLO26 with end2end=True outputs [cx, cy, w, h, conf, class_idx] per detection.
        Non-end2end outputs [cx, cy, w, h, class_scores...] per anchor.

        Steps:
        1. Parse end2end vs non-end2end output format
        2. Filter by confidence
        3. Convert xywh → xyxy
        4. Rescale from letterboxed coordinates → original image coordinates
        5. Clip to image bounds

        Args:
            pred: Raw predictions (N, 6) for end2end or (N, 4+nc) for non-end2end
            orig_shape: Original image shape (H, W)
            letterbox_info: {'ratio': float, 'dw': float, 'dh': float}
            conf: Confidence threshold

        Returns:
            Boxes object with [x1, y1, x2, y2, conf, cls] in original image coords
        """
        if pred is None or len(pred) == 0:
            return Boxes(np.empty((0, 6)), orig_shape)

        # Ensure 2D
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        n_cols = pred.shape[-1]

        if n_cols == 6:
            # End2end format: [cx, cy, w, h, conf, class_idx]
            boxes_xywh = pred[:, :4]
            scores = pred[:, 4]
            cls = pred[:, 5]
        else:
            # Non-end2end format: [cx, cy, w, h, class_scores...]
            boxes_xywh = pred[:, :4]
            class_scores = pred[:, 4:]
            scores = np.max(class_scores, axis=-1)
            cls = np.argmax(class_scores, axis=-1).astype(np.float32)

        # Filter by confidence
        mask = scores > conf
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        cls = cls[mask]

        if len(boxes_xywh) == 0:
            return Boxes(np.empty((0, 6)), orig_shape)

        # Convert xywh (center) → xyxy (corners)
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Rescale from letterboxed coordinates → original image coordinates
        # Letterbox: original → resize by ratio → pad by (dw, dh)
        # Reverse: subtract padding → divide by ratio
        ratio = letterbox_info["ratio"]
        dw = letterbox_info["dw"]
        dh = letterbox_info["dh"]

        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # Clip to original image bounds
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Combine into output format [x1, y1, x2, y2, conf, cls]
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        output = np.column_stack([boxes_xyxy, scores, cls])

        return Boxes(output, orig_shape)

    def _postprocess_segment(
        self, pred: np.ndarray, orig_shape: tuple, letterbox_info: dict, conf: float
    ) -> tuple:
        """Post-process segmentation predictions into Boxes and Masks.

        Args:
            pred: Raw model output array for one image.
            orig_shape: Original image shape (H, W) before preprocessing.
            letterbox_info: Dict with 'ratio', 'dw', 'dh' for coordinate rescaling.
            conf: Confidence threshold for filtering detections.

        Returns:
            Tuple of (Boxes, Masks) for this image.
        """
        # TODO: Implement segmentation post-processing
        boxes = self._postprocess_detect(pred, orig_shape, letterbox_info, conf)
        masks = Masks(None, orig_shape)
        return boxes, masks

    def _postprocess_pose(
        self, pred: np.ndarray, orig_shape: tuple, letterbox_info: dict, conf: float
    ) -> tuple:
        """Post-process pose predictions into Boxes and Keypoints.

        Args:
            pred: Raw model output array for one image.
            orig_shape: Original image shape (H, W) before preprocessing.
            letterbox_info: Dict with 'ratio', 'dw', 'dh' for coordinate rescaling.
            conf: Confidence threshold for filtering detections.

        Returns:
            Tuple of (Boxes, Keypoints) for this image.
        """
        # TODO: Implement pose post-processing
        boxes = self._postprocess_detect(pred, orig_shape, letterbox_info, conf)
        keypoints = Keypoints(None, orig_shape)
        return boxes, keypoints

    def _postprocess_obb(self, orig_shape: tuple) -> OBB:
        """Post-process oriented bounding box predictions.

        Args:
            orig_shape: Original image shape (H, W) before preprocessing.

        Returns:
            OBB object with oriented bounding boxes (currently empty placeholder).
        """
        # TODO: Implement OBB post-processing
        return OBB(np.empty((0, 7)), orig_shape)
