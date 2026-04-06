# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Global Motion Compensation (GMC) using OpenCV feature matching.

Estimates camera motion between consecutive frames via optical flow or
feature matching and returns an affine/homography transform that can be
used to warp track predictions, compensating for camera ego-motion.

All computation uses OpenCV (CPU) — no MLX ops in this module.
"""

from __future__ import annotations

import cv2
import numpy as np


class GMC:
    """Estimate inter-frame camera motion.

    Supported methods:
        - ``sparseOptFlow`` — Lucas-Kanade sparse optical flow (default, fast)
        - ``orb`` — ORB feature matching
        - ``sift`` — SIFT feature matching (requires opencv-contrib)
        - ``ecc`` — Enhanced Correlation Coefficient (slow, precise)
        - ``none`` — no compensation (identity transform)

    Usage::

        gmc = GMC(method="sparseOptFlow")
        for frame in video:
            H = gmc.apply(frame)  # (2, 3) affine matrix
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2):
        self.method = method.lower()
        self.downscale = max(1, downscale)
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == "sift":
            self.detector = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20
            )
            self.extractor = self.detector
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.detector = None
            self.extractor = None
            self.matcher = None

    def apply(self, raw_frame: np.ndarray) -> np.ndarray:
        """Compute affine transform from previous frame to current frame.

        Args:
            raw_frame: BGR or RGB uint8 image.

        Returns:
            (2, 3) affine matrix. Identity if first frame or method is 'none'.
        """
        if self.method == "none":
            return np.eye(2, 3, dtype=np.float64)

        # Convert to grayscale and downscale
        if len(raw_frame.shape) == 3:
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = raw_frame

        if self.downscale > 1:
            frame = cv2.resize(
                frame, (frame.shape[1] // self.downscale, frame.shape[0] // self.downscale)
            )

        H = np.eye(2, 3, dtype=np.float64)

        if self.prev_frame is None:
            self.prev_frame = frame
            self._detect_and_store(frame)
            return H

        if self.method == "sparseoptflow":
            H = self._sparse_optflow(frame)
        elif self.method == "ecc":
            H = self._ecc(frame)
        elif self.method in ("orb", "sift"):
            H = self._feature_match(frame)

        # Adjust for downscale
        if self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale

        self.prev_frame = frame
        self._detect_and_store(frame)
        return H

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _sparse_optflow(self, frame: np.ndarray) -> np.ndarray:
        """Lucas-Kanade sparse optical flow."""
        keypoints = cv2.goodFeaturesToTrack(
            self.prev_frame,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        H = np.eye(2, 3, dtype=np.float64)
        if keypoints is None or len(keypoints) == 0:
            return H

        matched, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame,
            frame,
            keypoints,
            None,
        )
        if matched is None:
            return H

        status = status.flatten()
        prev_pts = keypoints[status == 1]
        curr_pts = matched[status == 1]

        if len(prev_pts) < 4:
            return H

        transform, inliers = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if transform is not None:
            H = transform
        return H

    def _ecc(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced Correlation Coefficient alignment."""
        H = np.eye(2, 3, dtype=np.float64)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
            _, warp = cv2.findTransformECC(
                self.prev_frame,
                frame,
                H.astype(np.float32),
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                1,
            )
            H = warp.astype(np.float64)
        except cv2.error:
            pass
        return H

    def _feature_match(self, frame: np.ndarray) -> np.ndarray:
        """ORB or SIFT feature matching."""
        H = np.eye(2, 3, dtype=np.float64)
        keypoints = self.detector.detect(frame)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        if (
            descriptors is None
            or self.prev_descriptors is None
            or len(descriptors) < 2
            or len(self.prev_descriptors) < 2
        ):
            return H

        knn_matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Ratio test
        good = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.9 * n.distance:
                good.append(m)

        if len(good) < 4:
            return H

        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        transform, _ = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if transform is not None:
            H = transform
        return H

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_and_store(self, frame: np.ndarray) -> None:
        """Detect and store keypoints/descriptors for feature-based methods."""
        if self.detector is not None:
            kps = self.detector.detect(frame)
            kps, desc = self.extractor.compute(frame, kps)
            self.prev_keypoints = kps
            self.prev_descriptors = desc

    def reset(self) -> None:
        """Reset internal state (new video/sequence)."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
