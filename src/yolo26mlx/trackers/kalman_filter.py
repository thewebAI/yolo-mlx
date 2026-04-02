# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pure MLX Kalman filters for bounding box tracking."""

import mlx.core as mx


class KalmanFilterXYAH:
    """Kalman filter with state space (x, y, a, h, vx, vy, va, vh).

    Tracks bounding boxes using center position (x, y), aspect ratio (a),
    height (h), and their respective velocities. Uses constant-velocity
    motion model with linear observation.
    """

    def __init__(self):
        """Initialize motion model matrices and uncertainty weights."""
        ndim = 4
        dt = 1.0

        # Motion matrix F: [[I, dt*I], [0, I]]
        top = mx.concatenate([mx.eye(ndim), dt * mx.eye(ndim)], axis=1)
        bottom = mx.concatenate([mx.zeros((ndim, ndim)), mx.eye(ndim)], axis=1)
        self._motion_mat = mx.concatenate([top, bottom], axis=0)  # (8, 8)

        # Observation matrix H: [I, 0]
        self._update_mat = mx.concatenate([mx.eye(ndim), mx.zeros((ndim, ndim))], axis=1)  # (4, 8)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # Pre-computed std multiplier/addend vectors for h-scaled noise.
        # std = mul * h + add  (element-wise, h broadcasts from (1,) to (8,))
        wp = self._std_weight_position
        wv = self._std_weight_velocity
        self._init_std_mul = mx.array([2 * wp, 2 * wp, 0, 2 * wp, 10 * wv, 10 * wv, 0, 10 * wv])
        self._init_std_add = mx.array([0, 0, 1e-2, 0, 0, 0, 1e-5, 0])
        self._pred_std_mul = mx.array([wp, wp, 0, wp, wv, wv, 0, wv])
        self._pred_std_add = mx.array([0, 0, 1e-2, 0, 0, 0, 1e-5, 0])
        self._proj_std_mul = mx.array([wp, wp, 0, wp])
        self._proj_std_add = mx.array([0, 0, 1e-1, 0])

    def initiate(self, measurement):
        """Create track from unassociated measurement [x, y, a, h].

        Args:
            measurement: 4-dim measurement vector [x, y, a, h].

        Returns:
            (mean, covariance): 8-dim mean vector and 8x8 covariance matrix.
        """
        mean_pos = measurement
        mean_vel = mx.zeros_like(mean_pos)
        mean = mx.concatenate([mean_pos, mean_vel])

        h = measurement[3:4]  # (1,) — broadcasts to (8,)
        std = self._init_std_mul * h + self._init_std_add
        covariance = mx.diag(mx.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Args:
            mean: 8-dim state mean vector.
            covariance: 8x8 state covariance matrix.

        Returns:
            (mean, covariance): Predicted state distribution.
        """
        h = mean[3:4]  # (1,) — broadcasts to (8,)
        std = self._pred_std_mul * h + self._pred_std_add
        motion_cov = mx.diag(mx.square(std))

        F = self._motion_mat
        mean = mx.matmul(mean, mx.transpose(F))
        covariance = mx.matmul(mx.matmul(F, covariance), mx.transpose(F)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Args:
            mean: 8-dim state mean vector.
            covariance: 8x8 state covariance matrix.

        Returns:
            (mean, covariance): Projected mean (4-dim) and covariance (4x4).
        """
        h = mean[3:4]  # (1,) — broadcasts to (4,)
        std = self._proj_std_mul * h + self._proj_std_add
        innovation_cov = mx.diag(mx.square(std))

        H = self._update_mat
        mean = mx.matmul(H, mean)
        covariance = mx.matmul(mx.matmul(H, covariance), mx.transpose(H))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Batched Kalman filter prediction for N tracks.

        Args:
            mean: (N, 8) state mean matrix.
            covariance: (N, 8, 8) state covariance matrices.

        Returns:
            (mean, covariance): Predicted states, shapes (N, 8) and (N, 8, 8).
        """
        h = mean[:, 3:4]  # (N, 1) — broadcasts with (8,) to (N, 8)
        std = self._pred_std_mul * h + self._pred_std_add

        sqr = mx.square(std)
        motion_cov = mx.expand_dims(sqr, -1) * mx.eye(8)  # (N, 8, 8)

        F = self._motion_mat
        F_T = mx.transpose(F)

        mean = mx.matmul(mean, F_T)  # (N, 8)
        covariance = (
            mx.matmul(mx.matmul(mx.expand_dims(F, 0), covariance), F_T) + motion_cov
        )  # (N, 8, 8)

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Args:
            mean: 8-dim predicted state mean.
            covariance: 8x8 predicted state covariance.
            measurement: 4-dim measurement vector [x, y, a, h].

        Returns:
            (new_mean, new_covariance): Corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain: K = P H^T S^{-1}
        PH_T = mx.matmul(covariance, mx.transpose(self._update_mat))  # (8, 4)
        S_inv = mx.linalg.inv(projected_cov, stream=mx.cpu)  # (4, 4)
        kalman_gain = mx.matmul(PH_T, S_inv)  # (8, 4)

        innovation = measurement - projected_mean
        new_mean = mean + mx.matmul(innovation, mx.transpose(kalman_gain))
        new_covariance = covariance - mx.matmul(
            mx.matmul(kalman_gain, projected_cov), mx.transpose(kalman_gain)
        )
        return new_mean, new_covariance

    def multi_update(self, means, covariances, measurements):
        """Batched Kalman filter update for N tracks.

        Performs all N updates with a single ``mx.linalg.inv`` call,
        reducing N CPU/GPU sync points to one.

        Args:
            means: (N, 8) state mean matrix.
            covariances: (N, 8, 8) state covariance matrices.
            measurements: (N, 4) measurement matrix.

        Returns:
            (new_means, new_covariances): Updated states (N, 8) and (N, 8, 8).
        """
        H = self._update_mat  # (4, 8)
        H_T = mx.transpose(H)  # (8, 4)

        # Batched projection
        h = means[:, 3:4]  # (N, 1)
        std = self._proj_std_mul * h + self._proj_std_add  # (N, 4)
        R = mx.expand_dims(mx.square(std), -1) * mx.eye(4)  # (N, 4, 4)

        proj_means = mx.matmul(means, H_T)  # (N, 4)
        HP = mx.matmul(mx.expand_dims(H, 0), covariances)  # (N, 4, 8)
        proj_covs = mx.matmul(HP, H_T) + R  # (N, 4, 4)

        # Batched Kalman gain — ONE inv call for all N matrices
        PH_T = mx.matmul(covariances, H_T)  # (N, 8, 4)
        S_inv = mx.linalg.inv(proj_covs, stream=mx.cpu)  # (N, 4, 4)
        K = mx.matmul(PH_T, S_inv)  # (N, 8, 4)

        # Batched state correction
        innovations = measurements - proj_means  # (N, 4)
        new_means = means + mx.squeeze(
            mx.matmul(mx.expand_dims(innovations, 1), mx.transpose(K, axes=(0, 2, 1))),
            axis=1,
        )  # (N, 8)
        KS = mx.matmul(K, proj_covs)  # (N, 8, 4)
        new_covs = covariances - mx.matmul(KS, mx.transpose(K, axes=(0, 2, 1)))  # (N, 8, 8)

        return new_means, new_covs

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric="maha"):
        """Compute gating distance between state distribution and measurements.

        Args:
            mean: 8-dim state mean vector.
            covariance: 8x8 state covariance matrix.
            measurements: (N, 4) measurement matrix.
            only_position: If True, use only position (first 2 dims).
            metric: "maha" for Mahalanobis, "gaussian" for squared Euclidean.

        Returns:
            (N,) array of squared distances.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return mx.sum(d * d, axis=1)
        elif metric == "maha":
            L = mx.linalg.cholesky(covariance, stream=mx.cpu)
            z = mx.linalg.solve_triangular(L, mx.transpose(d), upper=False, stream=mx.cpu)
            return mx.sum(z * z, axis=0)
        else:
            raise ValueError(f"Invalid distance metric: {metric}")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """Kalman filter with state space (x, y, w, h, vx, vy, vw, vh).

    Uses width (w) instead of aspect ratio (a). Inherits from KalmanFilterXYAH
    and overrides std computation to use both width and height for scaling.
    """

    def initiate(self, measurement):
        """Create track from unassociated measurement [x, y, w, h].

        Args:
            measurement: 4-dim measurement vector [x, y, w, h].

        Returns:
            (mean, covariance): 8-dim mean vector and 8x8 covariance matrix.
        """
        mean_pos = measurement
        mean_vel = mx.zeros_like(mean_pos)
        mean = mx.concatenate([mean_pos, mean_vel])

        w = measurement[2:3]
        h = measurement[3:4]
        wp = self._std_weight_position
        wv = self._std_weight_velocity

        wh = mx.concatenate([w, h, w, h])  # (4,)
        std = mx.concatenate([2 * wp * wh, 10 * wv * wh])  # (8,)
        covariance = mx.diag(mx.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step (XYWH variant).

        Args:
            mean: 8-dim state mean vector.
            covariance: 8x8 state covariance matrix.

        Returns:
            (mean, covariance): Predicted state distribution.
        """
        w = mean[2:3]
        h = mean[3:4]
        wp = self._std_weight_position
        wv = self._std_weight_velocity

        wh = mx.concatenate([w, h, w, h])  # (4,)
        std = mx.concatenate([wp * wh, wv * wh])  # (8,)
        motion_cov = mx.diag(mx.square(std))

        F = self._motion_mat
        mean = mx.matmul(mean, mx.transpose(F))
        covariance = mx.matmul(mx.matmul(F, covariance), mx.transpose(F)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space (XYWH variant).

        Args:
            mean: 8-dim state mean vector.
            covariance: 8x8 state covariance matrix.

        Returns:
            (mean, covariance): Projected mean (4-dim) and covariance (4x4).
        """
        w = mean[2:3]
        h = mean[3:4]
        wp = self._std_weight_position

        wh = mx.concatenate([w, h, w, h])  # (4,)
        std = wp * wh
        innovation_cov = mx.diag(mx.square(std))

        H = self._update_mat
        mean = mx.matmul(H, mean)
        covariance = mx.matmul(mx.matmul(H, covariance), mx.transpose(H))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Batched Kalman filter prediction for N tracks (XYWH variant).

        Args:
            mean: (N, 8) state mean matrix.
            covariance: (N, 8, 8) state covariance matrices.

        Returns:
            (mean, covariance): Predicted states, shapes (N, 8) and (N, 8, 8).
        """
        w = mean[:, 2:3]  # (N, 1)
        h = mean[:, 3:4]  # (N, 1)
        wp = self._std_weight_position
        wv = self._std_weight_velocity

        wh = mx.concatenate([w, h, w, h], axis=1)  # (N, 4)
        std = mx.concatenate([wp * wh, wv * wh], axis=1)  # (N, 8)

        sqr = mx.square(std)
        motion_cov = mx.expand_dims(sqr, -1) * mx.eye(8)  # (N, 8, 8)

        F = self._motion_mat
        F_T = mx.transpose(F)

        mean = mx.matmul(mean, F_T)
        covariance = mx.matmul(mx.matmul(mx.expand_dims(F, 0), covariance), F_T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step (XYWH variant).

        Args:
            mean: 8-dim predicted state mean.
            covariance: 8x8 predicted state covariance.
            measurement: 4-dim measurement vector [x, y, w, h].

        Returns:
            (new_mean, new_covariance): Corrected state distribution.
        """
        return super().update(mean, covariance, measurement)

    def multi_update(self, means, covariances, measurements):
        """Batched Kalman filter update for N tracks (XYWH variant).

        Uses width- and height-scaled projection noise instead of
        the XYAH height-only scaling.

        Args:
            means: (N, 8) state mean matrix.
            covariances: (N, 8, 8) state covariance matrices.
            measurements: (N, 4) measurement matrix.

        Returns:
            (new_means, new_covariances): Updated states (N, 8) and (N, 8, 8).
        """
        H = self._update_mat  # (4, 8)
        H_T = mx.transpose(H)
        wp = self._std_weight_position

        w = means[:, 2:3]  # (N, 1)
        h = means[:, 3:4]  # (N, 1)
        wh = mx.concatenate([w, h, w, h], axis=1)  # (N, 4)
        std = wp * wh  # (N, 4)
        R = mx.expand_dims(mx.square(std), -1) * mx.eye(4)  # (N, 4, 4)

        proj_means = mx.matmul(means, H_T)
        HP = mx.matmul(mx.expand_dims(H, 0), covariances)
        proj_covs = mx.matmul(HP, H_T) + R

        PH_T = mx.matmul(covariances, H_T)
        S_inv = mx.linalg.inv(proj_covs, stream=mx.cpu)
        K = mx.matmul(PH_T, S_inv)

        innovations = measurements - proj_means
        new_means = means + mx.squeeze(
            mx.matmul(mx.expand_dims(innovations, 1), mx.transpose(K, axes=(0, 2, 1))),
            axis=1,
        )
        KS = mx.matmul(K, proj_covs)
        new_covs = covariances - mx.matmul(KS, mx.transpose(K, axes=(0, 2, 1)))
        return new_means, new_covs
