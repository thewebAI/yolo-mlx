# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MuSGD Optimizer for MLX — Port of PyTorch ultralytics MuSGD.

Implements the hybrid Muon + Nesterov SGD optimizer used by ultralytics
when `optimizer='auto'`. This is critical for matching MPS/CPU training
accuracy because the Newton-Schulz orthogonalization makes each gradient
step dramatically more effective.

Reference: ultralytics/ultralytics/optim/muon.py
"""

import re

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def zeropower_via_newtonschulz5(G, eps=1e-7):
    """Compute the zeroth power / orthogonalization of G via Newton-Schulz.

    Port of PyTorch's `zeropower_via_newtonschulz5` from ultralytics/optim/muon.py.
    Produces approximate UV^T from SVD, making gradient updates more effective
    by removing poor conditioning from the update direction.

    Args:
        G: 2D MLX array to orthogonalize.
        eps: Numerical stability epsilon.

    Returns:
        Orthogonalized matrix with same shape as G.
    """
    assert G.ndim == 2, f"Expected 2D input, got {G.ndim}D"
    # Use bfloat16 matching PyTorch: X = G.bfloat16()
    # The NS coefficients (3.4445, -4.7750, 2.0315) were optimized for bfloat16
    # convergence characteristics. Using float32 produces a tighter but *different*
    # approximation to UV^T, changing the effective update direction.
    #
    # Normalize in float32 first (PyTorch's .norm() accumulates in float32
    # internally even for bf16 inputs), then cast to bf16 for iterations.
    X = G.astype(mx.float32)
    X = X / (mx.linalg.norm(X) + eps)
    X = X.astype(mx.bfloat16)

    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T

    # 5 Newton-Schulz iterations with fixed coefficients
    # (optimized for maximum convergence slope at zero)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def muon_update(grad, momentum_buf, beta=0.9, nesterov=True):
    """Compute Muon optimizer update with momentum and orthogonalization.

    Port of PyTorch's `muon_update` from ultralytics/optim/muon.py.

    Args:
        grad: Gradient tensor (2D or 4D).
        momentum_buf: Momentum buffer (same shape as grad).
        beta: Momentum coefficient.
        nesterov: Whether to use Nesterov acceleration.

    Returns:
        (update, new_momentum_buf): Orthogonalized update and updated momentum.
    """
    # Update momentum: buf = beta * buf + (1 - beta) * grad
    new_buf = beta * momentum_buf + (1.0 - beta) * grad

    # Nesterov lookahead
    if nesterov:
        update = (1.0 - beta) * grad + beta * new_buf
    else:
        update = new_buf

    # Reshape 4D conv filters to 2D for orthogonalization
    orig_shape = update.shape
    if update.ndim == 4:
        update = update.reshape(update.shape[0], -1)

    # Orthogonalize via Newton-Schulz
    update = zeropower_via_newtonschulz5(update)

    # Scale by sqrt(max(1, rows/cols)) using ORIGINAL grad dimensions.
    # PyTorch: update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    # Must use grad (not reshaped update) to get correct scale for 4D convs.
    # E.g. 1x1 conv [256,128,1,1]: grad dims → scale=1.0, reshaped [256,128] → scale=√2 (wrong!)
    update = update * (max(1, grad.shape[-2] / grad.shape[-1]) ** 0.5)

    # Reshape back if it was 4D
    if len(orig_shape) == 4:
        update = update.reshape(orig_shape)

    return update, new_buf


class MuSGD:
    """Hybrid Muon + SGD optimizer for MLX.

    Port of PyTorch's MuSGD (ultralytics/optim/muon.py). For >=2D weight
    parameters, applies BOTH a Muon (Newton-Schulz orthogonalized) update and
    a Nesterov SGD update, scaled by muon_scale and sgd_scale respectively.
    For bias and normalization params, applies only Nesterov SGD.

    This matches the optimizer that 'optimizer=auto' resolves to in
    ultralytics when training yolo26 models.

    Args:
        model: MLX nn.Module whose parameters to optimize.
        lr: Learning rate.
        momentum: SGD momentum coefficient (default 0.9).
        weight_decay: L2 weight decay (default 0.0005).
        muon_scale: Scaling for Muon component (default 0.5).
        sgd_scale: Scaling for SGD component (default 0.5).
        nesterov: Use Nesterov momentum (default True).
    """

    def __init__(
        self,
        model,
        lr=0.000119,
        momentum=0.9,
        weight_decay=0.0005,
        muon_scale=0.5,
        sgd_scale=0.5,
        nesterov=True,
    ):
        """Initialize the MuSGD optimizer.

        Args:
            model: MLX nn.Module whose parameters to optimize.
            lr: Learning rate.
            momentum: SGD momentum coefficient.
            weight_decay: L2 regularization strength.
            muon_scale: Weight for the Muon (orthogonalized) update component.
            sgd_scale: Weight for the SGD update component.
            nesterov: Whether to use Nesterov momentum acceleration.
        """
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.muon_scale = muon_scale
        self.sgd_scale = sgd_scale
        self.nesterov = nesterov

        # Categorize parameters (matches PyTorch build_optimizer grouping)
        self._muon_paths = set()  # 2D+ weights → Muon + SGD
        self._weight_paths = set()  # other weights → SGD + decay
        self._bn_paths = set()  # BN/norm → SGD, no decay
        self._bias_paths = set()  # biases → SGD, no decay
        self._categorize_params(model)

        # Per-parameter LR scale (default 1.0 for all).
        # PyTorch applies lr*3 to fine-tuning parameters matching a regex.
        # _lr_scale[path] is the multiplier for that parameter.
        self._lr_scale = {}

        # State: momentum buffers per parameter path
        self._state = {}

    def _categorize_params(self, model):
        """Categorize model parameters into groups matching PyTorch behavior.

        Args:
            model: MLX nn.Module whose parameters will be grouped into
                muon, weight-decay, bias, and batch-norm sets.
        """
        for path, param in tree_flatten(model.parameters()):
            # Priority order MUST match PyTorch build_optimizer (trainer.py L958-965):
            #   1. ndim >= 2 → muon (Muon + SGD + decay)
            #   2. "bias" in name → bias (SGD, no decay)
            #   3. isinstance(module, BN) → bn (SGD, no decay)
            #   4. else → weight (SGD + decay)
            if param.ndim >= 2:
                self._muon_paths.add(path)
            elif "bias" in path:
                self._bias_paths.add(path)
            elif any(k in path for k in ["bn", "norm", "running_mean", "running_var"]):
                self._bn_paths.add(path)
            else:
                self._weight_paths.add(path)

    def _get_state(self, path, param, is_muon):
        """Get or initialize state for a parameter.

        Args:
            path: Dot-separated parameter path string (e.g. "model.0.conv.weight").
            param: The parameter array, used to initialize zero-filled momentum buffers.
            is_muon: If True, creates both Muon and SGD momentum buffers;
                otherwise only an SGD momentum buffer.

        Returns:
            State dict containing momentum buffer arrays ("muon_buf" and/or "sgd_buf").
        """
        if path not in self._state:
            if is_muon:
                self._state[path] = {
                    "muon_buf": mx.zeros_like(param),
                    "sgd_buf": mx.zeros_like(param),
                }
            else:
                self._state[path] = {
                    "sgd_buf": mx.zeros_like(param),
                }
        return self._state[path]

    @property
    def state(self):
        """Return flat list of all momentum buffer arrays for mx.eval synchronization."""
        arrays = []
        for s in self._state.values():
            for v in s.values():
                arrays.append(v)
        return arrays

    def set_lr_scale(self, model, pattern, scale):
        """Apply per-parameter LR scaling for paths matching a regex.

        Matches PyTorch's fine-tuning LR boost (trainer.py L993-1001):
        pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|flow_model")
        Matched parameters get lr * 3.

        In PyTorch, this is done by splitting each param group into two
        sub-groups (matched → lr*3, unmatched → lr) before creating the optimizer.
        Here we store per-path scale factors and apply them during step().

        Args:
            model: MLX model (used to enumerate parameter paths).
            pattern: Compiled regex or string pattern to match against param paths.
            scale: LR multiplier for matched parameters (e.g. 3.0).
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        for path, _ in tree_flatten(model.parameters()):
            if pattern.search(path):
                self._lr_scale[path] = scale

    def step(self, model, grads):
        """Perform one MuSGD optimization step.

        Matches PyTorch MuSGD.step():
        - For muon params (2D+ weights): apply Muon update, THEN SGD update (with decay)
        - For bias/BN params: SGD only, no decay
        - For other weight params: SGD + weight decay
        - Per-parameter LR scaling applied (fine-tuning boost)

        Args:
            model: MLX model to update.
            grads: Gradient tree (same structure as model.parameters()).
        """
        lr = self.learning_rate
        flat_params = dict(tree_flatten(model.parameters()))
        flat_grads = dict(tree_flatten(grads))

        updated = {}
        for path, param in flat_params.items():
            grad = flat_grads.get(path)
            if grad is None:
                updated[path] = param
                continue

            is_muon = path in self._muon_paths
            state = self._get_state(path, param, is_muon)

            # Per-parameter LR: base_lr * scale (fine-tuning boost)
            # Matches PyTorch per-group initial_lr (trainer.py L1001: lr * 3)
            plr = lr * self._lr_scale.get(path, 1.0)

            if is_muon:
                # --- Muon + SGD (for >=2D weights) ---
                # 1. Muon update (orthogonalized momentum)
                mu_update, state["muon_buf"] = muon_update(
                    grad,
                    state["muon_buf"],
                    beta=self.momentum,
                    nesterov=self.nesterov,
                )
                param = param - plr * self.muon_scale * mu_update

                # 2. SGD update with weight decay (on muon-updated param)
                grad_wd = grad + self.weight_decay * param
                state["sgd_buf"] = self.momentum * state["sgd_buf"] + grad_wd
                if self.nesterov:
                    sgd_update = grad_wd + self.momentum * state["sgd_buf"]
                else:
                    sgd_update = state["sgd_buf"]
                param = param - plr * self.sgd_scale * sgd_update

            elif path in self._bias_paths or path in self._bn_paths:
                # --- SGD only, NO weight decay ---
                state["sgd_buf"] = self.momentum * state["sgd_buf"] + grad
                if self.nesterov:
                    sgd_update = grad + self.momentum * state["sgd_buf"]
                else:
                    sgd_update = state["sgd_buf"]
                param = param - plr * sgd_update

            else:
                # --- Other weights: SGD + weight decay ---
                grad_wd = grad + self.weight_decay * param
                state["sgd_buf"] = self.momentum * state["sgd_buf"] + grad_wd
                if self.nesterov:
                    sgd_update = grad_wd + self.momentum * state["sgd_buf"]
                else:
                    sgd_update = state["sgd_buf"]
                param = param - plr * sgd_update

            updated[path] = param

        model.update(tree_unflatten(list(updated.items())))

    @staticmethod
    def auto_lr(nc=80, iterations=20):
        """Compute LR automatically from number of classes.

        Matches PyTorch: lr_fit = round(0.002 * 5 / (4 + nc), 6)
        Used when iterations <= 10000 (typical for short training runs).

        Args:
            nc: Number of classes.
            iterations: Total optimizer iterations.

        Returns:
            (lr, muon_scale, sgd_scale): LR and component scales.
        """
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        if iterations > 10000:
            lr = 0.01
            muon, sgd = 0.1, 1.0
        else:
            lr = lr_fit
            muon, sgd = 0.5, 0.5
        return lr, muon, sgd
