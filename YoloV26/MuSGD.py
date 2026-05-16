import torch
from torch.optim import Optimizer


def zeropower_via_newton_schulz(matrix, steps=5, eps=1e-7):
    """
    Muon-style orthogonalized update helper.

    Input:
        matrix = [out_features, in_features]

    Output:
        approximate orthogonalized matrix with same shape.
    """

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be a 2D matrix.")

    original_dtype = matrix.dtype
    x = matrix.float()

    if x.norm() == 0:
        return matrix

    transpose = False

    if x.shape[0] > x.shape[1]:
        x = x.T
        transpose = True

    x = x / (x.norm() + eps)

    for _ in range(steps):
        a = x @ x.T
        x = 1.5 * x - 0.5 * a @ x

    if transpose:
        x = x.T

    return x.to(dtype=original_dtype)


class MuSGD(Optimizer):
    """
    Clean MuSGD-style optimizer.

    It combines:
        - SGD-style momentum update
        - Muon-style orthogonalized update for matrix-like tensors

    This keeps your project self-contained while following the public YOLO26
    MuSGD direction.
    """

    def __init__(
        self,
        params,
        lr=0.0054,
        momentum=0.947,
        weight_decay=0.00064,
        muon_w=0.528,
        sgd_w=0.674,
        ns_steps=5,
        eps=1e-8,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "muon_w": muon_w,
            "sgd_w": sgd_w,
            "ns_steps": ns_steps,
            "eps": eps,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            muon_w = group["muon_w"]
            sgd_w = group["sgd_w"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(grad)

                momentum_update = state["momentum_buffer"]

                sgd_update = momentum_update

                use_muon = param.ndim >= 2 and momentum_update.numel() >= 16

                if use_muon:
                    original_shape = momentum_update.shape
                    matrix_update = momentum_update.reshape(original_shape[0], -1)

                    muon_update = zeropower_via_newton_schulz(
                        matrix=matrix_update,
                        steps=ns_steps,
                        eps=eps,
                    ).reshape(original_shape)

                    update = sgd_w * sgd_update + muon_w * muon_update
                else:
                    update = sgd_update

                param.add_(update, alpha=-lr)

        return loss