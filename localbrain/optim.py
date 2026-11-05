import math
import torch


class CPUAdamW(torch.optim.Optimizer):
    """Keep Adam's two moment tensors in CPU RAM; compute model ops on its device.

    FP32 model weights stay on the selected device. Each parameter gradient is
    copied to CPU and its update copied back. This trades transfer time for VRAM.
    """
    def __init__(self, params, lr=3e-4, weight_decay=0.01,
                 betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                     betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPUAdamW does not support closures")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().to(device="cpu", dtype=torch.float32)
                if grad.is_sparse:
                    raise ValueError("Sparse gradients are unsupported")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                state["step"] += 1
                step = int(state["step"])
                avg, sq = state["exp_avg"], state["exp_avg_sq"]
                avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = sq.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                update = avg.div(denom).mul_(group["lr"] / (1 - beta1 ** step))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.sub_(update.to(device=p.device, dtype=p.dtype))
        return None

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(device="cpu", dtype=torch.float32)
            if "step" in state:
                state["step"] = int(state["step"])


def orthogonalize(g, steps=5):
    """Quintic Newton-Schulz iteration: approximately U V^T of the gradient's SVD."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return x.T if transpose else x


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalized updates for 2-D hidden weight matrices (Keller Jordan's Muon).

    One momentum buffer per parameter (Adam keeps two). Embeddings, norms and the
    output head must use a different optimizer; see build_optimizer in train.py.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon does not support closures")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(group["momentum"]).add_(p.grad)
                update = p.grad.add(buf, alpha=group["momentum"])  # Nesterov
                update = orthogonalize(update)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)
        return None


class Combined:
    """Several optimizers stepped together, saved/loaded as one state dict."""
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        for o, s in zip(self.optimizers, state["combined"]):
            o.load_state_dict(s)
