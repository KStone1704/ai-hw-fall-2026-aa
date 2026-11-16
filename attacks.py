"""Untargeted white-box L-infinity attacks in [0, 1] pixel space."""

import torch
from torch.nn import functional as F

METHODS = ("fgsm", "pgd", "mi-fgsm")


def attack(model, clean, labels, method, eps, *, steps=10, seed=123, decay=1.0):
    if method not in METHODS or not 0 <= eps <= 1 or steps < 1:
        raise ValueError("invalid method, epsilon, or step count")
    base = clean.detach()
    if eps == 0:
        return base.clone()
    count = 1 if method == "fgsm" else steps
    step_size = eps if method == "fgsm" else eps / 4
    lower = (base - eps).clamp(0, 1)
    upper = (base + eps).clamp(0, 1)

    def project(x):
        return torch.maximum(lower, torch.minimum(x, upper))

    current = base.clone()
    if method == "pgd":
        rng = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.rand(base.shape, generator=rng, dtype=base.dtype).to(base.device)
        current = project(base + eps * (2 * noise - 1))
    velocity = torch.zeros_like(base) if method == "mi-fgsm" else None

    for _ in range(count):
        current = current.detach().requires_grad_(True)
        loss = F.cross_entropy(model(current), labels, reduction="sum")
        grad, = torch.autograd.grad(loss, current)
        with torch.no_grad():
            if velocity is not None:
                l1 = grad.abs().flatten(1).sum(1).clamp_min(1e-12)
                velocity = decay * velocity + grad / l1.view(-1, 1, 1, 1)
                direction = velocity
            else:
                direction = grad
            current = project(current + step_size * direction.sign())
    return current.detach()
