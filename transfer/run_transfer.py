"""Measure untargeted cross-model transfer on the official MNIST test split."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from attacks import attack
from model import PixelsToLogits
from models import build_model


NAMES = ("mlp", "cnn", "transformer")
EPSILONS = (0.05, 0.15)
METHODS = {"i-fgsm": 0.0, "mi-fgsm": 1.0}
STEPS = 10
BATCH = 256
THREADS = int(os.getenv("TRANSFER_CPU_THREADS", "4"))
NOISE_SEED = 20261115
NORMALIZATION = {"mean": 0.13067043953945248, "std": 0.30812228107625567}
# The validation winners of the earlier MNIST Recognition assignment.
SELECTION = {"mlp": (6, 8), "cnn": (5, 5), "transformer": (5, 8)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models():
    normalization = NORMALIZATION
    models, evidence = {}, {}
    for name in NAMES:
        trial, epoch = SELECTION[name]
        path = HERE / "checkpoints" / f"{name}-{trial}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert (ckpt["model"], ckpt["trial"], ckpt["epoch"]) == (name, trial, epoch)
        assert all(ckpt["normalization"][key] == value for key, value in normalization.items())
        network = build_model(name)
        network.load_state_dict(ckpt["state_dict"], strict=True)
        models[name] = PixelsToLogits(network, normalization["mean"], normalization["std"])
        models[name].eval().requires_grad_(False)
        evidence[name] = {
            "path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "parameters": sum(p.numel() for p in network.parameters()),
            "trial": trial, "epoch": epoch,
        }
    return models, evidence, normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results-reproduced.json")
    args = parser.parse_args()
    t0 = perf_counter()
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    models, evidence, normalization = load_models()
    data_root = args.data
    data = torchvision.datasets.MNIST(data_root, train=False, download=args.download)
    pixels = data.data.unsqueeze(1).float() / 255
    labels = data.targets

    def logits(model, images):
        with torch.no_grad():
            return model(images)

    clean_logits = {
        name: torch.cat([logits(model, x) for x in pixels.split(BATCH)])
        for name, model in models.items()
    }
    clean_correct = {name: score.argmax(1) == labels for name, score in clean_logits.items()}
    common = torch.stack(list(clean_correct.values())).all(0)
    ids = common.nonzero().flatten()
    n = len(ids)
    assert len(pixels) == 10000 and n > 0
    print(f"Common clean-correct: {n}/{len(pixels)}", flush=True)
    summary = defaultdict(Counter)
    checks = defaultdict(float)
    cosine = defaultdict(list)
    fgsm_alignment = defaultdict(Counter)
    diagnostic = defaultdict(Counter)
    zero_pixels = 0
    rng = torch.Generator().manual_seed(NOISE_SEED)

    @torch.no_grad()
    def mask_diagnostic(kind, adv, clean, y):
        # Keep the original attack components; do not re-optimize after masking.
        zero = clean == 0
        masked = {"zero_only": torch.where(zero, adv, clean),
                  "nonzero_only": torch.where(zero, clean, adv)}
        if kind == "random-sign":
            masked["uniform_zero_lift_0.075"] = torch.where(zero, 0.075, clean)
        for condition, variant in masked.items():
            for name, model in models.items():
                diagnostic[kind, condition][name] += int((model(variant).argmax(1) != y).sum())

    @torch.no_grad()
    def evaluate(x_adv, clean, y, clean_loss, key, eps, src=None):
        delta = (x_adv - clean).abs().max().item()
        assert x_adv.isfinite().all() and delta <= eps + 1e-6
        assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0
        checks[key] = max(checks[key], delta)
        predictions = {}
        for name, model in models.items():
            scores = model(x_adv)
            pred = scores.argmax(1)
            higher = F.cross_entropy(scores, y, reduction="none") > clean_loss[name]
            predictions[name] = pred
            s = summary[key + (name,)]
            s["misclassified"] += int((pred != y).sum())
            s["higher_true_label_loss"] += int(higher.sum())
        if src is not None:
            src_wrong = predictions[src] != y
            for dst in NAMES:
                if src == dst:
                    continue
                dst_wrong = predictions[dst] != y
                both = src_wrong & dst_wrong
                s = summary[key + (dst,)]
                s["both_misclassified"] += int(both.sum())
                s["both_misclassified_same_wrong_label"] += int((both & (predictions[src] == predictions[dst])).sum())

    for pos, batch in enumerate(ids.split(BATCH)):
        x, y = pixels[batch], labels[batch]
        zero_pixels += int((x == 0).sum())
        clean_loss = {name: F.cross_entropy(v[batch], y, reduction="none") for name, v in clean_logits.items()}
        grads = {}
        for name, model in models.items():
            xg = x.detach().requires_grad_(True)
            loss = F.cross_entropy(model(xg), y, reduction="sum")
            grads[name] = torch.autograd.grad(loss, xg)[0].flatten(1)
        for a, b in combinations(NAMES, 2):
            ga, gb = grads[a], grads[b]
            assert (ga.norm(dim=1) > 0).all() and (gb.norm(dim=1) > 0).all()
            cosine[a, b].append(F.cosine_similarity(ga, gb, dim=1, eps=1e-30))
        signs = (2 * torch.randint(2, x.shape, generator=rng) - 1).float()
        for eps in EPSILONS:
            noisy = (x + eps * signs).clamp(0, 1)
            evaluate(noisy, x, y, clean_loss, (eps, "random-sign", "none"), eps)
            if eps == 0.15:
                mask_diagnostic("random-sign", noisy, x, y)
            for src, model in models.items():
                for method, decay in METHODS.items():
                    adv = attack(model, x, y, "mi-fgsm", eps, steps=STEPS, decay=decay)
                    evaluate(adv, x, y, clean_loss, (eps, method, src), eps, src)
                    if eps == 0.15 and method == "mi-fgsm" and src == "cnn":
                        mask_diagnostic("cnn-mi-fgsm", adv, x, y)
                first_delta = ((x + eps * grads[src].reshape_as(x).sign()).clamp(0, 1) - x).flatten(1)
                for dst in NAMES:
                    if src != dst:
                        fgsm_alignment[eps, src, dst]["positive_dot_product"] += int(((grads[dst] * first_delta).sum(1) > 0).sum())
        if (pos + 1) % 4 == 0 or (pos + 1) * BATCH >= n:
            print(f"{min((pos+1)*BATCH,n):,}/{n:,} samples, {perf_counter()-t0:.1f}s", flush=True)

    out = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(perf_counter() - t0, 2),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "cpu_threads": THREADS,
        "checkpoints": evidence, "normalization": normalization,
        "dataset": "MNIST official test split", "test_examples": len(pixels),
        "test_file_sha256": {p.name: digest(p) for p in (data_root / "MNIST/raw").glob("t10k-*-ubyte")},
        "clean_correct_counts": {name: int(ok.sum()) for name, ok in clean_correct.items()},
        "common_clean_correct_count": n, "common_clean_index_sha256_int64le": hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest(),
        "attack": {
            "objective": "maximize cross entropy on true label; success is any wrong prediction",
            "threat_model": "source-only gradients; targets only score frozen adversarial examples",
            "domain": "L-infinity budget in raw [0,1] pixels; normalization inside each model",
            "steps": STEPS, "alpha": "epsilon / 4", "initialization": "clean", "restarts": 0,
            "momentum_by_method": METHODS, "epsilons": EPSILONS, "random_sign_seed": NOISE_SEED,
        },
        "transfer": [
            {"epsilon": eps, "method": method, "source": src, "target": dst,
             "max_linf_observed": checks[eps, method, src], "attack_success_rate": s["misclassified"] / n,
             **s}
            for (eps, method, src, dst), s in sorted(summary.items())
        ],
        "clean_gradient_cosine": [
            {"pair": [a, b], "mean": float(v.mean()), "median": float(v.median()),
             "fraction_positive": float((v > 0).float().mean())}
            for (a, b), chunks in cosine.items() for v in [torch.cat(chunks)]
        ],
        "clean_gradient_dot_clipped_source_fgsm": [
            {"epsilon": eps, "source": src, "target": dst, **s}
            for (eps, src, dst), s in sorted(fgsm_alignment.items())
        ],
        "pixel_mask_diagnostic": {
            "epsilon": 0.15, "clean_zero_pixel_fraction": zero_pixels / (n * 28 * 28),
            "conditions": [
                {"perturbation": kind, "condition": condition, "error_counts": dict(counts)}
                for (kind, condition), counts in sorted(diagnostic.items())
            ],
        },
    }
    assert not any(p.grad is not None for net in models.values() for p in net.parameters())
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
