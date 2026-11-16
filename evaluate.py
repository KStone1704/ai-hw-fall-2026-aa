"""Measure clean accuracy, robust accuracy and conditional attack success."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torchvision.datasets import MNIST

from attacks import METHODS, attack
from model import CNN, PixelsToLogits

EPS = (0.05, 0.15, 0.30)
PGD_SEEDS = (123, 456, 789)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--checkpoint", type=Path, default=Path("cnn-reference.pt"))
    p.add_argument("--output", type=Path, default=Path("results.json"))
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    cnn = CNN()
    cnn.load_state_dict(checkpoint["model_state"])
    model = PixelsToLogits(cnn, checkpoint["mean"], checkpoint["std"]).to(device).eval()
    model.requires_grad_(False)
    data = MNIST(args.data, train=False, download=args.download)
    results = []
    clean_count = 0

    def get_batch(i):
        x = data.data[i:i+256].unsqueeze(1).to(device, dtype=torch.float32) / 255
        return x, data.targets[i:i+256].to(device)

    with torch.no_grad():
        for i in range(0, len(data), 256):
            x, y = get_batch(i)
            clean_count += (model(x).argmax(1) == y).sum().item()

    for eps in EPS:
        for method in METHODS:
            start = time.perf_counter()
            correct = originally_correct_attacked = recovered = 0
            max_delta = 0.0
            restarts = PGD_SEEDS if method == "pgd" else (123,)
            for i in range(0, len(data), 256):
                x, y = get_batch(i)
                with torch.no_grad():
                    originally_correct = model(x).argmax(1) == y
                    still_correct = torch.ones_like(originally_correct)
                for seed in restarts:
                    adv = attack(model, x, y, method, eps, seed=seed + i)
                    if not torch.isfinite(adv).all() or adv.min().item() < 0 or adv.max().item() > 1:
                        raise RuntimeError("Invalid adversarial pixels")
                    max_delta = max(max_delta, (adv - x).abs().max().item())
                    with torch.no_grad():
                        still_correct &= model(adv).argmax(1) == y
                correct += still_correct.sum().item()
                originally_correct_attacked += (originally_correct & ~still_correct).sum().item()
                recovered += (~originally_correct & still_correct).sum().item()
            if max_delta > eps + 1e-6 or correct != clean_count - originally_correct_attacked + recovered:
                raise RuntimeError("Budget or count consistency check failed")
            row = {"method": method, "epsilon": eps, "restarts": len(restarts),
                   "steps_per_restart": 1 if method == "fgsm" else 10,
                   "step_size": eps if method == "fgsm" else eps / 4,
                   "correct_after_every_restart": correct, "clean_correct_fooled": originally_correct_attacked,
                   "clean_wrong_correct_after_every_restart": recovered,
                   "adversarial_accuracy": correct / len(data),
                   "attack_success_rate": originally_correct_attacked / clean_count,
                   "max_linf_observed": max_delta, "seconds": time.perf_counter() - start}
            results.append(row)
            print(f"eps={eps:.2f} {method:7} accuracy={row['adversarial_accuracy']:.2%} ASR={row['attack_success_rate']:.2%}", flush=True)
    payload = {
        "dataset": "MNIST official test split", "test_size": len(data),
        "device": str(device), "torch_version": str(torch.__version__),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "normalization": {"mean": checkpoint["mean"], "std": checkpoint["std"]},
        "threat_model": "untargeted, white-box, L-infinity, [0,1] pixel space",
        "clean_correct": clean_count, "clean_accuracy": clean_count / len(data),
        "pgd_random_start_seeds_before_batch_offset": list(PGD_SEEDS),
        "mi_fgsm_momentum_decay": 1.0, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
