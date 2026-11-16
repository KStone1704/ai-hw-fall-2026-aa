"""Fit the selected CNN recipe using only the official MNIST training split."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from model import CNN, PixelsToLogits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", type=Path, default=Path("trained-cnn.pt"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new filename")

    torch.set_num_threads(4)
    torch.manual_seed(42)
    data = MNIST(args.data, train=True, download=args.download)
    split = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
    train_ids, valid_ids = split[:55000], split[55000:]
    # Integer histogram gives float64 moments without holding all pixels in float64.
    hist = torch.bincount(data.data[train_ids].flatten().long(), minlength=256).double()
    levels = torch.arange(256, dtype=torch.float64) / 255
    mean = (hist * levels).sum().item() / hist.sum().item()
    std = ((hist * (levels - mean).square()).sum() / hist.sum()).sqrt().item()

    device = torch.device(args.device)
    cnn = CNN()
    model = PixelsToLogits(cnn, mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    order_rng = torch.Generator().manual_seed(42)

    def batch(ids):
        return (data.data[ids].unsqueeze(1).to(device, dtype=torch.float32) / 255,
                data.targets[ids].to(device))

    for ep in range(1, 6):
        model.train()
        for ids in train_ids[torch.randperm(len(train_ids), generator=order_rng)].split(256):
            x, y = batch(ids)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            for ids in valid_ids.split(256):
                x, y = batch(ids)
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {ep}: validation {correct}/{len(valid_ids)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": "cnn", "epoch": 5, "model_state": cnn.state_dict(),
                "mean": mean, "std": std}, args.output)


if __name__ == "__main__":
    main()
