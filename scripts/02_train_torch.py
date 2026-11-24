"""Langkah 2a: latih MobileNetV3-Small atau EfficientNet-B0 (PyTorch).

Contoh:
  python scripts/02_train_torch.py --arch mobilenetv3s --seed 1
  python scripts/02_train_torch.py --arch efficientnet_b0 --seed 1

Keluaran: runs/<arch>_s<seed>/best.pt  (+ history.json, labels.json)
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from common import accuracy, list_split, macro_f1
from config import CLASSES, IMG_SIZE

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RandomDegrade:
    """Meniru kualitas webcam: perkecil lalu perbesar lagi, dan kompresi JPEG acak."""

    def __call__(self, img):
        if random.random() < 0.7:
            s = random.choice([32, 40, 48, 64])
            img = img.resize((s, s), Image.BILINEAR).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        if random.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(30, 85))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


def train_tf():
    # TANPA flip horizontal: flip akan menukar makna kiri/kanan.
    return T.Compose([
        RandomDegrade(),
        T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.3),
        T.ToTensor(),
    ])


def eval_tf():
    return T.Compose([T.ToTensor()])


class EyeDataset(Dataset):
    def __init__(self, data_dir, split, tf):
        self.items = list_split(data_dir, split)
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


class Wrapped(nn.Module):
    """Normalisasi (mean/std) ada DI DALAM model, jadi input model selalu RGB 0-1."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build(arch, n_classes, pretrained):
    if arch == "mobilenetv3s":
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=w)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_classes)
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=w)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"arsitektur tidak dikenal: {arch}")
    return Wrapped(net)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenetv3s", choices=["mobilenetv3s", "efficientnet_b0"])
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out or f"runs/{args.arch}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    tr = EyeDataset(args.data, "train", train_tf())
    va = EyeDataset(args.data, "val", eval_tf())
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("Data train/val kosong. Jalankan 01_prepare_data.py dulu.")
    present = sorted({y for _, y in tr.items})
    print(f"device={device} train={len(tr)} val={len(va)} kelas ada di train: {[CLASSES[i] for i in present]}")

    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=False)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model = build(args.arch, len(CLASSES), not args.no_pretrained).to(device)

    # bobot kelas = kebalikan frekuensi (kelas yang tidak ada di train diberi bobot 0)
    counts = np.bincount([y for _, y in tr.items], minlength=len(CLASSES)).astype(float)
    w = np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, bad, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        ys, ps = evaluate(model, vl, device)
        f1, acc = macro_f1(ys, ps, len(CLASSES)), accuracy(ys, ps)
        hist.append({"epoch": ep, "loss": tot / n, "val_f1": f1, "val_acc": acc})
        print(f"ep {ep:02d} loss {tot / n:.4f} val_acc {acc:.3f} val_macroF1 {f1:.3f} ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            torch.save({"state": model.state_dict(), "arch": args.arch, "classes": CLASSES}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    json.dump(hist, open(out / "history.json", "w"), indent=1)
    json.dump({"classes": CLASSES, "arch": args.arch, "input_size": IMG_SIZE, "best_val_f1": best},
              open(out / "labels.json", "w"), indent=1)
    print(f"selesai. val macro-F1 terbaik = {best:.3f}; model: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
