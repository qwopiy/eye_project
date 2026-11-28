"""Langkah 3b: kontak-sheet gambar per kelas untuk pemeriksaan visual arah label. Hasil: preview/<split>_<kelas>.jpg

Contoh:  python scripts/01c_preview.py            (split train)
         python scripts/01c_preview.py --split real
Yang harus Anda lihat:
  left  : wajah menoleh ke KIRI-nya sendiri  -> hidung bergeser ke KANAN gambar (kamera tidak dicermin)
  right : wajah menoleh ke KANAN-nya sendiri -> hidung bergeser ke KIRI gambar
  up    : dagu naik (kepala mendongak)       down: dagu turun (menunduk)
Jika terbalik, JANGAN lanjut training: periksa 01b (kalibrasi) atau cek apakah gambar sumber dicermin.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from common import IMG_EXT
from config import CLASSES


def sheet(files, cols=8, rows=4, cell=112, title=""):
    canvas = np.full((rows * cell + 28, cols * cell, 3), 30, np.uint8)
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    for i, p in enumerate(files[: cols * rows]):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (cell, cell))
        r, c = divmod(i, cols)
        canvas[28 + r * cell: 28 + (r + 1) * cell, c * cell:(c + 1) * cell] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    Path(args.out).mkdir(exist_ok=True)
    for c in CLASSES:
        d = Path(args.data) / args.split / c
        files = [p for p in d.glob("*") if p.suffix.lower() in IMG_EXT] if d.is_dir() else []
        if not files:
            print(f"{c}: kosong")
            continue
        rng.shuffle(files)
        out = Path(args.out) / f"{args.split}_{c}.jpg"
        cv2.imwrite(str(out), sheet(files, title=f"{args.split} / {c}  ({len(files)} gambar)"))
        print(f"{c}: {len(files)} gambar -> {out}")


if __name__ == "__main__":
    main()
