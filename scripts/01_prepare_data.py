"""Langkah 1: siapkan dataset Mendeley menjadi data latih.

Yang dilakukan:
  1. Baca gambar mentah per folder kelas.
  2. Kelompokkan gambar yang mirip (near-duplicate) dengan hash gambar. Dataset ini berisi banyak
     frame hampir identik dari video, jadi tanpa langkah ini hasil evaluasi akan terlalu bagus.
  3. Batasi jumlah gambar per kelompok (--max-per-cluster).
  4. Bagi data ke train/val/test PER KELOMPOK (bukan per gambar) agar tidak bocor.
  5. Pad ke persegi, resize ke 96x96, simpan sebagai JPEG, tulis meta.csv.

Contoh:
  python scripts/01_prepare_data.py --raw data/mendeley_raw --out data/prep
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASSES, FOLDER_TO_CLASS, IMG_SIZE
from common import IMG_EXT


def dhash(img, size=16):
    """Hash perbedaan 256-bit (size*size) dari gambar abu-abu. Dikembalikan sebagai int Python."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def cluster_hashes(hashes, thr):
    """Klaster greedy: gambar masuk klaster pertama yang jarak Hamming-nya <= thr."""
    reps, labels = [], []
    for h in hashes:
        for k, r in enumerate(reps):
            if (h ^ r).bit_count() <= thr:
                labels.append(k)
                break
        else:
            reps.append(h)
            labels.append(len(reps) - 1)
    return labels


def pad_square_resize(img, size):
    """Pad tepi (replikasi piksel tepi) sampai persegi, lalu resize. Tidak ada bagian yang terpotong."""
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    a = np.pad(a, ((top, s - h - top), (left, s - w - left), (0, 0)), mode="edge")
    return Image.fromarray(a).resize((size, size), Image.LANCZOS)


def assign_splits(cluster_sizes, rng, frac_val, frac_test):
    """Bagi klaster ke split sehingga proporsi jumlah gambar mendekati target."""
    ids = list(cluster_sizes)
    rng.shuffle(ids)
    total = sum(cluster_sizes.values())
    target = {"test": frac_test * total, "val": frac_val * total}
    got = defaultdict(int)
    out = {}
    for cid in ids:
        if got["test"] < target["test"]:
            sp = "test"
        elif got["val"] < target["val"]:
            sp = "val"
        else:
            sp = "train"
        out[cid] = sp
        got[sp] += cluster_sizes[cid]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder dataset mentah (berisi folder per kelas)")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--hash-thr", type=int, default=8, help="batas jarak Hamming (dari 256 bit) untuk 'mirip'")
    ap.add_argument("--max-per-cluster", type=int, default=5)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw, out = Path(args.raw), Path(args.out)

    found = {p.name for p in raw.iterdir() if p.is_dir()}
    print("Folder ditemukan:", sorted(found))
    rows = []  # (class, orig_path, cluster_id, split)
    for folder, cls in FOLDER_TO_CLASS.items():
        d = raw / folder
        if not d.is_dir():
            print(f"[PERINGATAN] folder '{folder}' (kelas {cls}) tidak ada di {raw}, dilewati")
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        hashes, ok_files = [], []
        for p in files:
            try:
                with Image.open(p) as im:
                    hashes.append(dhash(im))
                ok_files.append(p)
            except Exception as e:  # gambar rusak
                print("  lewati", p.name, e)
        labels = cluster_hashes(hashes, args.hash_thr)
        by_cluster = defaultdict(list)
        for p, c in zip(ok_files, labels):
            by_cluster[c].append(p)
        # batasi per klaster (pilih acak, reproducible)
        kept = {}
        for c, ps in by_cluster.items():
            ps = sorted(ps)
            rng.shuffle(ps)
            kept[c] = ps[: args.max_per_cluster]
        sizes = {c: len(v) for c, v in kept.items()}
        split_of = assign_splits(sizes, rng, args.val, args.test)
        n_kept = sum(sizes.values())
        print(f"{cls:>7}: {len(ok_files)} gambar -> {len(by_cluster)} klaster -> {n_kept} dipakai")
        for c, ps in kept.items():
            for p in ps:
                rows.append((cls, p, c, split_of[c]))

    if not rows:
        raise SystemExit("Tidak ada gambar yang diproses. Periksa --raw dan FOLDER_TO_CLASS di config.py")

    # tulis gambar + meta
    out.mkdir(parents=True, exist_ok=True)
    counters = defaultdict(int)
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "split", "cluster", "orig_name", "orig_w", "orig_h"])
        for cls, p, c, sp in rows:
            with Image.open(p) as im:
                ow, oh = im.size
                im2 = pad_square_resize(im, IMG_SIZE)
            d = out / sp / cls
            d.mkdir(parents=True, exist_ok=True)
            counters[(sp, cls)] += 1
            dst = d / f"{cls}_{c:04d}_{counters[(sp, cls)]:05d}.jpg"
            im2.save(dst, quality=95)
            w.writerow([str(dst.relative_to(out)), cls, sp, f"{cls}-{c}", p.name, ow, oh])

    print("\nRingkasan jumlah gambar:")
    print(f"{'split':<7}" + "".join(f"{c:>8}" for c in CLASSES) + f"{'total':>8}")
    for sp in ["train", "val", "test"]:
        vals = [counters.get((sp, c), 0) for c in CLASSES]
        print(f"{sp:<7}" + "".join(f"{v:>8}" for v in vals) + f"{sum(vals):>8}")
    print(f"\nSelesai. Data siap di: {out}  (meta: {out / 'meta.csv'})")


if __name__ == "__main__":
    main()
