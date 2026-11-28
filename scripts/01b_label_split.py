"""Langkah 3: kalibrasi sudut -> label 5 kelas -> bagi per orang/rekaman -> data/prep (train/val/test + aflw).

Contoh:
  python scripts/01b_label_split.py
  python scripts/01b_label_split.py --cap 40 --val 0.15 --test 0.15

Yang dilakukan:
  1. Baca data/crops/*.csv (hasil 01a).
  2. Per dataset, KALIBRASI otomatis: kolom sudut mana yang yaw/pitch dan tandanya, dengan mencocokkan
     sudut dataset ke proksi hidung (posisi hidung terhadap garis mata). Hasilnya dicetak: PERIKSA.
  3. Beri label (ambang di config.py); sudut zona abu-abu, diagonal, atau terlalu ekstrem dibuang.
  4. Bagi per GRUP (orang untuk HPID, rekaman untuk BIWI) supaya orang yang sama tidak ada di train dan test.
  5. Jika ada dataset selain AFLW2000-3D, AFLW tidak dilatih: seluruhnya masuk split 'aflw' (uji lintas-dataset).
     Jika AFLW2000-3D SATU-SATUNYA dataset (mode "aflw-only"), AFLW dibagi per gambar (berstrata per kelas)
     menjadi train/val/test, karena AFLW tidak punya ID orang. Di mode ini train juga diperbanyak dengan salinan
     CERMIN (kiri <-> kanan ditukar): wajah yang menoleh ke kiri, bila dicerminkan, menjadi menoleh ke kanan.
  6. Batasi --cap gambar per (grup, kelas) agar tidak didominasi frame yang hampir sama (tidak berlaku untuk AFLW).
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, MAX_ANGLE
from headpose import apply_calibration, calibrate, label_from_angles

MIN_CORR = 0.5


def load_rows(crops_dir):
    rows = []
    for p in sorted(Path(crops_dir).glob("*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("a0", "a1", "a2", "px", "py"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def calibrate_dataset(rows):
    """rows satu dataset -> (cal, yaw[], pitch[])."""
    ds = np.array([[r["a0"], r["a1"], r["a2"]] for r in rows])
    ds = ds[:, ~np.isnan(ds).any(axis=0)]                     # buang kolom kosong (HPID hanya 2 sudut)
    px = np.array([r["px"] for r in rows]); py = np.array([r["py"] for r in rows])
    use = (np.abs(ds) <= MAX_ANGLE).all(axis=1)               # kalibrasi pada sudut wajar saja
    cal = calibrate(ds[use], px[use], py[use])
    yaw, pitch = apply_calibration(ds, cal)
    return cal, yaw, pitch


def split_groups(sizes, val, test, seed):
    """Bagi grup ke train/val/test mendekati proporsi jumlah gambar. Grup tidak pernah terpecah."""
    rng = random.Random(seed)
    items = list(sizes.items()); rng.shuffle(items)
    items.sort(key=lambda kv: -kv[1])
    total = sum(sizes.values())
    target = {"train": 1 - val - test, "val": val, "test": test}
    cur = {k: 0 for k in target}; out = {}
    for g, n in items:
        k = max(target, key=lambda s: target[s] * total - cur[s])
        out[g] = k; cur[k] += n
    return out


def stratified_split(items, val, test, seed):
    """items: daftar (row, kelas). Bagi per KELAS ke train/val/test. Mengembalikan daftar (row, kelas, split).
    Kelas yang sangat kecil (< 5 gambar) seluruhnya masuk train supaya tetap bisa dilatih."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r, c in items:
        by_class[c].append(r)
    out = []
    for c, rs in by_class.items():
        rs = sorted(rs, key=lambda r: r["crop"])
        rng.shuffle(rs)
        n = len(rs)
        n_test = max(1, round(n * test)) if n >= 5 else 0
        n_val = max(1, round(n * val)) if n >= 5 else 0
        for i, r in enumerate(rs):
            out.append((r, c, "test" if i < n_test else "val" if i < n_test + n_val else "train"))
    return out


MIRROR = {"left": "right", "right": "left", "center": "center", "up": "up", "down": "down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--cap", type=int, default=60, help="maks. gambar per (grup, kelas)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="lanjut walau kalibrasi tidak yakin")
    ap.add_argument("--flip-train", choices=["auto", "yes", "no"], default="auto",
                    help="tambah salinan cermin (kiri<->kanan) di train. auto = ya hanya pada mode aflw-only")
    args = ap.parse_args()

    rows = load_rows(args.crops)
    if not rows:
        raise SystemExit(f"Tidak ada CSV di {args.crops}. Jalankan 01a_extract_crops.py dulu.")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    rng = random.Random(args.seed)
    labeled = []                                              # (row, kelas)
    print("=== Kalibrasi sudut per dataset (PERIKSA) ===")
    for name, rs in by_ds.items():
        cal, yaw, pitch = calibrate_dataset(rs)
        ok = min(cal["yaw_corr"], cal["pitch_corr"]) >= MIN_CORR
        print(f"[{name}] {len(rs)} crop | yaw = {cal['yaw_sign']:+.0f} x kolom a{cal['yaw_col']} (korelasi {cal['yaw_corr']:.2f})"
              f" | pitch = {cal['pitch_sign']:+.0f} x kolom a{cal['pitch_col']} (korelasi {cal['pitch_corr']:.2f})"
              + ("" if ok else "   <-- LEMAH"))
        if not ok and not args.force:
            raise SystemExit(f"Kalibrasi {name} tidak yakin (korelasi < {MIN_CORR}). Cek 01a/--src, atau pakai --force.")
        for r, y, p in zip(rs, yaw, pitch):
            c = label_from_angles(y, p)
            if c is not None:
                labeled.append((r, c))

    # batasi per (grup, kelas)
    bucket = defaultdict(list)
    for r, c in labeled:
        bucket[(r["dataset"], r["group"], c)].append(r)
    kept = []
    for (ds, g, c), rs in bucket.items():
        rng.shuffle(rs)
        for r in (rs if ds == "aflw" else rs[:args.cap]):
            kept.append((r, c))

    # tentukan split
    aflw_only = set(by_ds) == {"aflw"}
    final = []                                                # (row, kelas, split)
    assign = {}
    if aflw_only:
        print("\n[MODE aflw-only] AFLW2000-3D dibagi per gambar (tanpa ID orang), berstrata per kelas.")
        final = stratified_split(kept, args.val, args.test, args.seed)
    else:
        for ds in by_ds:
            if ds == "aflw":
                continue
            sizes = Counter(r["group"] for r, _ in kept if r["dataset"] == ds)
            if not sizes:
                continue
            sp = split_groups(sizes, args.val, args.test, args.seed)
            if len(sizes) < 3:
                print(f"[PERINGATAN] {ds}: hanya {len(sizes)} grup, split tidak seimbang")
            for g, s_ in sp.items():
                assign[(ds, g)] = s_
        for r, c in kept:
            split = "aflw" if r["dataset"] == "aflw" else assign.get((r["dataset"], r["group"]))
            if split is not None:
                final.append((r, c, split))

    out = Path(args.out)
    if out.exists():
        for s_ in ("train", "val", "test", "aflw"):
            shutil.rmtree(out / s_, ignore_errors=True)
    flip = args.flip_train == "yes" or (args.flip_train == "auto" and aflw_only)
    table = defaultdict(Counter)
    for r, c, split in final:
        d = out / split / c
        d.mkdir(parents=True, exist_ok=True)
        name = f"{r['dataset']}_{r['group']}_{Path(r['crop']).stem}.jpg"
        shutil.copy(Path(args.crops) / r["crop"], d / name)
        table[split][c] += 1
        if flip and split == "train":                          # salinan cermin dengan label kiri/kanan ditukar
            im = cv2.imread(str(Path(args.crops) / r["crop"]))
            if im is not None:
                dm = out / split / MIRROR[c]
                dm.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dm / ("flip_" + name)), cv2.flip(im, 1), [cv2.IMWRITE_JPEG_QUALITY, 95])
                table[split][MIRROR[c]] += 1

    print("\n=== Ringkasan data/prep ===" + ("  (train sudah termasuk salinan cermin)" if flip else ""))
    print(f"{'split':8}" + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>9}")
    for s_ in ("train", "val", "test", "aflw"):
        if s_ in table:
            print(f"{s_:8}" + "".join(f"{table[s_][c]:>9}" for c in CLASSES) + f"{sum(table[s_].values()):>9}")
    if not aflw_only:
        print("\nGrup per split:")
        for s_ in ("train", "val", "test"):
            print(f"  {s_}: " + ", ".join(sorted(f"{d}/{g}" for (d, g), v in assign.items() if v == s_)))
    min_train, min_eval = (60, 8) if aflw_only else (150, 30)
    for c in CLASSES:
        if table["train"][c] < min_train:
            print(f"[PERINGATAN] kelas '{c}' hanya {table['train'][c]} gambar di train (disarankan >= {min_train})")
        for s_ in ("val", "test"):
            if aflw_only and table[s_][c] < min_eval:
                print(f"[PERINGATAN] kelas '{c}' hanya {table[s_][c]} gambar di {s_} (disarankan >= {min_eval}); angka kelas ini kurang bisa dipercaya")
        if not aflw_only and table["val"][c] < min_eval:
            print(f"[PERINGATAN] kelas '{c}' hanya {table['val'][c]} gambar di val (disarankan >= {min_eval})")
    print("\nLangkah berikut: python scripts/01c_preview.py   (lihat gambar per kelas, PASTIKAN arahnya benar)")


if __name__ == "__main__":
    main()
