"""Langkah 2: dari foto dataset publik -> crop wajah (MediaPipe) + sudut dataset + proksi pose. Hasil: data/crops/.

Contoh (jalankan sekali per dataset):
  python scripts/01a_extract_crops.py --dataset hpid --src data/raw/HeadPoseImageDatabase
  python scripts/01a_extract_crops.py --dataset biwi --src data/raw/kinect_head_pose_db --step 5
  python scripts/01a_extract_crops.py --dataset aflw --src data/raw/AFLW2000     <- yang dipakai pada panduan AFLW-only

Keluaran:
  data/crops/<dataset>/<grup>/<nama>.jpg   crop wajah FACE_SIZE x FACE_SIZE (crop yang sama dengan saat game berjalan)
  data/crops/<dataset>.csv                 dataset,grup,sumber,crop,a0,a1,a2,px,py   (a0..a2 = sudut asli dataset)
Skrip boleh dihentikan dan dijalankan lagi: gambar yang sudah diproses dilewati.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from config import IMG_SIZE
from face_crop import MARGIN, face_crop, make_landmarker
from headpose import parse_hpid_name, pose_proxy, read_aflw_pose, read_biwi_pose

FIELDS = ["dataset", "group", "src", "crop", "a0", "a1", "a2", "px", "py"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- penelusur dataset
def iter_hpid(src, step=1):
    """Head Pose Image Database / Pointing'04: nama berkas memuat orang + dua sudut."""
    files = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in IMG_EXT and parse_hpid_name(p.name))
    if not files:
        raise SystemExit(f"Tidak ada berkas 'personneXXXXX+A+B.jpg' di {src}. Cek jalur --src.")
    for i, p in enumerate(files):
        if i % step:
            continue
        person, series, num, a, b = parse_hpid_name(p.name)
        yield p, f"P{person}", (a, b, np.nan)


def iter_biwi(src, step=5):
    """BIWI Kinect: <urutan>/frame_00003_rgb.png + frame_00003_pose.txt. Grup = nomor urutan (rekaman)."""
    files = sorted(Path(src).rglob("frame_*_rgb.png"))
    if not files:
        raise SystemExit(f"Tidak ada 'frame_*_rgb.png' di {src}. Cek jalur --src (folder yang berisi 01, 02, ...).")
    for p in files:
        n = int(re.search(r"frame_(\d+)_rgb", p.name).group(1))
        if n % step:
            continue
        pose = p.with_name(p.name.replace("_rgb.png", "_pose.txt"))
        if not pose.exists():
            continue
        yield p, f"S{p.parent.name}", read_biwi_pose(pose)


def iter_aflw(src, step=1):
    """AFLW2000-3D: imageNNNNN.jpg + imageNNNNN.mat (Pose_Para). Semua masuk satu grup "A" (AFLW tidak punya ID orang)."""
    mats = sorted(Path(src).rglob("*.mat"))
    if not mats:
        raise SystemExit(f"Tidak ada berkas .mat di {src}. Versi unduhan ini mungkin tanpa anotasi sudut.")
    for i, m in enumerate(mats):
        if i % step:
            continue
        img = next((m.with_suffix(e) for e in (".jpg", ".png", ".jpeg") if m.with_suffix(e).exists()), None)
        if img is None:
            continue
        try:
            yield img, "A", read_aflw_pose(m)
        except Exception:
            continue


ITERS = {"hpid": iter_hpid, "biwi": iter_biwi, "aflw": iter_aflw}


# ---------------------------------------------------------------- inti
def extract(dataset, samples, detector, out_root, size=IMG_SIZE, margin=MARGIN, log_every=200):
    """samples: iterator (path_gambar, grup, (a0,a1,a2)). detector(bgr) -> daftar landmark atau None."""
    out_root = Path(out_root)
    csv_path = out_root / f"{dataset}.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            done = {r["src"] for r in csv.DictReader(f)}
    new_file = not csv_path.exists()
    n_ok = n_fail = n_skip = 0
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            wr.writeheader()
        for i, (path, group, ang) in enumerate(samples):
            if str(path) in done:
                n_skip += 1
                continue
            bgr = cv2.imread(str(path))
            lm = detector(bgr) if bgr is not None else None
            if lm is None:
                n_fail += 1
                continue
            h, w = bgr.shape[:2]
            prox = pose_proxy(lm, w, h)
            crop = face_crop(bgr, lm, size, margin)
            if prox is None or crop is None:
                n_fail += 1
                continue
            rel = Path(dataset) / group / (Path(path).stem + ".jpg")
            (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_root / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            wr.writerow({"dataset": dataset, "group": group, "src": str(path), "crop": str(rel),
                         "a0": ang[0], "a1": ang[1], "a2": ang[2], "px": prox[0], "py": prox[1]})
            n_ok += 1
            if (n_ok + n_fail) % log_every == 0:
                f.flush()
                print(f"  ... {n_ok} berhasil, {n_fail} gagal (wajah tak terdeteksi)")
    total = n_ok + n_fail
    print(f"[{dataset}] selesai: {n_ok} crop tersimpan, {n_fail} gagal"
          + (f" ({100 * n_fail / total:.0f}%)" if total else "") + (f", {n_skip} dilewati (sudah ada)" if n_skip else ""))
    return n_ok, n_fail


def mediapipe_detector(landmarker_path):
    import mediapipe as mp
    lm = make_landmarker(landmarker_path, video=False)

    def detect(bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return res.face_landmarks[0] if res.face_landmarks else None
    return detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ITERS))
    ap.add_argument("--src", required=True, help="folder hasil ekstrak dataset")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--landmarker", default="face_landmarker.task")
    ap.add_argument("--step", type=int, default=None, help="ambil 1 dari tiap N berkas (bawaan: biwi 5, lainnya 1)")
    args = ap.parse_args()
    step = args.step or (5 if args.dataset == "biwi" else 1)
    extract(args.dataset, ITERS[args.dataset](args.src, step), mediapipe_detector(args.landmarker), args.out)


if __name__ == "__main__":
    main()
