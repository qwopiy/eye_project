"""Langkah 2b: latih YOLOv8n-cls (Ultralytics).

Contoh:
  python scripts/02_train_yolo.py --seed 1

Keluaran: runs/yolov8n_cls_s<seed>/weights/best.pt

Catatan: urutan kelas di YOLO = urutan alfabet nama folder (down, left, right, up, center -> center, down, ...).
Karena itu skrip 03_export_onnx.py menulis urutan yang BENAR dari model.names ke file .json.
Augmentasi bawaan Ultralytics dipakai; flip horizontal dan random erasing dimatikan.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from config import IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prep", help="folder berisi train/ val/ test/")
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None, help="mis. 0 untuk GPU pertama, cpu untuk CPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        patience=8,
        project=str(Path("runs").resolve()),  # jalur absolut: kalau relatif, Ultralytics menyelipkan runs/classify/
        name=f"yolov8n_cls_s{args.seed}",
        exist_ok=True,
        fliplr=0.0,    # jangan flip: kiri/kanan akan tertukar
        flipud=0.0,
        erasing=0.0,   # jangan hapus patch acak: bisa menutup pupil
        scale=0.2,     # crop acak lebih lembut (80-100% area)
        device=args.device,
    )
    print("Bobot tersimpan di:", Path("runs") / f"yolov8n_cls_s{args.seed}" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
