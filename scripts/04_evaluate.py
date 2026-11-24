"""Langkah 4: evaluasi satu atau beberapa model ONNX pada set test dan bandingkan.

Contoh:
  python scripts/04_evaluate.py --onnx models/mobilenetv3s_v1_96.onnx models/efficientnet_b0_v1_96.onnx models/yolov8n_v1_96.onnx

Keluaran: tabel perbandingan di layar + results/<nama>.json (confusion matrix, metrik, latensi).
Latensi diukur di CPU, satu gambar per panggilan (cara yang sama dengan di game).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import accuracy, confusion_matrix, format_cm, list_split, macro_f1, prf_from_cm, to_probs
from config import CLASSES, IMG_SIZE


def load_x(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.transpose(a, (2, 0, 1))[None]


def evaluate(onnx_path, data_dir, split):
    meta = json.load(open(Path(onnx_path).with_suffix(".json")))
    model_classes = meta["classes"]  # urutan keluaran model
    # petakan indeks keluaran model -> indeks kelas proyek (CLASSES) berdasarkan NAMA
    remap = {i: CLASSES.index(n) for i, n in enumerate(model_classes)}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    items = list_split(data_dir, split)
    if not items:
        raise SystemExit(f"Tidak ada gambar di split '{split}'")

    for _ in range(20):  # pemanasan
        sess.run(None, {in_name: load_x(items[0][0])})

    y_true, y_pred, lat = [], [], []
    for p, y in items:
        x = load_x(p)
        t0 = time.perf_counter()
        out = sess.run(None, {in_name: x})[0]
        lat.append((time.perf_counter() - t0) * 1000)
        y_true.append(y)
        y_pred.append(remap[int(np.argmax(to_probs(out)))])

    cm = confusion_matrix(y_true, y_pred, len(CLASSES))
    prec, rec, f1 = prf_from_cm(cm)
    present = [i for i in range(len(CLASSES)) if cm[i].sum() > 0]
    return {
        "model": Path(onnx_path).stem,
        "n": len(items),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, len(CLASSES)),
        "per_class": {CLASSES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                   "n": int(cm[i].sum())} for i in present},
        "confusion": cm.tolist(),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "size_mb": Path(onnx_path).stat().st_size / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    rows = []
    for o in args.onnx:
        r = evaluate(o, args.data, args.split)
        json.dump(r, open(Path(args.out) / f"{r['model']}.json", "w"), indent=1)
        print(f"\n== {r['model']}  (n={r['n']})")
        print(format_cm(np.array(r["confusion"]), CLASSES))
        for c, v in r["per_class"].items():
            print(f"  {c:>7}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['n']})")
        rows.append(r)

    print("\n=== Perbandingan model ===")
    print(f"{'model':<26}{'acc':>7}{'macroF1':>9}{'med ms':>8}{'p95 ms':>8}{'MB':>7}")
    for r in rows:
        print(f"{r['model']:<26}{r['accuracy']:>7.3f}{r['macro_f1']:>9.3f}"
              f"{r['latency_ms_median']:>8.2f}{r['latency_ms_p95']:>8.2f}{r['size_mb']:>7.1f}")


if __name__ == "__main__":
    main()
