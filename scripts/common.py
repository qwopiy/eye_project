"""Fungsi bantu: daftar berkas per split, metrik (tanpa sklearn), tabel hasil."""
from pathlib import Path

import numpy as np

from config import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_split(data_dir, split, classes=CLASSES):
    """Kembalikan daftar (path, indeks_kelas) untuk satu split. Folder yang tidak ada dilewati."""
    items = []
    for ci, name in enumerate(classes):
        d = Path(data_dir) / split / name
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, ci))
    return items


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf_from_cm(cm):
    """Precision, recall, F1 per kelas dari confusion matrix (baris = benar, kolom = prediksi)."""
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return prec, rec, f1


def macro_f1(y_true, y_pred, n):
    """Macro-F1 hanya atas kelas yang punya sampel di y_true."""
    cm = confusion_matrix(y_true, y_pred, n)
    _, _, f1 = prf_from_cm(cm)
    present = cm.sum(1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def format_cm(cm, names):
    w = max(len(n) for n in names) + 2
    head = " " * w + "".join(f"{n:>{w}}" for n in names)
    rows = [head]
    for i, n in enumerate(names):
        rows.append(f"{n:<{w}}" + "".join(f"{v:>{w}d}" for v in cm[i]))
    return "\n".join(rows)


def to_probs(v):
    """Pastikan keluaran model berupa probabilitas (jumlah = 1). Jika belum, terapkan softmax."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.min() >= 0 and abs(v.sum() - 1.0) < 1e-3:
        return v
    e = np.exp(v - v.max())
    return e / e.sum()
