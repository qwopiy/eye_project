"""Langkah 3: export model ke ONNX + uji paritas (hasil ONNX harus sama dengan model asli).

Contoh:
  python scripts/03_export_onnx.py torch --run runs/mobilenetv3s_s1 --out models/mobilenetv3s_v1_96.onnx
  python scripts/03_export_onnx.py torch --run runs/efficientnet_b0_s1 --out models/efficientnet_b0_v1_96.onnx
  python scripts/03_export_onnx.py yolo --weights runs/yolov8n_cls_s1/weights/best.pt --out models/yolov8n_v1_96.onnx

Kontrak ONNX: input RGB 0-1 bentuk [1,3,96,96]; output probabilitas [1,jumlah_kelas].
Urutan kelas ditulis di <nama>.json di samping file .onnx (dibaca oleh evaluasi dan runtime).
"""
import argparse
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import list_split
from config import IMG_SIZE


def sample_images(data_dir, n=20):
    items = list_split(data_dir, "val")
    if not items:
        raise SystemExit("Tidak ada gambar val untuk uji paritas.")
    step = max(1, len(items) // n)
    return [p for p, _ in items[::step][:n]]


def load_rgb01(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.transpose(a, (2, 0, 1))[None]


def check(sess, ref_fn, paths):
    name = sess.get_inputs()[0].name
    worst = 0.0
    for p in paths:
        x = load_rgb01(p)
        got = sess.run(None, {name: x})[0].reshape(-1)
        ref = ref_fn(p, x).reshape(-1)
        worst = max(worst, float(np.abs(got - ref).max()))
    print(f"uji paritas: selisih maksimum probabilitas = {worst:.2e}  ->  {'LULUS' if worst < 1e-3 else 'GAGAL'}")
    return worst < 1e-3


def export_torch(args):
    import torch
    import torch.nn as nn

    tt = importlib.import_module("02_train_torch")
    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu")
    model = tt.build(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state"])
    model.eval()

    class WithSoftmax(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    net = WithSoftmax(model).eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["input"], output_names=["probs"], opset_version=17)
    try:
        torch.onnx.export(net, dummy, str(out), dynamo=False, **kw)
    except TypeError:  # torch lama tidak punya argumen dynamo
        torch.onnx.export(net, dummy, str(out), **kw)

    def ref_fn(p, x):
        with torch.no_grad():
            return net(torch.from_numpy(x)).numpy()

    ok = check(ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]), ref_fn, sample_images(args.data))
    json.dump({"classes": ck["classes"], "arch": ck["arch"], "input_size": IMG_SIZE},
              open(out.with_suffix(".json"), "w"), indent=1)
    return ok


def export_yolo(args):
    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.exists():
        # Ultralytics versi baru bisa menyimpan di runs/classify/...; cari berdasarkan nama run
        cands = sorted(Path("runs").rglob(f"{w.parent.parent.name}/weights/{w.name}"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"Bobot tidak ditemukan: {w}\nCari manual: dir /s /b runs\\best.pt (Windows) atau find runs -name best.pt")
        w = cands[-1]
        print("Bobot ditemukan di:", w)
    m = YOLO(str(w))
    path = m.export(format="onnx", imgsz=IMG_SIZE, opset=17, simplify=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, out)
    classes = [m.names[i] for i in range(len(m.names))]  # urutan alfabet dari Ultralytics
    print("urutan kelas YOLO:", classes)

    def ref_fn(p, x):
        r = m.predict(str(p), imgsz=IMG_SIZE, verbose=False)[0]
        return r.probs.data.cpu().numpy()

    ok = check(ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]), ref_fn, sample_images(args.data))
    json.dump({"classes": classes, "arch": "yolov8n-cls", "input_size": IMG_SIZE},
              open(out.with_suffix(".json"), "w"), indent=1)
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("torch")
    a.add_argument("--run", required=True, help="folder run, mis. runs/mobilenetv3s_s1")
    a.add_argument("--out", required=True)
    a.add_argument("--data", default="data/prep")
    b = sub.add_parser("yolo")
    b.add_argument("--weights", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--data", default="data/prep")
    args = ap.parse_args()
    ok = export_torch(args) if args.kind == "torch" else export_yolo(args)
    print("ONNX disimpan:", args.out, "| siap dipakai" if ok else "| JANGAN dipakai sebelum paritas lulus")


if __name__ == "__main__":
    main()
