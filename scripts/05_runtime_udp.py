"""Langkah 5: jalankan di laptop saat bermain. Webcam -> MediaPipe -> crop satu mata -> model ONNX -> UDP -> Unity.

Persiapan sekali:
  pip install mediapipe opencv-python onnxruntime numpy
  unduh model landmark wajah (lihat panduan) ke file face_landmarker.task

Jalankan:
  python scripts/05_runtime_udp.py --onnx models/mobilenetv3s_v1_96.onnx --landmarker face_landmarker.task --show

Tombol di jendela --show:  q = keluar,  s = simpan crop mata saat ini ke folder crops/
Pesan UDP (JSON, ke 127.0.0.1:5005, satu per frame):
  {"v":1,"frame_id":N,"t_capture_ms":epoch_ms,"face":true,"label":"left","conf":0.91,
   "probs":[...],"classes":[...],"model":"...","infer_ms":4.2}
"""
import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import to_probs
from config import EYE, IMG_SIZE
from roi import EYE_CORNERS, eye_roi


class EyeClassifier:
    """Membungkus sesi ONNX: crop BGR 96x96 -> (label, conf, probs, infer_ms)."""

    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        meta = json.load(open(Path(onnx_path).with_suffix(".json")))
        self.classes = meta["classes"]
        self.name = Path(onnx_path).stem

    def predict(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None]
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.in_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        probs = to_probs(out)
        k = int(np.argmax(probs))
        return self.classes[k], float(probs[k]), probs, ms


def make_message(frame_id, t_capture_ms, face, clf, label=None, conf=0.0, probs=None, infer_ms=0.0):
    return {
        "v": 1,
        "frame_id": frame_id,
        "t_capture_ms": t_capture_ms,
        "face": bool(face),
        "label": label if face else "none",
        "conf": round(float(conf), 4),
        "probs": [round(float(p), 4) for p in probs] if probs is not None else [],
        "classes": clf.classes,
        "model": clf.name,
        "infer_ms": round(float(infer_ms), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--landmarker", default="face_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--eye", default=EYE, choices=["left", "right"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--scale", type=float, default=1.9, help="sisi crop = scale x lebar mata")
    ap.add_argument("--shift", type=float, default=0.08, help="geser isi crop ke bawah (agar alis ikut)")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela debug")
    ap.add_argument("--save-crops", default=None, help="folder untuk menyimpan crop tiap 15 frame (untuk kalibrasi)")
    args = ap.parse_args()

    clf = EyeClassifier(args.onnx)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.landmarker),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} tidak bisa dibuka")
    if args.save_crops:
        Path(args.save_crops).mkdir(parents=True, exist_ok=True)

    ia, ib = EYE_CORNERS[args.eye]
    frame_id, last_ts, fps_t, fps_n, fps = 0, -1, time.time(), 0, 0.0
    print(f"Mengirim ke udp://{args.host}:{args.port} | model={clf.name} | mata={args.eye} | kelas={clf.classes}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            t_capture_ms = time.time() * 1000.0
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(time.monotonic() * 1000)
            ts = ts if ts > last_ts else last_ts + 1  # MediaPipe butuh timestamp naik terus
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            crop = None
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                pl = (lm[ia].x * w, lm[ia].y * h)
                pr = (lm[ib].x * w, lm[ib].y * h)
                crop = eye_roi(frame, pl, pr, IMG_SIZE, args.scale, args.shift)
                label, conf, probs, ms = clf.predict(crop)
                msg = make_message(frame_id, t_capture_ms, True, clf, label, conf, probs, ms)
            else:
                msg = make_message(frame_id, t_capture_ms, False, clf)
            sock.sendto(json.dumps(msg).encode("utf-8"), dest)
            frame_id += 1

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

            if args.save_crops and crop is not None and frame_id % 15 == 0:
                cv2.imwrite(str(Path(args.save_crops) / f"crop_{frame_id:06d}_{msg['label']}.jpg"), crop)

            if args.show:
                view = frame.copy()
                if crop is not None:
                    cv2.circle(view, (int(pl[0]), int(pl[1])), 3, (0, 255, 0), -1)
                    cv2.circle(view, (int(pr[0]), int(pr[1])), 3, (0, 255, 0), -1)
                    view[0:192, 0:192] = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
                txt = f"{msg['label']} {msg['conf']:.2f}  {fps:.0f} fps  {msg['infer_ms']:.1f} ms"
                cv2.putText(view, txt, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("eye runtime (q=keluar, s=simpan crop)", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s") and crop is not None:
                    Path("crops").mkdir(exist_ok=True)
                    cv2.imwrite(f"crops/crop_{frame_id:06d}.jpg", crop)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
