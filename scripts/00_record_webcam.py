"""Langkah 4: rekam data webcam SENDIRI untuk uji (dan boleh untuk melatih). Hasil: data/prep/real/<kelas>/*.jpg

Contoh (tiap anggota tim sekali, pakai nama masing-masing):
  python scripts/00_record_webcam.py --name andi

Cara kerja: untuk tiap kelas muncul instruksi. Siapkan posisi, tekan SPASI, hitung mundur 3 detik, lalu skrip
menyimpan crop wajah selama --seconds detik. Lakukan gerakan yang BENAR-BENAR akan Anda pakai saat bermain
(kecil, layar tetap terlihat) dan tahan sambil sedikit bergoyang. Tombol: SPASI = mulai, s = lewati kelas, q = selesai.
Tampilan jendela dicermin supaya terasa seperti cermin; berkas yang disimpan TIDAK dicermin (sama dengan runtime).
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

from config import CLASSES, IMG_SIZE
from face_crop import face_crop, make_landmarker

PROMPT = {
    "center": "LURUS: kepala tegak, lihat layar",
    "left": "KIRI: toleh kepala ke KIRI-mu (sekitar 15-25 derajat, layar masih terlihat)",
    "right": "KANAN: toleh kepala ke KANAN-mu (sekitar 15-25 derajat)",
    "up": "ATAS: dagu naik sedikit (mendongak)",
    "down": "BAWAH: dagu turun sedikit (menunduk)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nama/inisial pemain, dipakai pada nama berkas")
    ap.add_argument("--landmarker", default="face_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--every", type=int, default=3, help="simpan 1 dari tiap N frame")
    ap.add_argument("--out", default="data/prep/real")
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    args = ap.parse_args()

    landmarker = make_landmarker(args.landmarker, video=True)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} tidak bisa dibuka")

    last_ts = -1

    def grab():
        nonlocal last_ts
        ok, frame = cap.read()
        if not ok:
            return None, None
        ts = int(time.monotonic() * 1000)
        ts = ts if ts > last_ts else last_ts + 1
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        crop = face_crop(frame, res.face_landmarks[0], IMG_SIZE) if res.face_landmarks else None
        return frame, crop

    def show(frame, crop, line1, line2=""):
        view = cv2.flip(frame, 1)                       # hanya untuk tampilan
        if crop is not None:
            view[0:IMG_SIZE, 0:IMG_SIZE] = crop
        else:
            cv2.putText(view, "WAJAH TIDAK TERDETEKSI", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(view, line1, (8, view.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, line2, (8, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("rekam data (SPASI=mulai, s=lewati, q=selesai)", view)
        return cv2.waitKey(1) & 0xFF

    try:
        for c in args.classes:
            d = Path(args.out) / c
            d.mkdir(parents=True, exist_ok=True)
            skip = False
            while True:                                  # tunggu SPASI
                frame, crop = grab()
                if frame is None:
                    continue
                k = show(frame, crop, PROMPT[c], "tekan SPASI untuk mulai")
                if k == ord(" "):
                    break
                if k == ord("s"):
                    skip = True
                    break
                if k == ord("q"):
                    return
            if skip:
                continue
            t0 = time.time()
            while time.time() - t0 < 3.0:                # hitung mundur
                frame, crop = grab()
                if frame is not None:
                    show(frame, crop, PROMPT[c], f"mulai dalam {3 - (time.time() - t0):.0f}")
            saved, n, t0 = 0, 0, time.time()
            while time.time() - t0 < args.seconds:
                frame, crop = grab()
                if frame is None:
                    continue
                n += 1
                if crop is not None and n % args.every == 0:
                    cv2.imwrite(str(d / f"{args.name}_{int(time.time() * 1000)}.jpg"), crop)
                    saved += 1
                show(frame, crop, PROMPT[c], f"MEREKAM {args.seconds - (time.time() - t0):.0f}s  tersimpan: {saved}")
            print(f"{c}: {saved} gambar tersimpan di {d}")
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
