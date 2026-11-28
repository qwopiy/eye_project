"""Crop wajah dari landmark MediaPipe. DIPAKAI SAMA PERSIS saat menyiapkan data dan saat runtime game.

Beda dengan roi.py (mata): wajah TIDAK diputar/diluruskan, karena kemiringan kepala (roll) dan
posisi hidung/dagu justru informasi yang ingin dikenali model.
"""
import cv2
import numpy as np

FACE_SIZE = 112   # sisi crop keluaran (piksel). Samakan dengan IMG_SIZE di config.py
MARGIN = 1.25     # sisi crop = MARGIN x sisi terbesar kotak landmark (ada ruang untuk dahi/dagu/telinga)


def landmarks_box(landmarks, w, h):
    """Kotak pembatas (x0, y0, x1, y1) dalam piksel dari 478 landmark ber-koordinat ternormalisasi."""
    xs = np.array([p.x for p in landmarks], dtype=np.float64) * w
    ys = np.array([p.y for p in landmarks], dtype=np.float64) * h
    return xs.min(), ys.min(), xs.max(), ys.max()


def face_crop(frame_bgr, landmarks, size=FACE_SIZE, margin=MARGIN):
    """Potong persegi di sekitar wajah lalu ubah ukuran ke size x size. Tepi gambar diperpanjang (replicate).
    Mengembalikan None jika kotak wajah tidak masuk akal."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = landmarks_box(landmarks, w, h)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(round(max(bw, bh) * margin))
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    pad = side
    if left < -pad or top < -pad or left + side > w + pad or top + side > h + pad:
        return None
    padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[top + pad: top + pad + side, left + pad: left + pad + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def make_landmarker(model_path, video=False, num_faces=1):
    """Face Landmarker. video=False untuk gambar dataset, video=True untuk webcam (runtime)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def crop_from_image_file(landmarker, path, size=FACE_SIZE, margin=MARGIN):
    """Dipakai di skrip persiapan data: baca file -> deteksi -> crop. Mengembalikan crop BGR atau None."""
    import mediapipe as mp
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    return face_crop(bgr, res.face_landmarks[0], size, margin)


def _selftest():
    from types import SimpleNamespace as P
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :] = (10, 20, 30)
    # wajah sintetis: landmark menyebar di kotak x 200..300, y 100..220 (piksel) pada gambar 640x480
    rng = np.random.default_rng(0)
    pts = [P(x=(200 + 100 * a) / 640, y=(100 + 120 * b) / 480) for a, b in rng.random((478, 2))]
    pts[0] = P(x=200 / 640, y=100 / 480); pts[1] = P(x=300 / 640, y=220 / 480)
    c = face_crop(img, pts)
    assert c.shape == (FACE_SIZE, FACE_SIZE, 3), c.shape
    # wajah di pojok kiri-atas: harus tetap menghasilkan crop (tepi diperpanjang), bukan error
    pts2 = [P(x=p.x * 0.2, y=p.y * 0.1) for p in pts]
    pts2[0] = P(x=0.0, y=0.0); pts2[1] = P(x=0.1, y=0.12)
    c2 = face_crop(img, pts2)
    assert c2 is not None and c2.shape == (FACE_SIZE, FACE_SIZE, 3)
    # kotak sangat kecil -> None
    assert face_crop(img, [P(x=.5, y=.5)] * 10) is None
    # isi benar: tandai satu blok warna di tengah wajah, harus muncul di tengah crop
    img3 = img.copy(); img3[150:170, 240:260] = (255, 255, 255)
    c3 = face_crop(img3, pts)
    m = FACE_SIZE // 2
    assert c3[m - 3: m + 3, m - 3: m + 3].mean() > 200, c3[m - 3: m + 3, m - 3: m + 3].mean()
    print("face_crop selftest: LULUS")


if __name__ == "__main__":
    _selftest()
