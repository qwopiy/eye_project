"""Memotong satu mata dari frame webcam menjadi gambar 96x96 yang meniru crop dataset Mendeley.

Dipakai SAMA PERSIS di runtime (05_runtime_udp.py). Jika Anda mengubah parameter di sini,
bandingkan hasilnya dengan contoh gambar dataset (lihat --save-crops di 05_runtime_udp.py).

Konvensi: frame webcam TIDAK dicerminkan. Dua sudut mata diberikan sebagai (sudut_kiri_gambar, sudut_kanan_gambar).
Indeks landmark MediaPipe Face Mesh (478 titik):
  mata KIRI pengguna (tampak di sisi KANAN frame): sudut dalam 362 (kiri gambar), sudut luar 263 (kanan gambar)
  mata KANAN pengguna (tampak di sisi KIRI frame): sudut luar 33 (kiri gambar), sudut dalam 133 (kanan gambar)
"""
import math

import cv2
import numpy as np

# (indeks sudut di sisi kiri gambar, indeks sudut di sisi kanan gambar)
EYE_CORNERS = {"left": (362, 263), "right": (33, 133)}


def eye_matrix(p_left, p_right, size=96, scale=1.9, shift=0.08):
    """Matriks affine 2x3 (sumber -> crop) yang memutar mata agar horizontal, lalu memusatkannya.

    scale : sisi crop = scale x lebar mata. Makin besar = mata makin kecil di crop (lebih banyak alis/kulit).
    shift : geser isi crop ke bawah sebesar shift x size, supaya alis ikut masuk di bagian atas.
    """
    (x1, y1), (x2, y2) = p_left, p_right
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = math.hypot(x2 - x1, y2 - y1)
    theta = math.atan2(y2 - y1, x2 - x1)          # sudut garis antar sudut mata
    side = max(width * scale, 1e-6)
    s = size / side
    c, sn = math.cos(theta), math.sin(theta)
    return np.array([
        [s * c, s * sn, size / 2.0 - s * (c * cx + sn * cy)],
        [-s * sn, s * c, size / 2.0 + shift * size - s * (-sn * cx + c * cy)],
    ], dtype=np.float64)


def eye_roi(frame_bgr, p_left, p_right, size=96, scale=1.9, shift=0.08):
    """Kembalikan crop BGR size x size. Tepi di luar frame diisi dengan replikasi piksel tepi."""
    m = eye_matrix(p_left, p_right, size, scale, shift)
    return cv2.warpAffine(frame_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _selftest():
    # 1) sudut kiri/kanan harus jatuh di tengah-vertikal, simetris di crop, walau kepala miring 20 derajat
    ang = math.radians(20)
    cx, cy, half = 300.0, 200.0, 40.0
    pl = (cx - half * math.cos(ang), cy - half * math.sin(ang))
    pr = (cx + half * math.cos(ang), cy + half * math.sin(ang))
    m = eye_matrix(pl, pr, 96, 1.9, 0.08)
    a = m @ np.array([pl[0], pl[1], 1.0])
    b = m @ np.array([pr[0], pr[1], 1.0])
    assert abs(a[1] - b[1]) < 1e-6, "garis mata harus horizontal di crop"
    assert abs(a[1] - (48 + 0.08 * 96)) < 1e-6, "mata harus di tengah (+shift)"
    assert abs((a[0] + b[0]) / 2 - 48) < 1e-6, "mata harus di tengah horizontal"
    assert 48 - a[0] > 0 and b[0] - 48 > 0 and a[0] < b[0], "sudut kiri harus di kiri crop"
    expect = 96 / (1.9 * 80.0) * 40.0  # setengah lebar mata (40 px sumber) dalam piksel crop
    assert abs((b[0] - a[0]) / 2 - expect) < 1e-6
    # 2) ukuran keluaran dan tidak error saat mata di tepi frame
    frame = np.zeros((480, 640, 3), np.uint8)
    out = eye_roi(frame, (2, 5), (40, 8))
    assert out.shape == (96, 96, 3)
    print("roi selftest: LULUS")


if __name__ == "__main__":
    _selftest()
