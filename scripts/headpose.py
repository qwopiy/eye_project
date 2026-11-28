"""Bantu untuk data kepala: proksi pose dari landmark, kalibrasi sudut dataset, pelabelan lima kelas,
dan pembaca anotasi dataset (HPID/Pointing'04, BIWI, AFLW2000-3D).

Mengapa ada "kalibrasi"? Tiap dataset memakai konvensi sudut berbeda (tanda, urutan sumbu). Daripada menebak,
kita cocokkan sudut dataset dengan PROKSI yang arahnya pasti: posisi ujung hidung terhadap garis kedua mata
pada gambar. Korelasi menentukan kolom mana yang yaw/pitch dan tandanya.
"""
import math
import re
from pathlib import Path

import numpy as np

from config import CENTER_MAX, CROSS_MAX, MAX_ANGLE, SIDE_MIN, VERT_MIN

# indeks landmark MediaPipe: ujung hidung dan sudut luar mata (kiri/kanan GAMBAR)
LM_NOSE, LM_EYE_IMG_LEFT, LM_EYE_IMG_RIGHT = 1, 33, 263


def pose_proxy(landmarks, w, h):
    """(px, py) = posisi ujung hidung relatif terhadap titik tengah mata, dalam kerangka yang searah garis mata
    (jadi tidak terpengaruh kemiringan kepala/roll) dan dinormalkan oleh jarak antar sudut mata.
      px > 0 : hidung bergeser ke KANAN gambar  -> pengguna menoleh ke kirinya
      py > 0 : hidung lebih jauh di bawah garis mata -> dagu turun
    """
    a = np.array([landmarks[LM_EYE_IMG_LEFT].x * w, landmarks[LM_EYE_IMG_LEFT].y * h])
    b = np.array([landmarks[LM_EYE_IMG_RIGHT].x * w, landmarks[LM_EYE_IMG_RIGHT].y * h])
    n = np.array([landmarks[LM_NOSE].x * w, landmarks[LM_NOSE].y * h])
    d = float(np.linalg.norm(b - a))
    if d < 1e-6:
        return None
    u = (b - a) / d
    v = np.array([-u[1], u[0]])
    m = (a + b) / 2.0
    return float((n - m) @ u / d), float((n - m) @ v / d)


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def calibrate(ds, px, py):
    """ds: array N x K (K=2 atau 3 kolom sudut dalam derajat, urutan/tanda belum diketahui).
    Mengembalikan dict: indeks kolom yaw & pitch, tanda masing-masing, dan korelasinya."""
    ds = np.asarray(ds, float)
    px, py = np.asarray(px, float), np.asarray(py, float)
    k = ds.shape[1]
    c = np.array([[_corr(ds[:, i], px), _corr(ds[:, i], py)] for i in range(k)])
    best = None
    for i in range(k):
        for j in range(k):
            if i != j:
                s = abs(c[i, 0]) + abs(c[j, 1])
                if best is None or s > best[0]:
                    best = (s, i, j)
    _, i, j = best
    return {
        "yaw_col": i, "yaw_sign": 1.0 if c[i, 0] >= 0 else -1.0, "yaw_corr": abs(float(c[i, 0])),
        # pitch positif = dagu naik = py turun
        "pitch_col": j, "pitch_sign": -1.0 if c[j, 1] >= 0 else 1.0, "pitch_corr": abs(float(c[j, 1])),
    }


def apply_calibration(ds, cal):
    ds = np.asarray(ds, float)
    return ds[:, cal["yaw_col"]] * cal["yaw_sign"], ds[:, cal["pitch_col"]] * cal["pitch_sign"]


def label_from_angles(yaw, pitch):
    """Lima kelas dari sudut (derajat). None = dibuang (zona abu-abu / diagonal / terlalu ekstrem)."""
    ay, ap = abs(yaw), abs(pitch)
    if ay > MAX_ANGLE or ap > MAX_ANGLE:
        return None
    if ay <= CENTER_MAX and ap <= CENTER_MAX:
        return "center"
    if ay >= SIDE_MIN and ap <= CROSS_MAX:
        return "left" if yaw > 0 else "right"
    if ap >= VERT_MIN and ay <= CROSS_MAX:
        return "up" if pitch > 0 else "down"
    return None


# ----------------------------- pembaca anotasi dataset -----------------------------
def rotmat_to_euler_deg(R):
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll). Mengembalikan (a0, a1, a2) = (yaw, pitch, roll) dalam derajat.
    Konvensi BIWI tidak kita ketahui pasti; tanda/urutannya diselesaikan oleh calibrate()."""
    R = np.asarray(R, float)
    pitch = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    return tuple(math.degrees(t) for t in (yaw, pitch, roll))


def read_biwi_pose(txt_path):
    """frame_XXXXX_pose.txt: 3 baris pertama = matriks rotasi 3x3, lalu baris kosong, lalu pusat kepala."""
    rows = []
    for line in Path(txt_path).read_text().splitlines():
        vals = [float(t) for t in line.split()]
        if len(vals) == 3:
            rows.append(vals)
        if len(rows) == 3:
            break
    if len(rows) < 3:
        raise ValueError(f"matriks rotasi tidak lengkap: {txt_path}")
    return rotmat_to_euler_deg(np.array(rows))


HPID_RE = re.compile(r"personne(\d{2})(\d)(\d{2})([+-]\d+)([+-]\d+)", re.IGNORECASE)


def parse_hpid_name(name):
    """personne01146-15+0.jpg -> (orang '01', seri '1', nomor '46', sudut1 -15, sudut2 0). None bila tak cocok."""
    m = HPID_RE.search(Path(name).stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), float(m.group(4)), float(m.group(5))


def read_aflw_pose(mat_path):
    """AFLW2000-3D: Pose_Para = [pitch, yaw, roll, tx, ty, tz, skala]; sudut dalam radian."""
    from scipy.io import loadmat
    p = np.asarray(loadmat(str(mat_path))["Pose_Para"], float).reshape(-1)
    return tuple(math.degrees(t) for t in p[:3])


# ----------------------------- uji mandiri -----------------------------
def _selftest():
    rng = np.random.default_rng(0)
    # 1) euler <-> matriks (round trip)
    def Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for _ in range(200):
        y, p, r = np.radians(rng.uniform(-60, 60)), np.radians(rng.uniform(-50, 50)), np.radians(rng.uniform(-30, 30))
        got = rotmat_to_euler_deg(Ry(y) @ Rx(p) @ Rz(r))
        assert np.allclose(got, np.degrees([y, p, r]), atol=1e-6), (got, (y, p, r))
    # 2) proksi: hidung ke kanan gambar -> px>0; invarian terhadap roll
    class P:
        def __init__(s, x, y): s.x, s.y = x, y
    lm = [P(0.5, 0.5) for _ in range(478)]
    lm[33], lm[263], lm[1] = P(0.4, 0.4), P(0.6, 0.4), P(0.55, 0.5)
    px, py = pose_proxy(lm, 100, 100)
    assert px > 0 and py > 0, (px, py)
    th = math.radians(25)  # putar seluruh titik 25 derajat di sekitar pusat gambar
    def rot(q): x, y = q.x - .5, q.y - .5; return P(.5 + x * math.cos(th) - y * math.sin(th), .5 + x * math.sin(th) + y * math.cos(th))
    lm2 = [rot(q) for q in lm]
    px2, py2 = pose_proxy(lm2, 100, 100)
    assert abs(px - px2) < 1e-9 and abs(py - py2) < 1e-9
    # 3) kalibrasi: kolom tertukar + tanda terbalik harus terdeteksi
    n = 400
    yaw = rng.uniform(-50, 50, n); pitch = rng.uniform(-40, 40, n); roll = rng.uniform(-15, 15, n)
    pxs = 0.012 * yaw + rng.normal(0, .02, n)
    pys = 0.30 - 0.010 * pitch + rng.normal(0, .02, n)
    ds = np.stack([pitch, -yaw, roll], 1)            # urutan: pitch, -yaw, roll
    cal = calibrate(ds, pxs, pys)
    assert (cal["yaw_col"], cal["yaw_sign"], cal["pitch_col"], cal["pitch_sign"]) == (1, -1.0, 0, 1.0), cal
    yy, pp = apply_calibration(ds, cal)
    assert np.allclose(yy, yaw) and np.allclose(pp, pitch)
    ds2 = np.stack([yaw, -pitch], 1)                  # K=2, pitch terbalik
    cal2 = calibrate(ds2, pxs, pys)
    yy, pp = apply_calibration(ds2, cal2)
    assert np.allclose(yy, yaw) and np.allclose(pp, pitch)
    # 4) pelabelan
    L = label_from_angles
    assert L(0, 0) == "center" and L(7, -7) == "center"
    assert L(20, 3) == "left" and L(-20, 3) == "right" and L(15, 10) == "left"
    assert L(2, 20) == "up" and L(2, -20) == "down"
    assert L(12, 4) is None and L(20, 20) is None and L(70, 0) is None and L(0, 61) is None
    # 5) nama HPID
    assert parse_hpid_name("personne01146-15+0.jpg") == ("01", "1", "46", -15.0, 0.0)
    assert parse_hpid_name("personne15100-90+0.jpg")[3:] == (-90.0, 0.0)
    assert parse_hpid_name("readme.txt") is None
    print("headpose selftest: LULUS")


if __name__ == "__main__":
    _selftest()
