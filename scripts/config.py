"""Konfigurasi bersama untuk semua skrip (versi kontrol KEPALA). Ubah di sini, bukan di tiap skrip."""

# Urutan indeks keluaran model (0..4). Jangan diubah setelah training dimulai.
CLASSES = ["center", "left", "right", "up", "down"]

IMG_SIZE = 112   # sisi crop wajah yang masuk model (piksel, persegi)

# --- Ambang label dari sudut kepala (derajat) ---
# Konvensi akhir (setelah kalibrasi otomatis di 01b):
#   yaw   positif = pengguna menoleh ke KIRI-nya     (hidung bergeser ke kanan gambar, kamera tidak dicermin)
#   pitch positif = dagu naik / melihat ke ATAS
CENTER_MAX = 8.0    # |yaw| dan |pitch| <= nilai ini  -> center
SIDE_MIN = 15.0     # |yaw| >= nilai ini (dan |pitch| <= CROSS_MAX) -> left / right
VERT_MIN = 12.0     # |pitch| >= nilai ini (dan |yaw| <= CROSS_MAX) -> up / down
CROSS_MAX = 10.0    # sumbu lainnya harus di bawah ini; jika tidak (pose diagonal) gambar dibuang
MAX_ANGLE = 60.0    # sudut yang lebih besar dari ini dibuang (terlalu ekstrem untuk dipakai bermain)
