"""Konfigurasi bersama untuk semua skrip. Ubah di sini, bukan di tiap skrip."""

# Urutan indeks keluaran model (0..4). Jangan diubah setelah training dimulai.
CLASSES = ["center", "left", "right", "up", "down"]

# Nama folder di dataset Mendeley -> nama kelas kita.
# SESUAIKAN kunci di sebelah kiri dengan nama folder hasil unduhan Anda.
FOLDER_TO_CLASS = {
    "straight": "center",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
}

IMG_SIZE = 96          # ukuran gambar masuk model (piksel, persegi)
EYE = "left"           # mata yang dipakai: "left" (mata kiri pengguna) atau "right"
