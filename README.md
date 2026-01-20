```markdown
# 🌊 Water Segmentation Project

Aplikasi berbasis web untuk mendeteksi dan melakukan segmentasi area perairan (water bodies) pada citra satelit menggunakan metode Deep Learning (**U-Net**) dan **Streamlit**.

Projek ini dibangun menggunakan Python dan TensorFlow sebagai tugas Data Mining / Pengolahan Citra Digital.

## 📋 Fitur Utama
* **Upload Citra:** Pengguna dapat mengunggah citra satelit.
* **Prediksi Real-time:** Model U-Net memprediksi area air secara otomatis.
* **Visualisasi:** Menampilkan hasil *masking* (area air vs daratan) berdampingan dengan citra asli.

## 🛠️ Teknologi yang Digunakan
* **Python 3.11**
* **TensorFlow / Keras** (Deep Learning Framework)
* **Streamlit** (Web Interface)
* **OpenCV & NumPy** (Image Processing)

---

## 🚀 Panduan Instalasi (Cara Menjalankan)

Karena keterbatasan ukuran file di GitHub, file **Model (.h5)** dan **Dataset** tidak disertakan dalam repository ini. Silakan ikuti langkah-langkah berikut untuk menjalankan aplikasi:

### 1. Clone Repository
Salin repository ini ke komputer lokal Anda:
```bash
git clone [https://github.com/Tsaqif691/Datamining_Projek_Segmentasi.git](https://github.com/Tsaqif691/Datamining_Projek_Segmentasi.git)
cd Datamining_Projek_Segmentasi

```

### 2. Buat Virtual Environment

Sangat disarankan menggunakan virtual environment agar library tidak bentrok dengan sistem utama laptop Anda.

**Untuk Windows:**

```bash
# Buat environment (pastikan menggunakan Python 3.10 atau 3.11)
py -3.11 -m venv venv

# Aktifkan environment
.\venv\Scripts\Activate.ps1

```

**Untuk Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Library

Install semua dependensi yang diperlukan yang tercatat di file requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. ⚠️ Unduh File Penting (Wajib)

Karena batasan ukuran file di GitHub, Anda harus mengunduh file Model dan Dataset secara manual, lalu meletakkannya di folder utama projek ini.

Silakan unduh file **`water_unet_model.h5`** (Model AI yang sudah dilatih) melalui tautan ini: **https://drive.google.com/drive/folders/1emA4AuaRec5HO5sQqoiIhZhf4CmmegFZ?usp=sharing**.

Selain itu, jika Anda membutuhkan data untuk pelatihan ulang, Anda dapat mengunduh **`dataset.zip`** (Opsional) melalui tautan ini: **https://drive.google.com/drive/folders/1LxlluA-PpQHYm-4Az_tme6WFIkIxMRo5?usp=sharing**.

> **PENTING:** Setelah diunduh, pastikan kedua file tersebut diletakkan **tepat di folder utama** (sejajar dengan file `main.py`), jangan dimasukkan ke dalam sub-folder lain.

### 5. Jalankan Aplikasi

Setelah environment aktif dan file model sudah ada di folder yang benar, jalankan perintah berikut:

```bash
streamlit run main.py

```

Browser akan otomatis terbuka di alamat `http://localhost:8501`.

---

## 📂 Struktur Folder

Pastikan susunan folder Anda terlihat seperti ini agar program berjalan lancar:

```
## 📂 Struktur Folder
Pastikan susunan folder Anda terlihat seperti ini:

```text
PROJECT-SEGMENTASI-PCD/
├── dataset/
│   └── Water Bodies Dataset/
│       ├── Images/
│       └── Masks/
├── models/
│   └── water_unet_model.h5           # [MANUAL] File Model (Pastikan ada di folder ini)
├── pages/
│   ├── Segmentasi.py
│   ├── Segmentasi_backup.py
│   ├── Tentang.py
│   └── Tutorial.py
├── venv/                             # Virtual Environment
├── .gitignore
├── create_model_fix.py
├── create_unet_model.py
├── main.py                           # File Utama
├── README.md
├── requirements.txt
├── satellite-images-of-water-bodies.zip # [MANUAL] File Dataset Zip
└── train_water_segmentation.py

```

## 🤝 Kontribusi

Projek ini dikembangkan oleh **Water Body Segmentation Tim**.
Silakan lakukan *Pull Request* jika ingin berkontribusi atau melaporkan *Issue* jika menemukan bug.

---

© 2026 Water Body Segmentation Project.

```

```