# 🤟 Sistem Pengenalan Bahasa Isyarat (SIBI) Berbasis Random Forest

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)

<div align="center">
  <img src="https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=SIBI+Sign+Language+Recognition" alt="SIBI Recognition Banner">
</div>

## 📋 Daftar Isi
- [Ringkasan](#-ringkasan)
- [Latar Belakang](#-latar-belakang)
- [Fitur Utama](#-fitur-utama)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Hasil & Performa](#-hasil--performa)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dokumentasi](#-dokumentasi)
- [Tim Pengembang](#-tim-pengembang)
- [Referensi](#-referensi)
- [Lisensi](#-lisensi)

## 🎯 Ringkasan

Sistem Pengenalan Bahasa Isyarat Indonesia (SIBI) adalah aplikasi berbasis **Random Forest** yang mampu mengenali gerakan tangan untuk alfabet A-Z, angka 0-9, dan spasi dalam bahasa isyarat secara **real-time**. Sistem ini dikembangkan untuk membantu komunikasi antara penyandang tunarungu dengan masyarakat umum.

Proyek ini merupakan Tugas UTS Pengolahan Citra Digital dari **Program Studi S1 Sains Data, Universitas Negeri Surabaya**, Tahun Akademik 2024/2025 Semester Genap.

### ⚠️ Mengapa Proyek Ini Penting?

Menurut **World Health Organization (WHO)**, pada tahun 2050 diperkirakan **2,5 miliar orang** akan mengalami gangguan pendengaran, dengan lebih dari **700 juta orang** membutuhkan layanan rehabilitasi pendengaran. Sistem penerjemah bahasa isyarat otomatis seperti ini sangat penting untuk menciptakan lingkungan yang lebih inklusif bagi penyandang disabilitas.

## 🔍 Latar Belakang

Komunikasi adalah kebutuhan dasar manusia sebagai makhluk sosial. Namun, penyandang tunarungu menghadapi kesenjangan komunikasi yang signifikan karena:

- ❌ Tidak semua orang memahami bahasa isyarat
- ❌ Keterbatasan akses dalam pendidikan dan pekerjaan
- ❌ Minimnya sistem penerjemah otomatis yang akurat

### Apa itu SIBI?

**SIBI (Sistem Isyarat Bahasa Indonesia)** adalah bahasa isyarat resmi yang diakui pemerintah Indonesia, biasa digunakan dalam acara formal dan Sekolah Luar Biasa (SLB). SIBI mengadaptasi American Sign Language (ASL) dan menggunakan satu tangan untuk membentuk simbol dengan makna tertentu.

## ✨ Fitur Utama

- 🎥 **Deteksi Real-Time**: Mengenali gesture tangan langsung melalui webcam
- 🤖 **Random Forest Classifier**: Algoritma ensemble learning untuk akurasi tinggi
- 🎯 **99,46% Akurasi**: Hasil klasifikasi yang sangat akurat
- 📊 **37 Kelas Gesture**: Alfabet A-Z (26), Angka 0-9 (10), dan Spasi (1)
- 👋 **MediaPipe Integration**: Deteksi 21 landmark tangan untuk ekstraksi fitur
- 💻 **OpenCV Real-time**: Implementasi langsung dengan kamera
- 📦 **Lightweight Model**: Efisien dan cepat dalam prediksi

## 📊 Dataset

Dataset diperoleh dari **Kaggle** dengan judul "Sign Language Alphabet Dataset".

### Spesifikasi Dataset

| Karakteristik | Detail |
|---------------|--------|
| **Total Gambar** | 55,500 gambar |
| **Jumlah Kelas** | 37 kelas (A-Z, 0-9, Spasi) |
| **Gambar per Kelas** | 1,500 gambar |
| **Resolusi** | 50×50 piksel (diubah menjadi 250×250) |
| **Format** | RGB dan Binary (threshold) |

### Komposisi Dataset

```
📁 Dataset
├── 📂 Folder 1: Gambar RGB (50×50 px)
│   ├── A (1,500 gambar)
│   ├── B (1,500 gambar)
│   ├── ...
│   ├── Z (1,500 gambar)
│   ├── 0-9 (1,500 gambar masing-masing)
│   └── _ /spasi (1,500 gambar)
│
└── 📂 Folder 2: Gambar Binary (hasil preprocessing)
    └── (Struktur sama dengan Folder 1)
```

<div align="center">

| Contoh Gesture Tangan |
|:---------------------:|
| <img src="https://via.placeholder.com/100x100/FF5722/FFFFFF?text=A" width="80"> <img src="https://via.placeholder.com/100x100/3F51B5/FFFFFF?text=B" width="80"> <img src="https://via.placeholder.com/100x100/4CAF50/FFFFFF?text=C" width="80"> <img src="https://via.placeholder.com/100x100/FFC107/000000?text=5" width="80"> |

</div>

## 🔬 Metodologi

### Alur Kerja Sistem

```
📥 Dataset → 🔄 Pre-processing → 🎯 Ekstraksi Fitur (MediaPipe)
                                           ↓
    ✅ Model Terbaik ← 📊 Evaluasi ← 🌲 Training (Random Forest)
           ↓
    🎥 Implementasi Real-Time (OpenCV + Webcam)
```

### Tahapan Detail

#### 1️⃣ Pre-processing

**Tujuan**: Menyiapkan gambar untuk ekstraksi fitur yang optimal

- **Resize**: 50×50 → 250×250 piksel (agar MediaPipe dapat mendeteksi landmark)
- **Grayscale**: Konversi RGB → grayscale untuk menghilangkan noise warna
- **Thresholding (Otsu)**: Menghasilkan citra biner (hitam-putih) untuk memisahkan objek tangan dari latar belakang

<div align="center">

| RGB | Grayscale | Binary (Otsu) |
|:---:|:---------:|:-------------:|
| <img src="https://via.placeholder.com/150x150/FF5722/FFFFFF?text=RGB" width="150"> | <img src="https://via.placeholder.com/150x150/9E9E9E/FFFFFF?text=Gray" width="150"> | <img src="https://via.placeholder.com/150x150/000000/FFFFFF?text=Binary" width="150"> |

</div>

#### 2️⃣ Ekstraksi Fitur dengan MediaPipe Hands

MediaPipe mendeteksi **21 titik landmark** pada tangan:

- Ujung jari (fingertips)
- Sendi-sendi jari (finger joints)
- Pergelangan tangan (wrist)

Setiap landmark memiliki koordinat **(x, y)** → Total **42 fitur** per gambar (21 × 2).

<div align="center">
  <img src="https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=21+Hand+Landmarks" alt="MediaPipe Landmarks">
</div>

**Keunggulan MediaPipe:**
- ✅ Stabil terhadap variasi pencahayaan
- ✅ Robust terhadap perbedaan latar belakang
- ✅ Informasi posisi jari yang detail

#### 3️⃣ Pembagian Dataset

| Set | Persentase | Fungsi |
|-----|------------|--------|
| **Training** | 80% | Melatih model |
| **Testing** | 20% | Menguji performa model |

**Catatan**: Dataset dibagi secara acak (random split) tanpa stratifikasi.

#### 4️⃣ Training Model

##### a. Decision Tree (Baseline)

Decision Tree membentuk aturan berbasis fitur untuk mengklasifikasikan gesture. Namun, model tunggal cenderung overfitting.

##### b. Random Forest (Model Utama) 🌲🌲🌲

**Cara Kerja:**
1. **Bootstrap Sampling**: Membuat banyak subset data secara acak
2. **Multiple Trees**: Membangun banyak decision tree dari masing-masing subset
3. **Random Feature Selection**: Setiap tree hanya menggunakan subset fitur acak
4. **Voting Mayoritas**: Prediksi akhir berdasarkan hasil voting dari semua tree

**Parameter Optimal:**
- `n_estimators`: Jumlah pohon dalam forest
- `max_depth`: Kedalaman maksimum setiap pohon
- `criterion`: Gini Index untuk pemisahan node

**Keunggulan:**
- ✅ Mengurangi overfitting
- ✅ Performa baik pada dataset besar
- ✅ Mendukung klasifikasi multikelas (37 kelas)

#### 5️⃣ Evaluasi Model

Metrik yang digunakan:
- **Akurasi**: Persentase prediksi benar
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan mendeteksi kelas positif
- **F1-Score**: Harmonic mean dari precision dan recall

## 📈 Hasil & Performa

### 🏆 Akurasi Model: **99,46%**

```bash
99.45799457994579% of samples were classified correctly !
```

Model Random Forest berhasil mengklasifikasikan gesture tangan dengan akurasi **hampir sempurna**, menunjukkan bahwa kombinasi ekstraksi fitur MediaPipe dengan Random Forest sangat efektif.

### Perbandingan Model

| Model | Akurasi | Kecepatan | Kompleksitas |
|-------|---------|-----------|--------------|
| **Random Forest** 🥇 | **99,46%** | Cepat | Sedang |
| Decision Tree | ~85-90% | Sangat Cepat | Rendah |

### Analisis Hasil

**Kekuatan:**
- ✅ Akurasi sangat tinggi (99,46%)
- ✅ Generalisasi baik pada data testing
- ✅ Minim kesalahan klasifikasi
- ✅ Stabil terhadap variasi gesture

**Potensi Peningkatan:**
- 🔄 Implementasi stratified split untuk distribusi kelas seimbang
- 🔄 Hyperparameter tuning lebih lanjut
- 🔄 Augmentasi data untuk variasi lebih banyak

## 🚀 Instalasi

### Prasyarat

```bash
Python 3.8 atau lebih tinggi
pip (Python package manager)
Webcam (untuk implementasi real-time)
```

### Langkah Instalasi

1. **Clone Repository**
```bash
git clone https://github.com/username/sibi-sign-language-recognition.git
cd sibi-sign-language-recognition
```

2. **Buat Virtual Environment (Opsional tapi Disarankan)**
```bash
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies yang Diperlukan

```txt
opencv-python>=4.5.0
mediapipe>=0.9.0
scikit-learn>=1.0.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
pickle5>=0.0.11
```

## 💻 Cara Penggunaan

### 1. Pre-processing Dataset

```python
python preprocessing.py --input_path data/raw --output_path data/processed
```

### 2. Ekstraksi Fitur

```python
python feature_extraction.py --input_path data/processed --output_path data/features.pkl
```

### 3. Training Model

```python
python train_model.py --features data/features.pkl --model_output models/rf_model.pkl
```

**Opsi Training:**
- `--n_estimators`: Jumlah pohon (default: 100)
- `--max_depth`: Kedalaman maksimum (default: None)
- `--test_size`: Proporsi data testing (default: 0.2)

### 4. Evaluasi Model

```python
python evaluate_model.py --model models/rf_model.pkl --test_data data/features.pkl
```

Output:
```
Accuracy: 99.46%
Classification Report:
              precision    recall  f1-score   support
...
```

### 5. Real-Time Detection 🎥

```python
python real_time_detection.py --model models/rf_model.pkl --camera 0
```

**Cara Penggunaan:**
1. Jalankan script
2. Webcam akan aktif
3. Tunjukkan gesture tangan di depan kamera
4. Prediksi akan muncul secara real-time di layar
5. Tekan `q` untuk keluar

<div align="center">
  <img src="https://via.placeholder.com/600x400/2196F3/FFFFFF?text=Real-Time+Detection+Demo" alt="Real-time Demo">
</div>

## 📚 Dokumentasi

### Struktur Project

```
sibi-sign-language-recognition/
│
├── data/
│   ├── raw/                    # Dataset asli
│   ├── processed/              # Gambar hasil preprocessing
│   └── features.pkl            # Fitur hasil ekstraksi
│
├── models/
│   ├── rf_model.pkl           # Model Random Forest terlatih
│   └── dt_model.pkl           # Model Decision Tree (optional)
│
├── src/
│   ├── preprocessing.py        # Script pre-processing
│   ├── feature_extraction.py  # Ekstraksi fitur MediaPipe
│   ├── train_model.py         # Training Random Forest
│   ├── evaluate_model.py      # Evaluasi model
│   └── real_time_detection.py # Deteksi real-time
│
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   └── Model_Comparison.ipynb # Perbandingan model
│
├── docs/
│   ├── laporan_uts.pdf        # Laporan lengkap
│   └── presentasi.pdf         # Slide presentasi
│
├── requirements.txt            # Python dependencies
├── README.md                   # Dokumentasi ini
└── LICENSE                     # Lisensi project
```

### Contoh Kode: Feature Extraction

```python
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
        return landmarks
    return None
```

### Contoh Kode: Training Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['features']
y = data['labels']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    criterion='gini',
    random_state=42
)

rf_model.fit(X_train, y_train)

# Save model
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
```

## 👥 Tim Pengembang

**Kelas 2023B - Program Studi S1 Sains Data**  
**Fakultas Matematika dan Ilmu Pengetahuan Alam**  
**Universitas Negeri Surabaya**

| Nama | NIM | Role |
|------|-----|------|
| Tanti Ayu Hardiningtyas | 23031554002 | Project Lead & ML Engineer |
| Intan Ayu Lestari | 23031554051 | Data Scientist & Researcher |
| Dimas Fatkhul Rahman | 23031554211 | Developer & System Integrator |

## 📝 Kesimpulan

Sistem pengenalan bahasa isyarat SIBI berbasis Random Forest ini berhasil mencapai:

✅ **Akurasi 99,46%** dalam mengklasifikasikan gesture tangan  
✅ **Real-time detection** menggunakan webcam dengan OpenCV  
✅ **37 kelas gesture** (A-Z, 0-9, spasi) dapat dikenali dengan akurat  
✅ **Ekstraksi fitur robust** menggunakan MediaPipe Hands  
✅ **Model efisien** dengan Random Forest Classifier  

**Dampak Sosial:**
- 🤝 Memfasilitasi komunikasi penyandang tunarungu dengan masyarakat umum
- 🎓 Dapat digunakan sebagai alat bantu pembelajaran SIBI
- 🌍 Mendukung terciptanya lingkungan inklusif untuk penyandang disabilitas

## 🔮 Pengembangan Selanjutnya

- [ ] Menambahkan gesture dinamis (gerakan kata)
- [ ] Integrasi dengan aplikasi mobile
- [ ] Dataset lebih beragam dengan variasi pencahayaan
- [ ] Implementasi deep learning (CNN) untuk perbandingan
- [ ] Sistem text-to-speech untuk output suara
- [ ] Deployment ke edge devices (Raspberry Pi)

## 📚 Referensi

1. Erissa, D., & Widinarsih, D. (2022). Akses Penyandang Disabilitas Terhadap Pekerjaan. *Jurnal Pembangunan Manusia*, 3(1). [Link](https://scholarhub.ui.ac.id/jpm/vol3/iss1/28)

2. Nofiturrahmah, F. (2018). Problematika Anak Tunarungu Dan Cara Mengatasinya. *Quality*, 6(1), 1-15. [DOI](http://dx.doi.org/10.21043/quality.v6i1.5744)

3. WHO. (2025). Deafness and Hearing Loss. [Link](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss)

4. Das, S., et al. (2023). A hybrid approach for Bangla sign language recognition using deep transfer learning model with random forest classifier. *Expert Systems with Applications*, 213(B), 118914. [DOI](https://doi.org/10.1016/j.eswa.2022.118914)

5. Wiraswendro, P. E., & Soetanto, H. (2022). Penerapan Algoritma Random Forest Classifier pada Sistem Deteksi Simbol SIBI. *Jurnal BIT*, 19(2), 75-81. [Link](https://journal.budiluhur.ac.id/index.php/bit/article/view/2043/)

6. Mulyana, D. I., et al. (2025). Optimasi Deteksi Gerak Bahasa Isyarat dan Ekspresi Wajah Real Time Dengan Metode Random Forest. *Jurnal JTIK*, 9(1), 277-284. [DOI](https://doi.org/10.35870/jtik.v9i1.3188)

7. Suyudi, I., et al. (2022). Pengenalan Bahasa Isyarat Indonesia menggunakan Mediapipe dengan Model Random Forest. *JISTED*, 1(1), 65-80. [DOI](https://doi.org/10.35912/jisted.v1i1.1899)

*... dan 6 referensi lainnya (lihat laporan lengkap)*

## 📄 Lisensi

Project ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- **Kaggle** untuk menyediakan dataset Sign Language Alphabet
- **Google MediaPipe** untuk library hand landmark detection
- **Universitas Negeri Surabaya** untuk dukungan akademik
- **Dosen Pengolahan Citra Digital** untuk bimbingan proyek
- **Komunitas Open Source** untuk tools dan libraries

---

<div align="center">

**⭐ Jika project ini bermanfaat, berikan star pada repository ini! ⭐**

**💬 Ada pertanyaan? Buka [Issues](https://github.com/username/sibi-recognition/issues)**

Dibuat dengan ❤️ oleh Mahasiswa Sains Data UNESA

[📧 Email](mailto:datascience@unesa.ac.id) · [🌐 Website](https://datascience.unesa.ac.id)

</div>
