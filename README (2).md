# 🤟 Sistem Pengenalan Bahasa Isyarat (SIBI) Berbasis Random Forest

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)

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


