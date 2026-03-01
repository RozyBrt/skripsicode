# 📊 Analisis Sentimen & Klasterisasi Faktor Perpindahan Kerja (SkripsiCode)

Proyek ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi keputusan karyawan dalam berpindah karir menggunakan teknik **Machine Learning**. Sistem ini mengelompokkan ulasan ke dalam faktor-faktor tertentu dan menilai sentimen di setiap faktor tersebut.

## 🚀 Akses Aplikasi
Cobalah aplikasi secara langsung di sini:  
👉 [**Buka Aplikasi Streamlit**](https://skripsicode.streamlit.app/)

![Demo Aplikasi](demo_app/viewgif.gif)

---

## 🛠️ Fitur Utama
- **Preprocessing Otomatis:** Pembersihan teks dari data mentah hingga siap diolah (Normalisasi slang, Stemming Sastrawi, Stopwords).
- **Klasterisasi K-Means:** Mengelompokkan ulasan ke dalam 4 faktor (Kompensasi, Kepuasan Kerja, Aktualisasi, Hubungan Kerja) menggunakan *manual centroid*.
- **Analisis Sentimen:** Klasifikasi otomatis menggunakan *Logistic Regression* untuk menentukan emosi ulasan (Positif, Negatif, Netral).
- **Visualisasi Dinamis:** Grafik Bar Chart dan Pie Chart untuk mempermudah pengambilan kesimpulan.

---

## 🧭 Alur Sistem (Flow)
1. **Model Training (`build.ipynb`):** Melatih model sentimen secara offline menggunakan dataset berlabel dan teknik SMOTE untuk menangani *imbalance data*.
2. **Data Ingestion:** User mengunggah data ulasan mentah hasil crawling di halaman **Preprocessing**.
3. **Pembersihan Data:** Sistem membersihkan teks (menghapus angka, simbol, normalisasi kata gaul, dan mengecilkan huruf).
4. **Klasterisasi:** Di halaman **Clustering**, sistem membagi ulasan ke dalam 4 faktor kerja utama menggunakan K-Means.
5. **Analisis Sentimen:** Di halaman **Sentiment Analysis**, sistem menilai perasaan pada setiap kelompok faktor menggunakan model yang sudah dilatih.
6. **Kesimpulan Grafik:** Di halaman **Data Visualization**, hasil akhir ditampilkan dalam bentuk grafik untuk analisis skripsi.

---

## 📁 Struktur Folder
- `model/`: Menyimpan model `.pkl` (Sentimen & K-Means).
- `utils/`: Logika fungsi pembersihan teks (`preprocessing.py`).
- `txt/`: Kamus data (Stopwords, Kamus Slang, Kamus Indonesia).
- `leksikon/`: Kamus sentimen untuk labeling otomatis.
- `preprocessing/`, `klaster/`, `klasifikasi/`: Folder penyimpanan hasil setiap tahapan.

---

## 💻 Cara Menjalankan Lokal
1. **Instalasi Library:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Pelatihan Model (Opsional):** Jalankan seluruh baris pada file `build.ipynb` untuk melatih ulang model sentimen.
3. **Jalankan Aplikasi:**
   ```bash
   streamlit run main.py
   ```
4. **Evaluasi:** Jalankan `confusion_kmeans.ipynb` untuk melihat matriks evaluasi model.
