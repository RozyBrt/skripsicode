# Import library
import json
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

from utils.preprocessing import (
    load_file,
    load_lexicon,
    preprocessing_teks,
    preprocessing_stopwords,
)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────

PATH_CRAWLING    = 'code_filter_crawling/crawling.csv'
PATH_PREPROCESS  = 'preprocessing/preprocessing.csv'
PATH_KLASTER     = 'klaster'
PATH_KLASIFIKASI = 'klasifikasi'
PATH_MODEL       = 'model'

# key  = nama file (underscore, aman di semua OS)
# value = label tampilan (human-readable)
KLASTER_CONFIG: dict[str, str] = {
    'kompensasi':     'Kompensasi',
    'kepuasan_kerja': 'Kepuasan Kerja',
    'aktualisasi':    'Aktualisasi',
    'hubungan_kerja': 'Hubungan Kerja',
}
LABEL_KLASTER = list(KLASTER_CONFIG.keys())


# ─────────────────────────────────────────────
# Resource loader (di-cache agar tidak reload setiap interaksi)
# ─────────────────────────────────────────────

@st.cache_resource
def load_resources():
    """Muat semua resource berat sekali saja saat pertama kali dipanggil."""
    slang_dict      = json.load(open("txt/kamusSlang.json", "r", encoding="utf-8"))
    stopwords       = load_file('txt/stopwords-1.txt')
    stopwords2      = load_file('txt/stopwords-2.txt')
    kamus_indonesia = load_file('txt/kamusIndonesia.txt')
    pos_lexicon     = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon     = load_lexicon('leksikon/leksikon-neg.json')
    factory         = StemmerFactory()
    stemmer         = factory.create_stemmer()
    return slang_dict, stopwords, stopwords2, kamus_indonesia, pos_lexicon, neg_lexicon, stemmer


@st.cache_resource
def load_sentiment_model():
    """Muat model sentimen dan vectorizer sekali saja."""
    model      = joblib.load(f'{PATH_MODEL}/model_sentimen.pkl')
    vectorizer = joblib.load(f'{PATH_MODEL}/vectorizer_sentimen.pkl')
    return model, vectorizer


# ─────────────────────────────────────────────
# Halaman Preprocessing
# ─────────────────────────────────────────────

def HPreprocessing():
    st.title("Halaman Preprocessing")

    slang_dict, stopwords, _, kamus_indonesia, _, _, stemmer = load_resources()

    # Tombol unduh dataset mentah
    st.write("Unduh Dataset Mentah")
    try:
        with open(PATH_CRAWLING, "r", encoding="utf-8") as f:
            csv_raw_data = f.read()
        st.download_button(
            label="⬇️ Unduh Dataset",
            data=csv_raw_data,
            file_name="crawling.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except FileNotFoundError:
        st.warning("⚠️ File dataset crawling tidak ditemukan.")

    st.divider()

    # Upload dataset baru
    uploaded_file = st.file_uploader("📂 Upload File Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data_baru = pd.read_csv(uploaded_file)

        # ── Validasi kolom ──────────────────────────────
        if 'full_text' not in data_baru.columns:
            st.error(
                "❌ CSV harus memiliki kolom bernama **`full_text`**. "
                f"Kolom yang ditemukan: `{list(data_baru.columns)}`"
            )
            return
        # ────────────────────────────────────────────────

        data_baru = data_baru.rename(columns={"full_text": "teks"})
        st.write(f"**Data sebelum preprocessing ({len(data_baru):,} baris):**")
        st.dataframe(data_baru[['teks']], use_container_width=True)

        if st.button("🧹 Bersihkan", use_container_width=True):
            with st.spinner("Sedang memproses... Harap tunggu."):
                data_baru['teks'] = data_baru['teks'].apply(
                    lambda x: preprocessing_teks(
                        str(x), slang_dict, stopwords, kamus_indonesia, stemmer
                    )
                )
                data_baru.to_csv(PATH_PREPROCESS, index=False)
            st.success(f"✅ Preprocessing selesai! {len(data_baru):,} baris tersimpan.")

    st.divider()

    if st.button("📋 Tampilkan Hasil Preprocessing", use_container_width=True):
        try:
            hasil = pd.read_csv(PATH_PREPROCESS)
            st.write(f"**Hasil preprocessing ({len(hasil):,} baris):**")
            st.dataframe(hasil, use_container_width=True)
        except FileNotFoundError:
            st.error("⚠️ File belum tersedia. Jalankan proses 'Bersihkan' terlebih dahulu.")


# ─────────────────────────────────────────────
# Halaman Klasterisasi
# ─────────────────────────────────────────────

def HClustering():
    st.title("Halaman Klasterisasi")

    _, _, stopwords2, _, _, _, _ = load_resources()

    if st.button("🔵 Mulai Klasterisasi", use_container_width=True):
        try:
            df = pd.read_csv(PATH_PREPROCESS)
        except FileNotFoundError:
            st.error("⚠️ File preprocessing tidak ditemukan. Jalankan halaman Preprocessing terlebih dahulu.")
            return

        with st.spinner("Sedang melakukan klasterisasi K-Means..."):
            df['teks'] = df['teks'].fillna('').astype(str)
            df['teks'] = df['teks'].apply(lambda x: preprocessing_stopwords(x, stopwords2))

            num_rows = len(df)

            # Centroid manual per faktor
            centroid_sentences = {
                'kompensasi':     "gaji kompensasi",
                'kepuasan_kerja': "mental stres jam",
                'aktualisasi':    "berkembang kembang jabatan skill",
                'hubungan_kerja': "hubungan jahat hubungan baik lingkung",
            }

            additional_data = pd.DataFrame({'teks': list(centroid_sentences.values())})
            df_with_centroids = pd.concat([df, additional_data], ignore_index=True)

            centroid_indices = list(range(num_rows, num_rows + len(centroid_sentences)))
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df_with_centroids['teks'])
            lokasi_centroid = X[centroid_indices].toarray()

            kmeans = KMeans(n_clusters=4, init=lokasi_centroid, n_init=10, random_state=0)
            kmeans.fit(X)

            df_with_centroids['label_klaster'] = kmeans.labels_

            # ── Simpan model KMeans & vectorizer ────────────
            joblib.dump(kmeans,     f'{PATH_MODEL}/kmeans_model.pkl')
            joblib.dump(vectorizer, f'{PATH_MODEL}/vectorizer_kmeans.pkl')
            # ────────────────────────────────────────────────

        db_score = davies_bouldin_score(X.toarray(), kmeans.labels_)
        col1, col2 = st.columns(2)
        col1.metric("Total Data (+ centroid)", f"{len(df_with_centroids):,}")
        col2.metric("Davies-Bouldin Score", f"{db_score:.4f}")

        # Hapus baris centroid buatan sebelum simpan
        df_result = df_with_centroids.iloc[:-4].reset_index(drop=True)

        st.write("**Preview 10 data terakhir:**")
        st.dataframe(df_result.tail(10), use_container_width=True)

        # Pisah per klaster, simpan, dan tampilkan
        clusters = [
            df_result[df_result['label_klaster'] == i][['teks', 'label_klaster']].reset_index(drop=True)
            for i in range(4)
        ]

        for (key, display_name), cluster_data in zip(KLASTER_CONFIG.items(), clusters):
            st.subheader(f"Faktor {display_name} ({len(cluster_data):,} data)")
            st.dataframe(cluster_data, use_container_width=True)
            cluster_data.to_csv(f'{PATH_KLASTER}/{key}.csv', sep='\t', index=False)

        st.success("✅ Klasterisasi selesai! Model KMeans & vectorizer telah disimpan ke folder `model/`.")


# ─────────────────────────────────────────────
# Halaman Sentiment Analysis
# ─────────────────────────────────────────────

def HSentimentAnalysis():
    st.title("Halaman Klasifikasi Sentimen")

    if st.button("🔍 Analisis Sentimen", use_container_width=True):
        try:
            model, vectorizer = load_sentiment_model()
        except FileNotFoundError:
            st.error("⚠️ Model tidak ditemukan. Pastikan file `.pkl` ada di folder `model/`.")
            return

        with st.spinner("Sedang menganalisis sentimen..."):
            for key, display_name in KLASTER_CONFIG.items():
                try:
                    df = pd.read_csv(f'{PATH_KLASTER}/{key}.csv', sep='\t')
                    df['teks'] = df['teks'].fillna('').astype(str)

                    X = vectorizer.transform(df['teks'])
                    df['label_sentimen'] = model.predict(X)

                    df.to_csv(f'{PATH_KLASIFIKASI}/{key}.csv', index=False, sep='\t')

                    st.subheader(f"Faktor {display_name}")
                    dist = df['label_sentimen'].value_counts().to_dict()
                    st.caption(" | ".join([f"**{k}**: {v}" for k, v in dist.items()]))
                    st.dataframe(df, use_container_width=True)

                except FileNotFoundError:
                    st.warning(f"⚠️ File klaster '{display_name}' tidak ditemukan. Pastikan klasterisasi sudah dijalankan.")

        st.success("✅ Analisis sentimen selesai dan hasil disimpan!")


# ─────────────────────────────────────────────
# Halaman Visualisasi Data
# ─────────────────────────────────────────────

def HDataVisualization():
    st.title("Halaman Visualisasi Data")

    if st.button("📊 Tampilkan Visualisasi", use_container_width=True):
        # Muat semua dataframe klasifikasi
        dataframes: dict[str, pd.DataFrame] = {}
        for key, display_name in KLASTER_CONFIG.items():
            try:
                dataframes[key] = pd.read_csv(f'{PATH_KLASIFIKASI}/{key}.csv', sep='\t')
            except FileNotFoundError:
                st.warning(f"⚠️ File klasifikasi '{display_name}' tidak ditemukan.")

        if not dataframes:
            st.error("Tidak ada data untuk divisualisasikan. Jalankan Analisis Sentimen terlebih dahulu.")
            return

        # ── Bar chart: distribusi data per klaster ──
        st.subheader("Distribusi Data Pada Faktor-Faktor yang Mempengaruhi Perpindahan Karir")
        label_ada    = [KLASTER_CONFIG[k] for k in dataframes]
        jumlah_data  = [len(dataframes[k]) for k in dataframes]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(label_ada, jumlah_data, color='#4A90D9', edgecolor='white', linewidth=0.8)
        ax.set_title("Jumlah Data untuk Setiap Faktor", fontsize=16)
        ax.set_xlabel("Faktor-faktor", fontsize=13)
        ax.set_ylabel("Jumlah Data", fontsize=13)
        ax.bar_label(bars, padding=3)
        ax.set_ylim(0, max(jumlah_data) * 1.15)
        fig.tight_layout()
        st.pyplot(fig)

        st.divider()

        # ── Pie chart: distribusi sentimen per klaster ──
        for key, display_name in KLASTER_CONFIG.items():
            if key not in dataframes or dataframes[key].empty:
                continue

            df = dataframes[key]
            jumlah_sentimen = df['label_sentimen'].value_counts()
            num_sentimen    = len(jumlah_sentimen)
            explode         = tuple([0.03] * num_sentimen)

            st.subheader(f"Faktor {display_name}")
            st.write(f"Total data: **{len(df):,} ulasan**")

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#90EE90', '#32CD32', '#228B22'][:num_sentimen]
            ax.pie(
                jumlah_sentimen,
                labels=jumlah_sentimen.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                pctdistance=0.75,
                explode=explode,
            )
            ax.axis('equal')
            fig.tight_layout()
            st.pyplot(fig)

            for sentiment, count in jumlah_sentimen.items():
                st.write(f"- **{sentiment}**: {count:,} ulasan ({count/len(df)*100:.1f}%)")

            st.divider()


# ─────────────────────────────────────────────
# Navigasi
# ─────────────────────────────────────────────

page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Preprocessing", "Clustering", "Sentiment Analysis", "Data Visualization"]
)

if page == "Preprocessing":
    HPreprocessing()
elif page == "Clustering":
    HClustering()
elif page == "Sentiment Analysis":
    HSentimentAnalysis()
elif page == "Data Visualization":
    HDataVisualization()