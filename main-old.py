import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
import json
import os

# Pengaturan untuk halaman web streamlit
st.set_page_config(page_title="revisi-code")

# Streamlit application
st.title("ANALISIS SENTIMEN PADA FAKTOR-FAKTOR YANG MEMPENGARUHI PERPINDAHAN KARIR DENGAN MENGGUNAKAN METODE ASPECT-BASED SENTIMENT ANALYSIS DAN K-MEANS")
page = st.sidebar.selectbox("Tentukan Halaman:", ["Preprosesing", "Clustering", "Sentiment Analysis", "Data Visualization"])

# MENENTUKAN LOKASI FILE HASIL SCRAPING
file_path = "csv/crawling.csv"  # Lokasi file CSV data awal
with open(file_path, "r", encoding="utf-8") as f:
    csv_raw_data = f.read()

# MUAT SUMBERDAYA
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(json.load(file))

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())

slang_dict = json.load(open("txt/kamusSlang.json", "r", encoding="utf-8"))
stopwords = load_file('txt/stopwords.txt')
kamus_indonesia = load_file('txt/kamusIndonesia.txt')

# INISIALISASI STEMMER
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(text, slang_dict, stopwords, kamus_indonesia, stemmer):
    text = text.lower()  # Case folding
    text = re.sub(r"\\t|\\n|\\u|\\|http[s]?://\\S+|[@#][A-Za-z0-9_]+", " ", text)  # Menghapus karakter khusus
    text = re.sub(r"\\d+", "", text)  # Menghapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Menghapus tanda baca (pakai import string)
    text = re.sub(r"\\s+", ' ', text).strip()  # merapihkan spasi ganda
    text = re.sub(r"\b[a-zA-Z]\b", "", text) # Menghapus satu huruf (besar/kecil)
    text = ' '.join([slang_dict.get(word, word) for word in text.split()]) # Normalisasi (pemanfaatan kamus slang)
    text = word_tokenize(text) # Tokenisasi (sebelum stemming)
    text = [stemmer.stem(word) for word in text] # Stemming
    text = [word for word in text if word not in stopwords and len(word) > 3 and word in kamus_indonesia] # Stopwords & memilah kata
    text = ' '.join(text)
    return text

# FUNGSI UNTUK MENJALANKAN SELURUH FUNGSI PREPROCESSING
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Memeriksa apakah kolom 'full_text' ada
    if 'full_text' not in df.columns:
        st.error("File CSV tidak memiliki kolom 'full_text'. Pastikan format file benar.")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika kolom tidak ada

    df['filtered'] = df['full_text'].apply(lambda x: preprocessing(x, slang_dict, stopwords, kamus_indonesia, stemmer))

    # Hapus baris yang memiliki nilai kosong (termasuk yang berisi spasi atau karakter non-huruf)
    df = df[df['filtered'].str.strip().astype(bool)]
    df.to_csv('preprosesing/hasil.txt', index=None, header=True)
    return df

# Logika aplikasi
if page == "Preprosesing":
    st.header("Persiapan Data")
    
    # Tombol Download Data Mentah
    st.write("Unduh dataset untuk melanjutkan proses analisis di bawah")
    st.download_button(
        label="Download File CSV",
        data=csv_raw_data,
        file_name="dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type='csv')

    # Memeriksa apakah file ada dalam session_state
    if 'uploaded_file' in st.session_state:
        st.write("File yang sedang diproses ...")
        st.write(st.session_state.uploaded_file.name)  # Menampilkan nama file yang diupload

    if uploaded_file is not None:
        # Menyimpan file ke dalam session_state untuk akses di semua halaman tanpa upload ulang
        st.session_state.uploaded_file = uploaded_file

        # Memastikan nama file hanya disimpan satu kali
        if 'uploaded_file_name' not in st.session_state:
            # Menyimpan nama file untuk digunakan kembali nantinya.
            st.session_state.uploaded_file_name = uploaded_file.name

    # Tampilkan tombol "Bersihkan" jika ada file yang di-upload
    if 'uploaded_file' in st.session_state:
        # Cek apakah preprocessing sudah dilakukan
        if 'preprocessed' not in st.session_state or not st.session_state.preprocessed:
            # Jika tombol "Bersihkan" ditekan, lakukan preprocessing
            if st.button("Bersihkan"):
                df_preprocessed = preprocess_data(st.session_state.uploaded_file)
                
                # Simpan hasil preprocessing di session_state
                if 'preprocessed_data' not in st.session_state:
                    st.session_state.preprocessed_data = []  # Inisialisasi list jika belum ada
                # Simpan data yang diproses dan nama file-nya
                st.session_state.preprocessed_data.append({
                    'data': df_preprocessed,
                    'filename': st.session_state.uploaded_file.name
                })  
                st.success("Preprocessing Selesai.")
        else:
            # Warning jika file sudah diproses sebelumnya
            st.warning("File ini sudah di upload")

    # Menampilkan semua data yang telah diproses
    if 'preprocessed_data' in st.session_state:
        for item in st.session_state.preprocessed_data:
            df = item['data']  # Ambil DataFrame yang telah diproses
            filename = item['filename']  # Ambil nama file
            st.write(f"Hasil Preprosesing dari file: {filename}:")  # Tampilkan nama file
            st.dataframe(df)  # Tampilkan DataFrame

elif page == "Clustering":
    st.header("Analisis Faktor")  # Menampilkan judul halaman

    # Load data hasil preprocessing dari file
    def load_preprocessed_data():
        file_path = 'preprosesing/hasil.txt'  # Ganti dengan path yang sesuai
        return pd.read_csv(file_path)

    df_selected = load_preprocessed_data()

    # Mendefinisikan kalimat untuk setiap centroid
    centroid_sentences = {
        'kompensasi': "kompensasi uang pendapatan dapat penghasilan hasil intensif gaji sedikit bonus",
        'kepuasan_kerja': "kepuasan puas bahagia sedih nyaman lembur jam kerja waktu cape capek lelah stres stress suntuk pening pusing",
        'aktualisasi': "pengembangan kembang potensi kreatif prestasi jabatan jabat gelar industri karir ahli ilmu bakat capai",
        'hubungan_kerja': "hubungan hubung rekan teman kawan kolaborasi tempat suasana dukungan dukung toksik toxic jahat benci toleransi takut buruk"
    }

    # Menghitung posisi dalam DataFrame untuk setiap centroid
    num_rows = len(df_selected)
    centroid_positions = {
        int(num_rows * 0.25): centroid_sentences['kompensasi'],
        int(num_rows * 0.50): centroid_sentences['kepuasan_kerja'],
        int(num_rows * 0.75): centroid_sentences['aktualisasi'],
        int(num_rows * 0.90): centroid_sentences['hubungan_kerja']
    }

    # Menyisipkan kalimat ke dalam DataFrame pada posisi yang ditentukan
    for pos, sentence in centroid_positions.items():
        df_selected.at[pos, 'filtered'] = sentence

    # Memastikan semua entri di 'filtered' adalah string untuk proses TF-IDF
    df_selected['filtered'] = df_selected['filtered'].apply(lambda x: str(x))

    texts = df_selected['filtered'].astype(str)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Mengambil centroid awal berdasarkan posisi kalimat yang disisipkan
    initial_centroids = X[list(centroid_positions.keys())].toarray()

    if st.button("Klaster"):
        kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=10, random_state=0)
        kmeans.fit(X)
        df_selected['cluster'] = kmeans.labels_
        # Menghitung Silhouette Score
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")

        # Menghitung DBI
        db_score = davies_bouldin_score(X.toarray(), kmeans.labels_)
        st.write(f"**Davies-Bouldin Score:** {db_score:.2f}")

        # Memisahkan klaster menjadi DataFrame yang berbeda
        cluster_0 = df_selected[df_selected['cluster'] == 0][['filtered']].reset_index(drop=True)
        cluster_1 = df_selected[df_selected['cluster'] == 1][['filtered']].reset_index(drop=True)
        cluster_2 = df_selected[df_selected['cluster'] == 2][['filtered']].reset_index(drop=True)
        cluster_3 = df_selected[df_selected['cluster'] == 3][['filtered']].reset_index(drop=True)
        
        # Mendefinisikan label kustom untuk setiap klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']
    
        # Daftar kalimat yang ingin dihapus
        kalimat_yang_ingin_dihapus = [ 
            "kompensasi uang pendapatan dapat penghasilan hasil intensif gaji sedikit bonus",
            "kepuasan puas bahagia sedih nyaman lembur jam kerja waktu cape capek lelah stres stress suntuk pening pusing",
            "pengembangan kembang potensi kreatif prestasi jabatan jabat gelar industri karir ahli ilmu bakat capai",
            "hubungan hubung rekan teman kawan kolaborasi tempat suasana dukungan dukung toksik toxic jahat benci toleransi takut buruk"
        ]

        # Menghapus tanda baca (Ubah tokenisasi kalimat semula)
        def hapus_tandabaca(text):
            if isinstance(text, str):
                text = re.sub(r'[^\w\s]', '', text)
            elif isinstance(text, list):
                # Jika teks berupa/didalam list
                text = [re.sub(r'[^\w\s]', '', word) for word in text]
            return text

        # Menghapus kata (value)
        def hapus_kalimat(text):
            if isinstance(text, str):
                # Menghapus kalimat yang ada dalam daftar kalimat_yang_ingin_dihapus
                for kalimat in kalimat_yang_ingin_dihapus:
                    text = text.replace(kalimat, '')
            elif isinstance(text, list):
                # Jika teks berupa/didalam list
                text = [word for word in text if word not in kalimat_yang_ingin_dihapus]
            return text

        # Pemanggilan fungsi penghapusan dan menghapus nilai null
        cleaned_data_0 = cluster_0.applymap(hapus_tandabaca).applymap(hapus_kalimat)
        cleaned_data_0 = cleaned_data_0.replace('', np.nan).dropna()

        cleaned_data_1 = cluster_1.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_1 = cleaned_data_1.replace('', np.nan).dropna()

        cleaned_data_2 = cluster_2.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_2 = cleaned_data_2.replace('', np.nan).dropna()

        cleaned_data_3 = cluster_3.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_3 = cleaned_data_3.replace('', np.nan).dropna()

        #m Menampilkan 
        for i, (label, dataframe_klaster) in enumerate(zip(label_klaster, [cleaned_data_0, cleaned_data_1, cleaned_data_2, cleaned_data_3])):
            st.write(f"### Faktor {label.capitalize()}")
            st.dataframe(dataframe_klaster[['filtered']])

        
        # Menyimpan data yang telah dibersihkan ke file
        cleaned_data_0.to_csv('klaster/kompensasi.txt', sep='\t', index=False, header=True)
        cleaned_data_1.to_csv('klaster/kepuasan kerja.txt', sep='\t', index=False, header=True)
        cleaned_data_2.to_csv('klaster/aktualisasi.txt', sep='\t', index=False, header=True)
        cleaned_data_3.to_csv('klaster/hubungan kerja.txt', sep='\t', index=False, header=True)

elif page == "Sentiment Analysis":
    st.header("Analisis Sentimen Faktor")

    # Load leksikon positif dan negatif
    pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

    # Tombol untuk memulai analisis sentimen
    if st.button("Lakukan proses analisis sentimen"):

        # Load data kluster dari folder 'klaster'
        def load_cluster_data(cluster_name):
            file_path = f'klaster/{cluster_name}.txt'
            if os.path.exists(file_path):
                return pd.read_csv(file_path, sep='\t')
            else:
                st.error(f"File {cluster_name}.txt tidak ditemukan di folder 'klaster'.")
                return pd.DataFrame()  # Kembalikan DataFrame kosong jika file tidak ditemukan

        # Fungsi untuk menghitung sentimen berdasarkan leksikon
        def hitung_sentimen_berdasarkan_leksikon(text):
            if not isinstance(text, str):
                text = ""
            pos_count = sum(1 for word in text.split() if word in pos_lexicon)
            neg_count = sum(1 for word in text.split() if word in neg_lexicon)
            if pos_count > neg_count:
                return 'Positif', 1  # Sentimen positif dan skor
            elif neg_count > pos_count:
                return 'Negatif', -1  # Sentimen negatif dan skor
            else:
                return 'Netral', 0  # Sentimen netral dan skor

        # Label untuk setiap kluster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Menyimpan hasil analisis sentimen dari setiap kluster
        kumpulan_sentimen = {}

        # Memuat dan memproses setiap kluster
        for label in label_klaster:
            dataframe_klaster = load_cluster_data(label)  # Ambil DataFrame dari file .txt
            if not dataframe_klaster.empty:
                # Analisis sentimen dan tambahkan label dan skor
                dataframe_klaster[['sentiment_label', 'sentiment_score']] = dataframe_klaster['filtered'].apply(
                    lambda x: pd.Series(hitung_sentimen_berdasarkan_leksikon(x))
                )
                
                # Simpan DataFrame untuk kluster saat ini
                kumpulan_sentimen[label] = dataframe_klaster

                # Tampilkan hasil analisis sentimen untuk kluster saat ini
                st.write(f"### Analisis Sentimen Faktor {label.capitalize()}")
                st.write(dataframe_klaster[['filtered', 'sentiment_label', 'sentiment_score']])

                # Simpan hasil analisis ke file
                output_file_path = f'analisis/{label}.txt'
                dataframe_klaster[['filtered', 'sentiment_label', 'sentiment_score']].to_csv(
                    output_file_path, sep='\t', index=False, header=['analisis', 'sentiment_label', 'sentiment_score']
                )

elif page == "Data Visualization":
    st.header("Visualisasi Data")
    if st.button("Visualisasikan"):
        # Load hasil analisis sentimen dari file
        def memuat_data_sentimen(cluster_name):
            return pd.read_csv(f'analisis/{cluster_name}.txt', sep='\t')

        # Label klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Load dan visualisasikan data untuk setiap klaster
        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label)  # Ambil DataFrame dari file
            if not dataframe_klaster.empty:
                jumlah_sentimen = dataframe_klaster['sentiment_label'].value_counts()

                # Buat Pie Chart
                st.subheader(f"Visualisasi Analisis Sentimen Faktor {label.capitalize()}")
                st.write(f"Total data pada faktor {label.capitalize()} sebanyak : {len(dataframe_klaster)}")
                fig, ax = plt.subplots()
                colors = ['#ADD8E6', '#87CEFA', '#4682B4']
                ax.pie(jumlah_sentimen, labels=jumlah_sentimen.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Membuat pie chart berbentuk lingkaran.
                st.pyplot(fig)

                # Deskripsi singkat hasil analisis
                st.write(f"Faktor {label.capitalize()}, analisis sentimen menunjukkan distribusi sebagai berikut:")
                for sentiment, count in jumlah_sentimen.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")

