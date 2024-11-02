import streamlit as st
import pandas as pd
import json
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

import nltk
nltk.download('punkt_tab')

# Pengaturan untuk halaman web steamlit
st.set_page_config(page_title="ABSA-KMeans", page_icon="☕")

# Streamlit application
st.title("ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI PERPINDAHAN KARIR DENGAN PEMANFAATAN ASPECT-BASED SENTIMENT ANALYSIS MENGGUNAKAN METODE K-MEANS")
page = st.sidebar.selectbox("Select a page:", ["Preprocessing", "Clustering", "Sentiment Analysis", "Data Visualization"])


# KUMPULAN FUNGSI PREPROSESING

# Fungsi untuk mengubah teks menjadi huruf kecil (lowercase)
def case_folding(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].str.lower()
    return dataframe

# Fungsi untuk menghapus karakte-karakter spesial twitter(X) dari data hasil scrapping
def remove_tweet_special(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

# Fungsi untuk menghapus angka untuk penyederhanaan data clear
def remove_number(text):
    return re.sub(r"\d+", "", text)

# Fungsi untuk menghapus tanda baca seperti titik, koma, tanda seru, dll.
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# Fungsi untuk menghapus whitespace atau spasi yang ada di awal dan akhir teks.
def remove_whitespace_LT(text):
    return text.strip()

# Fungsi untuk mengganti spasi berturut turut dengan satu spasi tunggal.
def remove_whitespace_multiple(text):
    return re.sub(r'\s+', ' ', text)

# Fungsi untuk menghapus karakter tunggal yang berdiri sendiri di dalam teks
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# Fungsi untuk membaca file JSON yang berisi kamus slang (bahasa gaul) dan menyimpannya ke dalam bentuk dictionary
def load_slang_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# Fungsi untuk menggantikan kata kata slang (bahasa gaul) di dalam teks dengan padanan yang lebih baku atau sesuai (berdasarkan kamus slang yang telah dimuat sebelumnya oleh fungsi load_slang_dict)
def replace_slang(text, slang_dict):
    return " ".join(slang_dict.get(word, word) for word in text.split())

# Fungsi untuk memecah teks menjadi token atau kata kata individual
def word_tokenize_wrapper(text):
    return word_tokenize(text)

# Kelas yang digunakan untuk menghilangkan kata kata umum yang tidak penting
class StopWordsId:
    # Menerima stopword_file yang berisi daftar stopword dalam format satu kata perbaris dan menyinpannya ke dalam atribut self.stopword sebagai sebuah set
    def __init__(self, stopwords_file):
        self.stopwords = self.load_stopwords(stopwords_file)
    
    # Membuka file stopword_file, menghapus karakter karakter newline (\n), dan mengembalikan kumpulan (set) stopwords.
    def load_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
        
    # Menerima teks yang akan diproses sebagai parameter
    # Setiap kata akan dicek apakah ada dalam self.stopwords atau tidak
    # Jika kata tersebut tidak ada dalam stopwords dan panjangnya lebih dari 3 karakter (len(word) > 3), maka kata tersebut akan disimpan dalam list filtered_words.
    # Menggabungkan kembali kata-kata yang tersaring ke dalam satu string yang dipisahkan oleh spasi, dan mengembalikan hasilnya.
    def remove_stopwords(self, text):
        words_tokenized = text.split()
        filtered_words = [word for word in words_tokenized if word not in self.stopwords and len(word) > 3]
        return " ".join(filtered_words)

#  Digunakan untuk memfilter kata-kata dalam sebuah dokumen agar hanya menyisakan kata-kata yang ada dalam kamus bahasa Indonesia yang diberikan
class KamusFilter:
    
    # Memuat daftar kata dari file kamus (kamus_file) dan menyimpannya dalam self.term_dict sebagai sebuah set
    def __init__(self, kamus_file):
        self.term_dict = self.load_kamus(kamus_file)

    # Membuka file kamus dan membaca daftar kata. Jika tidak ditemukan akan menampilkan pesan error dan mengembalikan set kosong
    def load_kamus(self, kamus_file):
        try:
            with open(kamus_file, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            print(f"File {kamus_file} tidak ditemukan.")
            return set()

    # Menerima dokumen dalam bentuk list kata dan menghasilkan list yang hanya berisi kata-kata bahasa Indonesia dari dokumen.
    def filter_non_indonesian(self, document):
        return [term for term in document if term in self.term_dict]

# Stemming
factory = StemmerFactory() # Digunakan untuk membuat objek stemmer.
stemmer = factory.create_stemmer() # Membuat objek stemmer yang dapat digunakan untuk melakukan proses stemming.

# Fungsi untuk mengubah setiap kata ke bentuk dasarnya.
def stem_text(text):
    return stemmer.stem(text)

# Fungsi untuk memuat leksikon sentimen
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Fungsi untuk melakukan analisis sentimen pada sebuah teks
def sentiment_analysis(text, pos_lexicon, neg_lexicon):
    words = text.split()
    pos_count = sum(1 for word in words if word in pos_lexicon)
    neg_count = sum(1 for word in words if word in neg_lexicon)
    
    if pos_count > neg_count:
        return 'Positif', pos_count, neg_count
    elif neg_count > pos_count:
        return 'Negatif', pos_count, neg_count
    else:
        return 'Netral', pos_count, neg_count

# Preproses data untuk menjalankan fungsi yang sudah dibuat
def preprocess_data(uploaded_file):
    corpus_df = pd.read_csv(uploaded_file)
    corpus_df = case_folding(corpus_df, 'full_text')
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_tweet_special)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_number)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_punctuation)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_whitespace_LT)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_whitespace_multiple)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_single_char)

    # Load dan replace slang
    slang_dict = load_slang_dict("txt/kamusSlang.json")
    corpus_df['full_text'] = corpus_df['full_text'].apply(lambda x: replace_slang(x, slang_dict))

    # Tokenisasi
    corpus_df['tokenisasi'] = corpus_df['full_text'].apply(word_tokenize_wrapper)

    # Stemming
    corpus_df['stemmed'] = corpus_df['full_text'].apply(stem_text)

    # Inisialisasi stopword dan kamus filter
    stopwords_processor = StopWordsId('txt/stopwords.txt')
    kamus_filter = KamusFilter("txt/kamusIndonesia.txt")

    # Hapus stopword
    corpus_df['stopwords'] = corpus_df['stemmed'].apply(lambda x: stopwords_processor.remove_stopwords(x))

    # Filter term non-indonesia
    corpus_df['filtered'] = corpus_df['stopwords'].apply(lambda x: kamus_filter.filter_non_indonesian(x.split()))

    # Menyimpan hasil preproses kedalam file 'hasil.txt'
    corpus_df.to_csv('preprocessing/hasil.txt', index=None, header=True)
    return corpus_df


# Logika aplikasi
if page == "Preprocessing":
    st.header("Persiapan Data")
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
            df = item['data'] # Ambil DataFrame yang telah diproses
            filename = item['filename'] # Ambil nama file
            st.write(f"Hasil Preprocessing dari file: {filename}:") # Tampilkan nama file
            st.dataframe(df) # Tampilkan DataFrame



elif page == "Clustering":
    st.header("Analisis Faktor") # Menampilkan judul halaman

    if 'preprocessed_data' in st.session_state:
        # Membuat daftar pilihan file dari data yang telah diproses
        file_options = [f"Data set from file: {item['filename']}" for item in st.session_state.preprocessed_data]
        selected_file = st.selectbox("Pilih file yang ingin digunakan untuk klaster:", file_options)
        selected_file_index = file_options.index(selected_file)
        df_selected = st.session_state.preprocessed_data[selected_file_index]['data']

        # Mendefinisikan kalimat untuk setiap centroid
        centroid_sentences = {
            'kompensasi': "kompensasi gaji uang pendapatan dapat penghasilan hasil intensif gaji sedikit gaji banyak bonus",
            'kepuasan_kerja': "kepuasan puas kerja karir bahagia sedih dedikasi nyaman lembur jam kerja waktu cape capek lelah stres stress",
            'aktualisasi': "aktualisasi aktual pengembangan kembang potensi diri kreatif prestasi jabatan jabat gelar",
            'hubungan_kerja': "hubungan rekan kerja suasana dukungan dukung kolaborasi tempat toxic jahat benci suka"
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
            kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=0)
            kmeans.fit(X)
            df_selected['cluster'] = kmeans.labels_

            # Memisahkan klaster menjadi DataFrame yang berbeda
            cluster_0 = df_selected[df_selected['cluster'] == 0][['filtered']].reset_index(drop=True)
            cluster_1 = df_selected[df_selected['cluster'] == 1][['filtered']].reset_index(drop=True)
            cluster_2 = df_selected[df_selected['cluster'] == 2][['filtered']].reset_index(drop=True)
            cluster_3 = df_selected[df_selected['cluster'] == 3][['filtered']].reset_index(drop=True)
            
            # Mendefinisikan label kustom untuk setiap klaster
            cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

            # Menampilkan setiap klaster dengan label deskriptif
            for i, (label, cluster_df) in enumerate(zip(cluster_labels, [cluster_0, cluster_1, cluster_2, cluster_3])):
                st.write(f"### Faktor {label.capitalize()}")
                st.dataframe(cluster_df[['filtered']])

                # Save clusters with descriptive names in session state
                st.session_state[f'cluster_{label}_df'] = cluster_df

            # Menyimpan setiap cluster ke dalam file .txt di folder 'klaster'
            cluster_0.to_csv('klaster/kompensasi.txt', sep='\t', index=False, header=True)
            cluster_1.to_csv('klaster/kepuasan kerja.txt', sep='\t', index=False, header=True)
            cluster_2.to_csv('klaster/aktualisasi.txt', sep='\t', index=False, header=True)
            cluster_3.to_csv('klaster/hubungan kerja.txt', sep='\t', index=False, header=True)

            def remove_punctuation(text):
                if isinstance(text, str):
                    return re.sub(r'[^\w\s]', '', text)
                elif isinstance(text, list):
                    return [re.sub(r'[^\w\s]', '', word) for word in text]
                return text

            # Membersihkan data setiap cluster
            cleaned_data_0 = cluster_0.applymap(remove_punctuation)
            cleaned_data_1 = cluster_1.applymap(remove_punctuation)
            cleaned_data_2 = cluster_2.applymap(remove_punctuation)
            cleaned_data_3 = cluster_3.applymap(remove_punctuation)

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
    if st.button("Proceed to Sentiment Analysis"):

        # Load data kluster dari folder 'klaster'
        def load_cluster_data(cluster_name):
            file_path = f'klaster/{cluster_name}.txt'
            if os.path.exists(file_path):
                return pd.read_csv(file_path, sep='\t')
            else:
                st.error(f"File {cluster_name}.txt tidak ditemukan di folder 'klaster'.")
                return pd.DataFrame()  # Kembalikan DataFrame kosong jika file tidak ditemukan

        # Fungsi untuk menghitung sentimen berdasarkan leksikon
        def analyze_sentiment(text):
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
        cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Menyimpan hasil analisis sentimen dari setiap kluster
        sentiment_dfs = {}

        # Memuat dan memproses setiap kluster
        for label in cluster_labels:
            cluster_df = load_cluster_data(label)  # Ambil DataFrame dari file .txt
            if not cluster_df.empty:
                # Analisis sentimen dan tambahkan label dan skor
                cluster_df[['sentiment_label', 'sentiment_score']] = cluster_df['filtered'].apply(
                    lambda x: pd.Series(analyze_sentiment(x))
                )
                
                # Simpan DataFrame untuk kluster saat ini
                sentiment_dfs[label] = cluster_df

                # Tampilkan hasil analisis sentimen untuk kluster saat ini
                st.write(f"### Analisis Sentimen Faktor {label.capitalize()}")
                st.write(cluster_df[['filtered', 'sentiment_label', 'sentiment_score']])

                # Simpan hasil analisis ke file
                output_file_path = f'analisis/{label}.txt'
                cluster_df[['filtered', 'sentiment_label', 'sentiment_score']].to_csv(
                    output_file_path, sep='\t', index=False, header=['analisis', 'sentiment_label', 'sentiment_score']
                )



elif page == "Data Visualization":
    st.header("Visualisasi Data")
    if st.button("Visualisasikan"):
        # Load hasil analisis sentimen dari file
        def load_sentiment_data(cluster_name):
            return pd.read_csv(f'analisis/{cluster_name}.txt', sep='\t')

        # Label klaster
        cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Load dan visualisasikan data untuk setiap klaster
        for label in cluster_labels:
            cluster_df = load_sentiment_data(label)  # Ambil DataFrame dari file
            if not cluster_df.empty:
                sentiment_counts = cluster_df['sentiment_label'].value_counts()

                # Buat Pie Chart
                st.subheader(f"Visualisasi Analisis Sentimen Faktor {label.capitalize()}")
                st.write(f"Total data pada faktor {label.capitalize()} sebanyak : {len(cluster_df)}")
                fig, ax = plt.subplots()
                colors = ['#ADD8E6', '#87CEFA', '#4682B4']
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Membuat pie chart berbentuk lingkaran.
                st.pyplot(fig)

                # Deskripsi singkat hasil analisis
                st.write(f"Faktor {label.capitalize()}, analisis sentimen menunjukkan distribusi sebagai berikut:")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")

