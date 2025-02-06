# Import library
import json
import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# fungsi pembacaan file
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(json.load(file))

def load_file(file_path):
    with open(file_path, 'r', encoding='    utf-8') as file:
        return set(file.read().splitlines())

# load file sumberdaya
slang_dict = json.load(open("txt/kamusSlang.json", "r", encoding="utf-8"))
stopwords = load_file('txt/stopwords-1.txt')
kamus_indonesia = load_file('txt/kamusIndonesia.txt')
pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

# Halaman Preprocessing
def preprocessing_page():
    st.title("Halaman Preprocessing")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # fungsi preprocessing
    def preprocessing(text, slang_dict, stopwords, kamus_indonesia, stemmer):
        text = text.lower()  # Case folding
        text = re.sub(r"\\t|\\n|\\u|\\|http[s]?://\\S+|[@#][A-Za-z0-9_]+", " ", text)  # Hapus karakter khusus
        text = re.sub(r"\\d+", "", text)  # Hapus angka
        text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
        text = re.sub(r"\\s+", ' ', text).strip()  # Rapikan spasi ganda
        text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Hapus huruf tunggal
        text = ' '.join([slang_dict.get(word, word) for word in text.split()])  # Normalisasi slang
        text = word_tokenize(text)  # Tokenisasi
        text = [stemmer.stem(word) for word in text]  # Stemming
        text = [word for word in text if word not in stopwords and len(word) > 3 and word in kamus_indonesia]  # Filter
        return ' '.join(text)
    
    # lokasi tujuan dataset hasil crawling
    file_path = 'code_filter_crawling/crawling.csv'
    with open(file_path, "r", encoding="utf-8") as f:
        csv_raw_data = f.read()
    st.write("Unduh Dataset")
    st.download_button(
        label="Unduh Dataset",
        data=csv_raw_data,
        file_name="crawling.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # upload dataset baru
    uploaded_file = st.file_uploader("Upload File Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # menampilkan data sebelum preprocessing
        data_baru = pd.read_csv(uploaded_file)
        data_baru = data_baru.rename(columns={"full_text": "teks"})
        st.write("Data sebelum preprocessing:")
        st.dataframe(data_baru[['teks']])

        # menambahkan tombol preproses
        if st.button("Preproses Data"):
            # Proses preprocessing setelah tombol diklik
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            data_baru['teks'] = data_baru['teks'].apply(lambda x: preprocessing(x, slang_dict, stopwords, kamus_indonesia, stemmer))
            data_baru.to_csv('preprocessing/preprocessing.csv', index=0)
            st.write("Preprosesing selesai")

    # menampilkan hasil preprocessing
    if st.button("Tampilkan Hasil Preprosesing"):
        hasilpreprocessing = pd.read_csv("preprocessing/preprocessing.csv")
        st.dataframe(hasilpreprocessing)

# Halaman Klasifikasi
def klasifikasi_page():
    st.title("Halaman Klasifikasi")
    if st.button("Analisis Sentimen"):

        # memuat model dan vectorizer yang telah disimpan
        model = joblib.load('model/model_sentimen.pkl')
        vectorizer = joblib.load('model/vectorizer_sentimen.pkl')

        # membaca data yang telah dipreproses
        data_baru = pd.read_csv("preprocessing/preprocessing.csv")
        data_baru['teks'] = data_baru['teks'].fillna('').astype(str)
        data_baru = data_baru.drop_duplicates(subset=['teks'])

        # vektorisasi dari model yang telah dilatih
        X_baru = vectorizer.transform(data_baru['teks']) 

        # melakukan prediksi menggunakan model yang sudah dilatih
        prediksi = model.predict(X_baru)

        # menambahkan hasil prediksi ke dalam dataset baru
        data_baru['label'] = prediksi

        # menyimpan hasil klasifikasi ke folder hasilklasifikasi
        data_baru.to_csv('klasifikasi/klasifikasi.csv', index=False)
        
        # menampilkan file hasil klasifikasi
        st.write("Hasil Klasifikasi")
        st.dataframe(data_baru)

# Halaman Klasterisasi
def klasterisasi_page():
    st.title("Halaman Klasterisasi")
    if st.button("Clustering"):
        df_selected = pd.read_csv('klasifikasi/klasifikasi.csv')

        # cek nilai null dan memastikan data berupa string
        df_selected['teks'] = df_selected['teks'].fillna('').astype(str)

        # load stopwords
        def load_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return set(file.read().splitlines())
            except FileNotFoundError:
                st.error(f"File '{file_path}' tidak ditemukan.")
                return set()
        stopwords2 = load_file('txt/stopwords-2.txt')

        # fungsi stopwords
        def preprocessing(text, stopwords):
            text = [word for word in text.split() if word not in stopwords]
            return ' '.join(text) 

        # menghapus stopwords dari kolom 'teks' dan menyimpannya dalam kolom baru 'teks-kmeans'
        df_selected['teks-kmeans'] = df_selected['teks'].apply(lambda x: preprocessing(x, stopwords2))

        centroid_sentences = {
            'kompensasi': "gaji kompensasi",
            'kepuasan_kerja': "mental stres jam",
            'aktualisasi': "berkembang kembang jabatan skill",
            'hubungan_kerja': "hubungan jahat hubungan baik lingkung"
        }

        # menghitung posisi dalam DataFrame untuk setiap centroid
        num_rows = len(df_selected)
        posisi = {
            int(num_rows * 0.25): centroid_sentences['kompensasi'],
            int(num_rows * 0.50): centroid_sentences['kepuasan_kerja'],
            int(num_rows * 0.75): centroid_sentences['aktualisasi'],
            int(num_rows * 0.90): centroid_sentences['hubungan_kerja']
        }

        # menyisipkan kalimat ke dalam DataFrame pada posisi yang ditentukan
        for pos, sentence in posisi.items():
            df_selected.at[pos, 'teks-kmeans'] = sentence

        # vektorisasi teks menggunakan TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_selected['teks-kmeans'])  # Menggunakan kolom teks yang telah dibersihkan  
        lokasi_centroid = X[list(posisi.keys())].toarray()

        # K-means clustering
        kmeans = KMeans(n_clusters=4, init=lokasi_centroid, n_init=10, random_state=0)
        kmeans.fit(X)

        # menyimpan hasil klaster pada kolom baru 'label-klaster'
        df_selected['label-klaster'] = kmeans.labels_

        # nampilkan Davies-Bouldin Score
        db_score = davies_bouldin_score(X.toarray(), kmeans.labels_)
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")

        # menghapus baris yang berisi teks yang dijadikan centroid sebelumnya dari dataframe
        centroid_texts = set(centroid_sentences.values())
        df_selected = df_selected[~df_selected['teks-kmeans'].isin(centroid_texts)].reset_index(drop=True)

        df_selected.to_csv("dataset_berlabel/klaster_prediksi.csv", index=False) # Digunakan untuk confussion matrix k-means

        # pisah klaster menjadi dataframe yang berbeda dan menambahkan kolom 'label'
        clusters = [df_selected[df_selected['label-klaster'] == i][['teks-kmeans', 'label', 'label-klaster']].reset_index(drop=True) for i in range(4)]

        # label untuk setiap klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # menampilkan dan menyimpan hasil
        for label, cleaned_data in zip(label_klaster, clusters):
            st.subheader(f"Faktor {label.capitalize()}")
            st.dataframe(cleaned_data)
            # Menyimpan data ke file
            cleaned_data.to_csv(f'klaster/{label}.csv', sep='\t', index=False, header=True)

# Halaman Visualisasi
def visualisasi_page():
    st.header("Visualisasi Data")
    if st.button("Data Visualization"):
        # Load hasil analisis sentimen dari file
        def memuat_data_sentimen(cluster_name):
            return pd.read_csv(f'klaster/{cluster_name}.csv', sep='\t')

        # Label klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Visualisasi Bar Chart untuk jumlah data pada setiap klaster
        jumlah_data_klaster = []
        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label)  # Ambil DataFrame dari file
            if not dataframe_klaster.empty:
                jumlah_data_klaster.append(len(dataframe_klaster))

        # Bar Chart jumlah data pada setiap klaster
        st.subheader("Jumlah Data pada Setiap Klaster")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(label_klaster, jumlah_data_klaster, color='#87CEFA')
        ax.set_title("Jumlah Data untuk Setiap Klaster", fontsize=16)
        ax.set_xlabel("Klaster", fontsize=14)
        ax.set_ylabel("Jumlah data", fontsize=14)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)

        # Load dan visualisasikan data untuk setiap klaster
        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label)  # Ambil DataFrame dari file
            if not dataframe_klaster.empty:
                jumlah_sentimen = dataframe_klaster['label'].value_counts()

                # Buat Pie Chart untuk distribusi sentimen
                st.subheader(f"Visualisasi Data Sentimen Klaster {label.capitalize()}")
                st.write(f"Total data pada klaster {label.capitalize()} sebanyak: {len(dataframe_klaster)}")
                explode = (0.03, 0.03, 0.03)  # Jarak antara potongan pie dan pusatnya
                fig, ax = plt.subplots(figsize=(5,5))
                colors = ['#ADD8E6', '#87CEFA', '#4682B4']
                ax.pie(jumlah_sentimen, labels=jumlah_sentimen.index, autopct='%1.1f%%', startangle=0, colors=colors, pctdistance=0.7, explode=explode)
                ax.axis('equal')  # Membuat pie chart berbentuk lingkaran.
                st.pyplot(fig)

                # Deskripsi singkat hasil analisis
                st.write(f"Distribusi sentimen pada klaster {label.capitalize()} menunjukkan:")
                for sentiment, count in jumlah_sentimen.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")
  

# navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing", "Sentiment Analysis", "Clustering", "Data Visualization"])

# tampilan halaman yang dipilih user
if page == "Preprocessing":
    preprocessing_page()
elif page == "Sentiment Analysis":
    klasifikasi_page()
elif page == "Clustering":
    klasterisasi_page()
elif page == "Data Visualization":
    visualisasi_page()
