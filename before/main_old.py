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
def HPreprocessing():
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
        if st.button("Bersihkan"):
            # Proses preprocessing setelah tombol diklik
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            data_baru['teks'] = data_baru['teks'].apply(lambda x: preprocessing(x, slang_dict, stopwords, kamus_indonesia, stemmer))
            data_baru.to_csv('preprocessing/preprocessing.csv', index=0)
            st.write("Preprocessing selesai")

    # menampilkan hasil preprocessing
    if st.button("Tampilkan Hasil Preprocessing"):
        hasilpreprocessing = pd.read_csv("preprocessing/preprocessing.csv")
        st.dataframe(hasilpreprocessing)

# Halaman Klasterisasi
def HClustering():
    st.title("Halaman Klasterisasi")
    if st.button("Klaster"):
        df_selected = pd.read_csv('preprocessing/preprocessing.csv')

        # cek nilai null dan memastikan data berupa string
        df_selected['teks'] = df_selected['teks'].fillna('').astype(str)

        def load_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        stopwords2 = load_file('txt/stopwords-2.txt')

        # fungsi stopwords
        def preprocessing(text, stopwords):
            text = [word for word in text.split() if word not in stopwords]
            return ' '.join(text) 

        # menghapus stopwords dari kolom 'teks' dan menyimpannya dalam kolom baru 'teks-kmeans'
        df_selected['teks'] = df_selected['teks'].apply(lambda x: preprocessing(x, stopwords2))

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
            df_selected.at[pos, 'teks'] = sentence

        # vektorisasi teks menggunakan TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_selected['teks']) 
        lokasi_centroid = X[list(posisi.keys())].toarray()

        # K-means clustering
        kmeans = KMeans(n_clusters=4, init=lokasi_centroid, n_init=10, random_state=0)
        kmeans.fit(X)

        # menyimpan hasil klaster pada kolom baru 'label-klaster'
        df_selected['label_klaster'] = kmeans.labels_

        # nampilkan Davies-Bouldin Score
        db_score = davies_bouldin_score(X.toarray(), kmeans.labels_)
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")

        # menghapus baris yang berisi teks yang dijadikan centroid sebelumnya dari dataframe
        centroid_texts = set(centroid_sentences.values())
        df_selected = df_selected[~df_selected['teks'].isin(centroid_texts)].reset_index(drop=True)
        df_selected = df_selected[~df_selected['teks'].str.strip().eq('')]

        df_selected.to_csv("dataset_berlabel/klaster_prediksi.csv", index=False) # digunakan untuk confussion matrix k-means

        # pisah klaster menjadi dataframe yang berbeda dan menambahkan kolom 'label'
        clusters = [df_selected[df_selected['label_klaster'] == i][['teks', 'label_klaster']].reset_index(drop=True) for i in range(4)]

        # label untuk setiap klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # menampilkan dan menyimpan hasil
        for label, cleaned_data in zip(label_klaster, clusters):
            st.subheader(f"Faktor {label.capitalize()}")
            st.dataframe(cleaned_data)
            # Menyimpan data ke file
            cleaned_data.to_csv(f'klaster/{label}.csv', sep='\t', index=False, header=True)

def HSentimentAnalysis():
    st.title("Halaman Klasifikasi")
    if st.button("Analisis Sentimen"):

        # membaca data yang telah dipreproses
        dfkKompensasi = pd.read_csv("klaster/kompensasi.csv", sep='\t')
        dfkKompensasi['teks'] = dfkKompensasi['teks'].fillna('').astype(str)

        dfkKepuasanKerja = pd.read_csv("klaster/kepuasan kerja.csv", sep='\t')
        dfkKepuasanKerja['teks'] = dfkKepuasanKerja['teks'].fillna('').astype(str)

        dfkAktualisasi = pd.read_csv("klaster/aktualisasi.csv", sep='\t')
        dfkAktualisasi['teks'] = dfkAktualisasi['teks'].fillna('').astype(str)

        dfkHubunganKerja = pd.read_csv("klaster/hubungan kerja.csv", sep='\t')
        dfkHubunganKerja['teks'] = dfkHubunganKerja['teks'].fillna('').astype(str)

        # memuat model dan vectorizer yang telah disimpan
        model = joblib.load('model/model_sentimen.pkl')
        vectorizer = joblib.load('model/vectorizer_sentimen.pkl')
        
        # vektorisasi dari model yang telah dilatih
        XKompensasi = vectorizer.transform(dfkKompensasi['teks']) 
        xKepuasanKerja = vectorizer.transform(dfkKepuasanKerja['teks']) 
        XAktualisasi = vectorizer.transform(dfkAktualisasi['teks'])
        xHubunganKerja = vectorizer.transform(dfkHubunganKerja['teks']) 

        # melakukan prediksi menggunakan model yang sudah dilatih
        LRKompensasi = model.predict(XKompensasi)
        LRKepuasanKerja = model.predict(xKepuasanKerja)
        LRAktualisasi = model.predict(XAktualisasi)
        LRHubunganKerja = model.predict(xHubunganKerja)

        # menambahkan hasil prediksi ke dalam dataset baru
        dfkKompensasi['label_sentimen'] = LRKompensasi
        dfkKepuasanKerja['label_sentimen'] = LRKepuasanKerja
        dfkAktualisasi['label_sentimen'] = LRAktualisasi
        dfkHubunganKerja['label_sentimen'] = LRHubunganKerja

        # menyimpan hasil klasifikasi ke folder hasilklasifikasi
        dfkKompensasi.to_csv('klasifikasi/kompensasi.csv', index=False, sep='\t')
        dfkKepuasanKerja.to_csv('klasifikasi/kepuasan kerja.csv', index=False, sep='\t')
        dfkAktualisasi.to_csv('klasifikasi/aktualisasi.csv', index=False, sep='\t')
        dfkHubunganKerja.to_csv('klasifikasi/hubungan kerja.csv', index=False, sep='\t')
        
        # menampilkan file hasil klasifikasi
        st.subheader("Analisis Sentimen Faktor Kompensasi")
        st.dataframe(dfkKompensasi)
        st.subheader("Analisis Sentimen Faktor Kepuasan Kerja")
        st.dataframe(dfkKepuasanKerja)
        st.subheader("Analisis Sentimen Faktor Aktualisasi")
        st.dataframe(dfkAktualisasi)
        st.subheader("Analisis Sentimen Faktor Hubungan Kerja")
        st.dataframe(dfkHubunganKerja)

# Halaman Visualisasi
def HDataVisualization():
    st.title("Halaman Visualisasi Data")
    if st.button("Visualisasi"):
        def memuat_data_sentimen(cluster_name):
            return pd.read_csv(f'klasifikasi/{cluster_name}.csv', sep='\t')

        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Visualisasi Bar Chart untuk jumlah data pada setiap klaster
        jumlah_data_klaster = []
        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label)
            if not dataframe_klaster.empty:
                jumlah_data_klaster.append(len(dataframe_klaster))
        st.subheader("Distribusi Data Pada Faktor-Faktor yang Mempengaruhi Perpindahan Karir")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(label_klaster, jumlah_data_klaster, color='green')
        ax.set_title("Jumlah Data untuk Setiap Klaster", fontsize=16)
        ax.set_xlabel("Faktor-faktor", fontsize=14)
        ax.set_ylabel("Jumlah Data", fontsize=14)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)

        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label) 
            if not dataframe_klaster.empty:
                jumlah_sentimen = dataframe_klaster['label_sentimen'].value_counts()

                st.subheader(f"Faktor {label.capitalize()}")
                st.write(f"Total data pada klaster {label.capitalize()} sebanyak: {len(dataframe_klaster)}")
                explode = (0.03, 0.03, 0.03)
                fig, ax = plt.subplots(figsize=(10,5))
                colors = ['#90EE90', '#32CD32', '#228B22']
                ax.pie(jumlah_sentimen, labels=jumlah_sentimen.index, autopct='%1.1f%%', startangle=0, colors=colors, pctdistance=0.7, explode=explode)
                ax.axis('equal')
                st.pyplot(fig)

                st.write(f"Distribusi sentimen pada klaster {label.capitalize()} menunjukkan:")
                for sentiment, count in jumlah_sentimen.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")

# navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing", "Clustering", "Sentiment Analysis", "Data Visualization"])

# tampilan halaman yang dipilih user
if page == "Preprocessing":
    HPreprocessing()
elif page == "Sentiment Analysis":
    HSentimentAnalysis()
elif page == "Clustering":
    HClustering()
elif page == "Data Visualization":
    HDataVisualization()