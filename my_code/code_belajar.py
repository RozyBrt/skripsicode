import streamlit as st

import pandas as pd
import json
import re
import string

from nltk.tokenize import word_tokenize as token_kata 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # stemming otomatis

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF
from sklearn.cluster import KMeans # K- MEANS


from sklearn.metrics import silhouette_score # silhouette
import matplotlib.pyplot as plt # visualisasi

import nltk
nltk.download('punkt')

st.set_page_config(page_title="TUGAS AKHIR")

# default page
st.title("ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI PERPINDAHAN KARIR DENGAN PEMANFAATAN ASPECT-BASED SENTIMENT ANALYSIS MENGGUNAKAN METODE K-MEANS")
page = st.sidebar.selectbox("tentukan halaman:", ["preprocessing", "klastering", "analisis sentimen", "visualisasi data"])

# # import data (alternatif)
# file_path = "csv/data_tweet.csv" # lokasi hasil scraping (data awal)

# # fungsi sementara untuk membuka file dan menutup setelah operasi selesai 
# with open (file_path, "r") as f: # "r" = mode pembacaan file.  "f" variabel sementara untuk .read()
#     csv_raw_data = f.read() 


# case folding (menggunakan fungsi built-in dalam pandas)
def ubah_ke_huruf_kecil(dataframe, column_name): # fungsi ini akan dipanggil (dataframe | nama kolom)
    dataframe[column_name] = dataframe[column_name].str.lower # .str.lower fungsi dari pandas
    return dataframe # tetap dikembalikan pada variabel dataframe, dst...

# (menggunakan fungsi kustom .apply yg tidak ada dalam pandas)
def bersihkan_karakter_twitter(text):
    # 1. \\t menjadi spasi (" "): Menggantikan tabulasi (tab) dengan spasi.
    # 2. \\n menjadi spasi (" "): Menggantikan newline (enter) dengan spasi.
    # 3. \\u menjadi spasi (" "): Menggantikan escape sequence Unicode dengan spasi.
    # 4. \\ menjadi kosong (""): Menghapus backslash.
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'ignore').decode('ascii')
    # note : dipelajari lagi
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ") # menghapus hyperlink

def hapus_angka(text):
    # \d : angka
    return re.sub(r"\d"+ "", text) # r : fungsi read (ketika ditemukan maka diganti...) replace string kosong "" (pada text)

def hapus_tandabaca(text):
    # tahapan ini menggunanakn tabel translasi : .translate() dan dikustom str.maketrans() menghapus karakter tertentu
    # Membuat tabel translasi untuk menggantikan atau menghapus karakter. Sintaks:
    # str.maketrans(tabel_ganti, tabel_hapus, tabel_ganti_tambahan)
    # alasan : lebih efisien dari pada replace() yang berulang kali proses nya.
    return text.translate(str.maketrans("","", string.punctuation))

def hapus_spasi_awalakhir(text):
    # .strip() : menghapus karakter tertentu pada awal dan akhir string
    # default yang dihapus : spasi(),tabulasi(\t),newline(\n),return(\r)
    return text.strip()

def ganti_spasi_tunggal(text):
    # hapus spasi berlebih
    return re.sub(r'\s+',' ',text)

def hapus_karakter_tunggal(text):
    # \b :boundaries (batas kata) secara utuh dan akan dihapus
    return re.sub(r"\b[a-zA-Z]\b","",text)

def muat_kamus_slang(file_path):
    # fungsi cara membaca file json (kamus) yang akan dipanggil 
    with open(file_path,"r",encoding="utf-8") as f:
        return json.load

def slang_ke_baku(text, kamus_bahasa_gaul):
    # .split() : memecah kata berdasarkan spasi
    # .get akses kamus (dict)
    # word pertama dari kamus, word kedua default (kata itu sendiri)
    # " ".join : disatukan lagi dengan spasi sebagai pemisah
    return " ".join(kamus_bahasa_gaul.get(word, word) for word in text.split())

def bungkus_tokenisasi_kata(text):
    # tokenisasi dari NLTK (diatas)
    return token_kata(text)


# stopword kustom
# stopword dipecah menjadi 2 bagian (class), pertama untuk ngapus stopwords, kedua untuk ngapus kata yang ga punya arti dan merujuk

class StopWordsIndo: 
    # 1. buat fungsi read file stopwords
    # 2. buat fungsi pemecah stopwords jadi baris dan himpunan
    # 3. buat fungsi penghapusnya
    def __init__(self, stopwords_file):
        # menggunakan self sebagai atribut variabel fungsi agar bisa diakses pada semua def
        self.stopwords = self.olah_stopword(stopwords_file)

    def olah_stopword(self, stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            # .splitlines() : memecahkan data menjadi baris per baris
            # set() : mengubah daftar baris menjadi himpunan untuk hilangkan duuplikat
            return set(f.read().splitlines()) 

    def hapus_stopwords(self,text):
        pisahkan_kata = text.split() # pisahkan kata jadi token 'contoh', 'contoh2
        # word not in self.stopwords (yang disimpan di word pertama hanya kata yang tidak ada di self.stopwords dan len > 3)
        kata_bersih = [word for word in pisahkan_kata if word not in self.stopwords and len(word) > 3] # gunain word karena token
        return " ".join(kata_bersih) 
    
class KamusFilter:
    def __init__(self, kamus_file):
        self.term_dict = self.baca_kamus(kamus_file)

    def baca_kamus(self, kamus_file):
        # gunain fungsi try untuk mengecek kamus ada atau tidak
        try:
            # jika ditemukan di split menjadi baris perbaris dan digabung lagi menjadi sebuah himpunan
            with open(kamus_file, 'r', encoding='utf-k8') as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
                print(f'file {kamus_file} tidak ditemukan.')
    # fungsi menghapus yang bukan dari kamus (pakai atribut term)
    def hapus_bukan_id(self, document):
        return [term for term in document if term in self.term_dict]
    # cara kerja fungsi nya sama dengan stopwords, dikarenakan ada fungsi untuk mengecek kamus tersebut maka splitline nya di dalam try

# stemming
pengolahdata = StemmerFactory() # manggil dari class
stemmer = pengolahdata.create_stemmer() # manggil dari dalam class (def nya)

# fungsi untuk stem
def sederhanakan_teks(text):
    return stemmer.stem(text)

# membaca file berdasarkan format JSON
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# menyatukan pemanggilan preprocessing dalam satu def
def preprocess_data(uploaded_file):
    corpus_df = pd.read_csv(uploaded_file)
    # case folding
    corpus_df = ubah_ke_huruf_kecil(corpus_df, "full_text")
    # hapus angka
    corpus_df["full_text"] = corpus_df["full_text"].apply(bersihkan_karakter_twitter)
    corpus_df["full_text"] = corpus_df["full_text"].apply(hapus_angka)
    corpus_df["full_text"] = corpus_df["full_text"].apply(hapus_tandabaca)
    corpus_df["full_text"] = corpus_df["full_text"].apply(hapus_spasi_awalakhir)
    corpus_df["full_text"] = corpus_df["full_text"].apply(ganti_spasi_tunggal)
    corpus_df["full_text"] = corpus_df["full_text"].apply(hapus_karakter_tunggal)
    
    # load , normalisasi bahasa gaul
    kamus_bahasa_gaul = muat_kamus_slang("txt/kamusSlang.json")
    # lambda x untuk menyimpan fungsi slang_ke_baku ke x dan dipanggil di (x, kamus_bahasa_gaul)
    corpus_df["full_text"] = corpus_df["full_text"].apply(lambda x: slang_ke_baku(x, kamus_bahasa_gaul))
    
    # tokenisasi (pada tahapan stemming membutuhkan data berupa token)
    corpus_df["full_text"] = corpus_df["full_text"].apply(bungkus_tokenisasi_kata)
    
    # stemming (motong imbuhan kata)
    corpus_df["full_text"] = corpus_df["full_text"].apply(sederhanakan_teks)
    
    # load stopwords dan kamus filter
    stopwords_processor = StopWordsIndo("txt/stopwords.txt")
    kamus_filter = KamusFilter("txt/kamusIndonesia.txt")
    # stopwords (cara manggilnya disesuaikan dengan pendefinisian nya)
    corpus_df["full_text"] = corpus_df["full_text"].apply(lambda x : stopwords_processor.hapus_stopwords(x))
    corpus_df["full_text"] = corpus_df["full_text"].apply(lambda x: kamus_filter.hapus_bukan_id(x.split)) # displit karena di fungsi tidak ada split (memastikan kata terpecah)

    # simpan csv
    # header diperlukan dikarenakan karena pada proses selanjutnya mengambil data berdasarkan nama kolom ['full_text']
    corpus_df.to_csv('preprocessing/hasil.txt', index=None, header=True)
    return corpus_df # semua hasil tersebut disimpan pada corpus_df

    
# APLIKASI (menggunakan fungsi fungsi yang ada pada steamlit)

# default page di 'preprocessing'
if page == "Preprocessing":
    st.header("Persiapkan Data")

    # Tombol download data (Data hasil scrapping dari developer)
    st.write("Unduh dataset untuk melanjutkan proses analisis dibawah")
    st.download_button(
        label="Download File CSV",
        data=csv_raw_data,
        file_name="dataset.csv", # nama 
        mime="text/csv", # jenis file yang akan diunduh
        use_container_width= True,
    )

    # variabel (untuk file yang diupload)
    uploaded_file = st.file_uploader('pilih file csv', type='csv')

    if 'uploaded_file' in st.session_state: # session_state 
        st.write('file yang sedang diproses ...')
        st.write(st.session_state.uploaded_file.name) #.name fungsi dari st (penamaan)   

    if uploaded_file is not None:
        # menyimpan data terakhir berupa file csv yang telah diupload (tanpa harus disimpan di repo)
        st.session_state.uploaded_file = uploaded_file
        # penamaan file (file terakhir)
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = uploaded_file.name 

    if 'uploaded_file' in st.session_state:

        if 'preprocessed' not in st.session_state or not st.session_state.preprocessing:
            if st.button('bersihkan'):
                df_preprocessed = preprocess_data(st.session_state.uploaded_file) # fungsi preprocessing dan pemanggilan dilakukan dipage yang sama
                # simpan hasil preprocessing di session_state untuk backup
                if 'preprocessed_data' not in st.session_state:
                    st.session_state.preprocessed_data = [] # inisialisasi (Dikosongi dulu)
                
                # penamaan hasil proses preprocessing (simpan)
                st.session_state.preprocessed_data.append({
                    'data' : df_preprocessed,
                    'filename' : st.session_state.uploaded_file.name
                })
                st.success("preprocessing selesai")
        else:
            # untuk menghindari duplikasi proses
            st.warning('file ini sudah diproses sebelumnya')
        

        # menampilkan data yang sudah di proses (output)
        if 'preprocessed_data' in st.session_state:
            for item in st.session_state.preprocessed_data:
                # dipanggil dari .append diatas
                df = item['data']
                filename = item['filename']
                st.write(f'hasil preprocessing dari file: {filename}:')
                st.dataframe(df)


elif page == 'klastering':
    st.header('analisis faktor')

    # ngambil data hasil preprocessing
    def load_preprocessed_data():
        file_path = 'preprocessing/hasil.txt'
        return pd.read_csv(file_path)
    df_selected = load_preprocessed_data()

    # KATA KUNCI
    centroid_sentences = {
        'kompensasi': "kompensasi gaji uang pendapatan dapat penghasilan hasil intensif gaji sedikit gaji banyak bonus",
        'kepuasan_kerja': "kepuasan puas kerja karir bahagia sedih dedikasi nyaman lembur jam kerja waktu cape capek lelah stres stress",
        'aktualisasi': "aktualisasi aktual pengembangan kembang potensi diri kreatif prestasi jabatan jabat gelar",
        'hubungan_kerja': "hubungan rekan kerja suasana dukungan dukung kolaborasi tempat toxic jahat benci suka"
    }


    # MELETAKKAN CENTROID
    num_rows = len(df_selected) # hitung baris
    # posisi centroid
    centroid_positions = {
        # key : value
        int(num_rows * 0.25): centroid_sentences['kompensasi'],
        int(num_rows * 0.50): centroid_sentences['kepuasan_kerja'],
        int(num_rows * 0.75): centroid_sentences['aktualisasi'],
        int(num_rows * 0.90): centroid_sentences['hubungan_kerja']
    }

    # nyisipkan KATA KUNCI berdasarkan lokasi diatas (perulangan)
    # pos : posisi baris (indeks), sentence : "kompensasi gaji uang ..."
    for pos, sentence in centroid_positions.items():  # .items() mengembalikan nilai list tuple. index pertama tuple :key, index kedua tuple :value
        # DataFrame.at[index, column] = value
        df_selected.at[pos, 'filtered'] = sentence

    # memastikan semua data berupa string
    df_selected['filtered'] = df_selected['filtered'].apply(lambda x: str(x))

    # TD-IDF
    texts = df_selected["filtered"].astype(str)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # mengambil centroid awal
    initial_centroids = X[list(centroid_positions.keys())].toarray() # mengambil key aja, disimpan dalam bentuk array dikarenakan datanya homogen


    # KLASTERISASI
    if st.button("klaster"):
        # 1 : menentukan jumlah klaster
        # 2 : posisi centroid
        # 3 : jumlah iterasi
        # 4 : mengatur seed acak, dibuat 0 agar lebih konsisten
        kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=0)
        kmeans.fit(X) # melatih model kmeans pada data X
        df_selected['cluster'] = kmeans.labels_ # .labels_ = nomor klaster : 0 , 1, 2, 3


        # menghitung silhouette scroe
        silhouette_avg = silhouette_score(X, kmeans.labels_) # salah satu evaluasi (klasterisasi)
        st.write(f"silhouette score: {silhouette_avg:.2f}") #.2f dibulatkan menjadi dua angka desimal

        # mengambil klaster di df_selected['cluster'] berdasarkan angka .labels_ : [0,1,2,3] disimpan di kolom ['filtered']
        cluster_0 = df_selected[df_selected['cluster'] == 0][['filtered']].reset_index(drop=True) # akan menghapus index yang lama , dan meresetnya dari index 0
        cluster_1 = df_selected[df_selected['cluster'] == 1][['filtered']].reset_index(drop=True)
        cluster_2 = df_selected[df_selected['cluster'] == 2][['filtered']].reset_index(drop=True)
        cluster_3 = df_selected[df_selected['cluster'] == 3][['filtered']].reset_index(drop=True)

        # inisialsiasi nama label (yang akan digunakan) pada tampilan (nanti)
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']


        # MENAMPILKAN KLASTERISASI
        # perulangan (4cluster) 
        # 1. zip() menggabungkan iterasi dalam (...)
        # 2. label_klaster dan list[cluster...] menjadi pasangan (tuple)
        # 3. enumerate menambahkan indeks pada iterasi i
        for i, (label, dataframe_klaster) in enumerate(zip(label_klaster, [cluster_0, cluster_1, cluster_2, cluster_3])):
            st.write(f"faktor {label.capitalize()}") # label dibuat huruf besar
            st.dataframe(dataframe_klaster[['filtered']]) # nampilin dataframe
            
            # Menyimpan setiap cluster ke dalam file .txt di folder 'klaste
            dataframe_klaster.to_csv(f"klaster/{label}.txt",sep='\t', index=False, header=True)

        def hapus_tandaaca(text): 
            if isinstance(text, str):
                return re.sub(r'[^\w\s]', '', text)
            elif isinstance(text, list):
                return [re.sub (r'^\w\s', '', word) for word in text]
            return text
        # menyimpan hasil klasterisasi dalam variabel baru
        cleaned_data_0 = cluster_0.applymap(hapus_tandabaca) # menggunakan .applymap dikarenakan satu variabel satu data kolom
        cleaned_data_1 = cluster_1.applymap(hapus_tandabaca)
        cleaned_data_2 = cluster_2.applymap(hapus_tandabaca)
        cleaned_data_3 = cluster_3.applymap(hapus_tandabaca)
        # simpan dataframenya dalam bentuk .txt
        cleaned_data_0.to_csv('klaster/kompensasi.txt', sep='\t', index=False, header=True) # setiap kolom dipisah oleh tabulator (tab)
        cleaned_data_1.to_csv('klaster/kepuasan kerja.txt', sep='\t', index=False, header=True)
        cleaned_data_2.to_csv('klaster/aktualisasi.txt', sep='\t', index=False, header=True)
        cleaned_data_3.to_csv('klaster/hubungan kerja.txt', sep='\t', index=False, header=True)

elif page == 'analisis sentimen':
    st.header('analisis sentimen faktor')

    # load file json
    pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

    # tombol memproses analisis
    if st.button('lakukan proses analisis sentimen'):
        
        def load_cluster_data(cluster_name):
            file_path = f'klaster/{cluster_name}.txt' # menyimpan file pada load_cluster_data
            if os.path.exits(file_path)

