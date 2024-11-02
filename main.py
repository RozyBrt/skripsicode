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
nltk.download('punkt')

# setting tab web
st.set_page_config(page_title="ABSA-KMeans", page_icon="☕")

# Streamlit application
st.title("ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI PERPINDAHAN KARIR DENGAN PEMANFAATAN ASPECT-BASED SENTIMENT ANALYSIS MENGGUNAKAN METODE K-MEANS")
page = st.sidebar.selectbox("Select a page:", ["Preprocessing", "Clustering", "Sentiment Analysis", "Data Visualization"])


# Preprocessing functions
def case_folding(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].str.lower()
    return dataframe

def remove_tweet_special(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

def remove_number(text):
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_whitespace_LT(text):
    return text.strip()

def remove_whitespace_multiple(text):
    return re.sub(r'\s+', ' ', text)

def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def load_slang_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def replace_slang(text, slang_dict):
    return " ".join(slang_dict.get(word, word) for word in text.split())

def word_tokenize_wrapper(text):
    return word_tokenize(text)

class StopWordsId:
    def __init__(self, stopwords_file):
        self.stopwords = self.load_stopwords(stopwords_file)
    
    def load_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def remove_stopwords(self, text):
        words_tokenized = text.split()
        filtered_words = [word for word in words_tokenized if word not in self.stopwords and len(word) > 3]
        return " ".join(filtered_words)

class KamusFilter:
    def __init__(self, kamus_file):
        self.term_dict = self.load_kamus(kamus_file)

    def load_kamus(self, kamus_file):
        try:
            with open(kamus_file, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            print(f"File {kamus_file} tidak ditemukan.")
            return set()

    def filter_non_indonesian(self, document):
        return [term for term in document if term in self.term_dict]

# Stemming setup
factory = StemmerFactory()
stemmer = factory.create_stemmer()

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

# Preprocessing function
def preprocess_data(uploaded_file):
    corpus_df = pd.read_csv(uploaded_file)
    corpus_df = case_folding(corpus_df, 'full_text')
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_tweet_special)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_number)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_punctuation)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_whitespace_LT)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_whitespace_multiple)
    corpus_df['full_text'] = corpus_df['full_text'].apply(remove_single_char)

    # Load slang dictionary and replace slang
    slang_dict = load_slang_dict("txt/kamusSlang.json")
    corpus_df['full_text'] = corpus_df['full_text'].apply(lambda x: replace_slang(x, slang_dict))

    # Tokenization
    corpus_df['tokenisasi'] = corpus_df['full_text'].apply(word_tokenize_wrapper)

    # Stemming
    corpus_df['stemmed'] = corpus_df['full_text'].apply(stem_text)

    # Initialize stopwords and kamus filters
    stopwords_processor = StopWordsId('txt/stopwords.txt')
    kamus_filter = KamusFilter("txt/kamusIndonesia.txt")

    # Remove stopwords
    corpus_df['stopwords'] = corpus_df['stemmed'].apply(lambda x: stopwords_processor.remove_stopwords(x))

    # Filter non-Indonesian terms
    corpus_df['filtered'] = corpus_df['stopwords'].apply(lambda x: kamus_filter.filter_non_indonesian(x.split()))

    # Save the processed text to 'hasil.txt'
    corpus_df.to_csv('preprocessing/hasil.txt', index=None, header=True)
    return corpus_df


# Main application logic
if page == "Preprocessing":
    st.header("Persiapan Data")
    uploaded_file = st.file_uploader("Pilih file CSV", type='csv')


    # Check if there's already an uploaded file in session state
    if 'uploaded_file' in st.session_state:
        st.write("File yang sedang diproses ...")
        st.write(st.session_state.uploaded_file.name)  # Display the name of the uploaded file

    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.uploaded_file = uploaded_file

        # # Store the name of the uploaded file in session state
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = uploaded_file.name


    # Display the "Bersihkan" button only if a file is uploaded
    if 'uploaded_file' in st.session_state:
        # Check if preprocessing has already been done
        if 'preprocessed' not in st.session_state or not st.session_state.preprocessed:
            if st.button("Bersihkan"):
                df_preprocessed = preprocess_data(st.session_state.uploaded_file)
                
                # Store the preprocessed dataframe in session state as a dictionary
                if 'preprocessed_data' not in st.session_state:
                    st.session_state.preprocessed_data = []  # Initialize the list if not present
                # Save both the DataFrame and the filename
                st.session_state.preprocessed_data.append({
                    'data': df_preprocessed,
                    'filename': st.session_state.uploaded_file.name
                })  
                st.success("Preprocessing Selesai.")
        else:
            st.warning("File ini sudah di upload")

    # Display all preprocessed data
    if 'preprocessed_data' in st.session_state:
        for item in st.session_state.preprocessed_data:
            df = item['data']
            filename = item['filename']
            st.write(f"Hasil Preprocessing dari file: {filename}:")
            st.dataframe(df) 



elif page == "Clustering":
    st.header("Analisis Faktor")

    if 'preprocessed_data' in st.session_state:
        file_options = [f"Data set from file: {item['filename']}" for item in st.session_state.preprocessed_data]
        selected_file = st.selectbox("Pilih file yang ingin digunakan untuk klaster:", file_options)
        selected_file_index = file_options.index(selected_file)
        df_selected = st.session_state.preprocessed_data[selected_file_index]['data']

        # Define sentences for each centroid
        centroid_sentences = {
            'kompensasi': "kompensasi gaji uang pendapatan dapat penghasilan hasil intensif gaji sedikit gaji banyak bonus",
            'kepuasan_kerja': "kepuasan puas kerja karir bahagia sedih dedikasi nyaman lembur jam kerja waktu cape capek lelah stres stress",
            'aktualisasi': "aktualisasi aktual pengembangan kembang potensi diri kreatif prestasi jabatan jabat gelar",
            'lingkungan_kerja': "lingkungan rekan kerja suasana dukungan dukung kolaborasi tempat toxic jahat benci suka"
        }

        # Calculate positions in DataFrame for each centroid
        num_rows = len(df_selected)
        centroid_positions = {
            int(num_rows * 0.25): centroid_sentences['kompensasi'],
            int(num_rows * 0.50): centroid_sentences['kepuasan_kerja'],
            int(num_rows * 0.75): centroid_sentences['aktualisasi'],
            int(num_rows * 0.90): centroid_sentences['lingkungan_kerja']
        }

        # Insert sentences into DataFrame at specified positions
        for pos, sentence in centroid_positions.items():
            df_selected.at[pos, 'filtered'] = sentence

        # Ensure all entries in 'filtered' are strings for TF-IDF processing
        df_selected['filtered'] = df_selected['filtered'].apply(lambda x: str(x))

        texts = df_selected['filtered'].astype(str)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Extract initial centroids based on inserted sentences' positions
        initial_centroids = X[list(centroid_positions.keys())].toarray()

        if st.button("Klaster"):
            kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=0)
            kmeans.fit(X)
            df_selected['cluster'] = kmeans.labels_

            # Separate clusters into different DataFrames
            cluster_0 = df_selected[df_selected['cluster'] == 0][['filtered']].reset_index(drop=True)
            cluster_1 = df_selected[df_selected['cluster'] == 1][['filtered']].reset_index(drop=True)
            cluster_2 = df_selected[df_selected['cluster'] == 2][['filtered']].reset_index(drop=True)
            cluster_3 = df_selected[df_selected['cluster'] == 3][['filtered']].reset_index(drop=True)
            # Define custom labels for each cluster
            cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'lingkungan kerja']

            # Display each cluster with descriptive labels
            for i, (label, cluster_df) in enumerate(zip(cluster_labels, [cluster_0, cluster_1, cluster_2, cluster_3])):
                st.write(f"### Faktor {label.capitalize()}")
                st.dataframe(cluster_df[['filtered']])

                # Save clusters with descriptive names in session state
                st.session_state[f'cluster_{label}_df'] = cluster_df

            # Menyimpan setiap cluster ke dalam file .txt di folder 'klaster'
            cluster_0.to_csv('klaster/kompensasi.txt', sep='\t', index=False, header=True)
            cluster_1.to_csv('klaster/kepuasan kerja.txt', sep='\t', index=False, header=True)
            cluster_2.to_csv('klaster/aktualisasi.txt', sep='\t', index=False, header=True)
            cluster_3.to_csv('klaster/lingkungan kerja.txt', sep='\t', index=False, header=True)

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
            cleaned_data_3.to_csv('klaster/lingkungan kerja.txt', sep='\t', index=False, header=True)


elif page == "Sentiment Analysis":
    st.header("Analisis Sentimen Faktor")

    # Load lexicon
    pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

    # Display the button to proceed with sentiment analysis
    if st.button("Proceed to Sentiment Analysis"):

        # Fungsi untuk memuat data cluster dari folder 'klaster'
        def load_cluster_data(cluster_name):
            file_path = f'klaster/{cluster_name}.txt'
            if os.path.exists(file_path):
                return pd.read_csv(file_path, sep='\t')
            else:
                st.error(f"File {cluster_name}.txt tidak ditemukan di folder 'klaster'.")
                return pd.DataFrame()  # Return empty DataFrame if file not found

        # Fungsi untuk menghitung sentimen berdasarkan leksikon
        def analyze_sentiment(text):
            if not isinstance(text, str):
                text = ""
            pos_count = sum(1 for word in text.split() if word in pos_lexicon)
            neg_count = sum(1 for word in text.split() if word in neg_lexicon)
            if pos_count > neg_count:
                return 'Positif', 1  # Positive sentiment and score
            elif neg_count > pos_count:
                return 'Negatif', -1  # Negative sentiment and score
            else:
                return 'Netral', 0  # Neutral sentiment and score

        # Daftar label untuk setiap cluster
        cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'lingkungan kerja']

        # Dictionary untuk menyimpan hasil sentiment dari setiap cluster
        sentiment_dfs = {}

        # Memuat dan memproses setiap cluster
        for label in cluster_labels:
            cluster_df = load_cluster_data(label)  # Memuat DataFrame dari file .txt
            if not cluster_df.empty:
                # Analisis sentimen dan menambahkan label dan skor
                cluster_df[['sentiment_label', 'sentiment_score']] = cluster_df['filtered'].apply(
                    lambda x: pd.Series(analyze_sentiment(x))
                )
                
                # Menyimpan DataFrame untuk cluster saat ini di dictionary
                sentiment_dfs[label] = cluster_df

                # Menampilkan hasil analisis sentimen untuk cluster saat ini
                st.write(f"### Analisis Sentimen Faktor {label.capitalize()}")
                st.write(cluster_df[['filtered', 'sentiment_label', 'sentiment_score']])

                # Menyimpan hasil analisis ke file dengan format yang ditentukan
                output_file_path = f'analisis/{label}.txt'
                cluster_df[['filtered', 'sentiment_label', 'sentiment_score']].to_csv(
                    output_file_path, sep='\t', index=False, header=['analisis', 'sentiment_label', 'sentiment_score']
                )


elif page == "Data Visualization":
    st.header("Visualisasi Data")
    if st.button("Visualisasikan"):
    # Function to load sentiment analysis results from files
        def load_sentiment_data(cluster_name):
            # Directly load the DataFrame from the specified file path
            return pd.read_csv(f'analisis/{cluster_name}.txt', sep='\t')

        # List of cluster labels
        cluster_labels = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'lingkungan kerja']

        # Load and visualize data for each cluster
        for label in cluster_labels:
            cluster_df = load_sentiment_data(label)  # Load DataFrame from file
            if not cluster_df.empty:
                sentiment_counts = cluster_df['sentiment_label'].value_counts()

                # Pie Chart
                st.subheader(f"Visualisasi Analisis Sentimen Faktor {label.capitalize()}")
                st.write(f"Total data pada faktor {label.capitalize()} sebanyak : {len(cluster_df)}")
                # st.write(sentiment_counts)
                fig, ax = plt.subplots()
                colors = ['#ADD8E6', '#87CEFA', '#4682B4']
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

                # Add a brief description of the results
                st.write(f"Faktor {label.capitalize()}, analisis sentimen menunjukkan distribusi sebagai berikut:")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")
