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

import nltk
nltk.download('punkt_tab')

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

# Inisialisasi session state untuk menyimpan visualisasi
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []

# Fungsi untuk menambahkan visualisasi ke session state
def add_visualization(fig):
    st.session_state.visualizations.append(fig)

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
    corpus_df.to_csv('hasil.txt', index=False, header=False)
    return corpus_df

# Clustering function with custom centroid input and dataset preview
def cluster_data_with_custom_centroids():
    # Load data
    try:
        df = pd.read_csv('hasil.txt', header=None, names=['preprocessing'])
    except FileNotFoundError:
        st.error("Preprocessed file not found. Please preprocess the data first on the 'Text Preprocessing' page.")
        return
    
    st.write("### Tampilan Preprocessing Data")
    st.dataframe(df)  # Display the first few rows of the preprocessed data

    texts = df['preprocessing'].astype(str)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Allow user to input custom centroids (4 indices)
    st.write("### Input Index Titik Centroid")
    centroid_index_1 = st.number_input("Index for Centroid 1", min_value=0, max_value=len(texts) - 1, step=1, value=0)
    centroid_index_2 = st.number_input("Index for Centroid 2", min_value=0, max_value=len(texts) - 1, step=1, value=1)
    centroid_index_3 = st.number_input("Index for Centroid 3", min_value=0, max_value=len(texts) - 1, step=1, value=2)
    centroid_index_4 = st.number_input("Index for Centroid 4", min_value=0, max_value=len(texts) - 1, step=1, value=3)
    
    if st.button("Klaster"):
        # Extract custom centroids from input indices
        initial_centroids = X[[centroid_index_1, centroid_index_2, centroid_index_3, centroid_index_4]].toarray()
        
        # Apply KMeans with custom centroids
        kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=0)
        kmeans.fit(X)
        df['cluster'] = kmeans.labels_
        
        # Display results
        st.write("### Clustering Results")
        st.dataframe(df[['preprocessing', 'cluster']])
        
        # Word Cloud for each cluster
        st.write("## Word Clouds for Each Cluster")
        for cluster_num in range(4):
            cluster_texts = df[df['cluster'] == cluster_num]['preprocessing'].str.cat(sep=' ')
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(cluster_texts)
            plt.figure(figsize=(5, 3))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Cluster {cluster_num + 1}')
            st.pyplot(plt)

# Main application logic
if page == "Preprocessing":
    st.header("Text Preprocessing")
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')


    # Check if there's already an uploaded file in session state
    if 'uploaded_file' in st.session_state:
        st.write("File terakhir di upload:")
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
                st.success("Preprocessing completed.")
        else:
            st.warning("File ini sudah di upload")

    # Display all preprocessed data
    if 'preprocessed_data' in st.session_state:
        st.write("### Preprocessed Data")
        for item in st.session_state.preprocessed_data:
            df = item['data']
            filename = item['filename']
            st.write(f"Data set from file: {filename}:")
            st.dataframe(df)  # Display the preprocessed data

# elif page == "Clustering":
#     st.header("K-Means Clustering")
#     cluster_data_with_custom_centroids()

elif page == "Clustering":
    st.header("K-Means Clustering")

    # Pastikan data preprocessed sudah ada
    if 'preprocessed_data' in st.session_state:
        # Tampilkan dropdown untuk memilih file yang ingin digunakan
        file_options = [f"Data set from file: {item['filename']}" for item in st.session_state.preprocessed_data]
        selected_file = st.selectbox("Pilih file yang ingin digunakan untuk clustering:", file_options)

        # Dapatkan index dari file yang dipilih
        selected_file_index = file_options.index(selected_file)

        # Ambil dataframe dari file yang dipilih
        df_selected = st.session_state.preprocessed_data[selected_file_index]['data']
        
        st.write(f"### Tampilan Data Preprocessing dari file: {st.session_state.preprocessed_data[selected_file_index]['filename']}")
        st.dataframe(df_selected)  # Display the selected preprocessed data

        # Cluster data yang dipilih
        texts = df_selected['full_text'].astype(str)

        # Vectorize text
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Input untuk titik centroid
        st.write("### Input Index Titik Centroid")
        centroid_index_1 = st.number_input("Index for Centroid 1", min_value=0, max_value=len(texts) - 1, step=1, value=0)
        centroid_index_2 = st.number_input("Index for Centroid 2", min_value=0, max_value=len(texts) - 1, step=1, value=1)
        centroid_index_3 = st.number_input("Index for Centroid 3", min_value=0, max_value=len(texts) - 1, step=1, value=2)
        centroid_index_4 = st.number_input("Index for Centroid 4", min_value=0, max_value=len(texts) - 1, step=1, value=3)

        if st.button("Klaster"):
            # Extract custom centroids from input indices
            initial_centroids = X[[centroid_index_1, centroid_index_2, centroid_index_3, centroid_index_4]].toarray()

            # Apply KMeans with custom centroids
            kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=0)
            kmeans.fit(X)
            df_selected['cluster'] = kmeans.labels_

            # Display results
            st.write("### Clustering Results")
            st.dataframe(df_selected[['full_text', 'cluster']])

            # Word Cloud for each cluster
            st.write("## Word Clouds for Each Cluster")
            for cluster_num in range(4):
                cluster_texts = df_selected[df_selected['cluster'] == cluster_num]['full_text'].str.cat(sep=' ')
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(cluster_texts)
                plt.figure(figsize=(5, 3))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Cluster {cluster_num + 1}')
                st.pyplot(plt)
                # Simpan visualisasi
                add_visualization(plt.gcf())

    else:
        st.warning("Belum ada data yang di-preprocess untuk ditampilkan di sini.")

# elif page == "Sentiment Analysis":
#     st.header("Sentiment Analysis")
    # Implement sentiment analysis logic here
    # Page for Sentiment Analysis
    
elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")

    # Load lexicon
    pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

    if 'preprocessed_data' in st.session_state:
        # Tampilkan dropdown untuk memilih file yang ingin dianalisis
        file_options = [f"Data set from file: {item['filename']}" for item in st.session_state.preprocessed_data]
        selected_file = st.selectbox("Pilih file yang ingin dianalisis sentimennya:", file_options)

        # Dapatkan index dari file yang dipilih
        selected_file_index = file_options.index(selected_file)

        # Ambil dataframe dari file yang dipilih
        df_selected = st.session_state.preprocessed_data[selected_file_index]['data']
        
        st.write(f"### Tampilan Data Preprocessing dari file: {st.session_state.preprocessed_data[selected_file_index]['filename']}")
        st.dataframe(df_selected)  # Display the selected preprocessed data

        # Analisis sentimen untuk setiap teks
        df_selected['sentiment'], df_selected['pos_count'], df_selected['neg_count'] = zip(*df_selected['full_text'].apply(lambda x: sentiment_analysis(x, pos_lexicon, neg_lexicon)))

        # Tampilkan hasil analisis sentimen
        st.write("### Hasil Analisis Sentimen")
        st.dataframe(df_selected[['full_text', 'sentiment', 'pos_count', 'neg_count']])
        
        # Tampilkan analisis sentimen berdasarkan cluster
        st.write("### Analisis Sentimen Berdasarkan Cluster")
        sentiment_counts = df_selected.groupby('cluster')['sentiment'].value_counts().unstack().fillna(0)
        st.bar_chart(sentiment_counts)
        
        # Simpan visualisasi bar chart
        add_visualization(plt.gcf()) 

    else:
        st.warning("Belum ada data yang di-preprocess untuk ditampilkan di sini.")

elif page == "Data Visualization":
    st.header("Data Visualization")
    # Implement visualization logic here
    
    # Menampilkan semua visualisasi yang disimpan 
    for i, fig in enumerate(st.session_state.visualizations):
        st.write(f'### Visualisasi {i+1}')
        st.pyplot(fig)
