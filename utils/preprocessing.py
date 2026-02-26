"""
utils/preprocessing.py
-----------------------
Fungsi-fungsi preprocessing teks yang dipakai bersama antara
main.py (Streamlit app) dan build.ipynb (notebook pelatihan model).
"""

import re
import json
import string
import nltk
from nltk.tokenize import word_tokenize


def load_lexicon(file_path: str) -> set:
    """Memuat leksikon dari file JSON, mengembalikan set kata."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(json.load(file))


def load_file(file_path: str) -> set:
    """Memuat daftar kata dari file teks (satu kata per baris)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())


def preprocessing_teks(
    text: str,
    slang_dict: dict,
    stopwords: set,
    kamus_indonesia: set,
    stemmer
) -> str:
    """
    Preprocessing lengkap untuk data tweet/teks mentah:
    case folding → hapus karakter khusus → hapus angka → hapus tanda baca
    → rapikan spasi → hapus huruf tunggal → normalisasi slang
    → tokenisasi → stemming → filter stopwords & kamus
    """
    text = text.lower()                                                               # Case folding
    text = re.sub(r"\t|\n|\\u|\\|http[s]?://\S+|[@#][A-Za-z0-9_]+", " ", text)     # Hapus karakter khusus & URL
    text = re.sub(r"\d+", "", text)                                                   # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))                  # Hapus tanda baca
    text = re.sub(r"\s+", ' ', text).strip()                                          # Rapikan spasi ganda
    text = re.sub(r"\b[a-zA-Z]\b", "", text)                                         # Hapus huruf tunggal
    text = ' '.join([slang_dict.get(word, word) for word in text.split()])           # Normalisasi slang
    tokens = word_tokenize(text)                                                      # Tokenisasi
    tokens = [stemmer.stem(word) for word in tokens]                                 # Stemming
    tokens = [
        word for word in tokens
        if word not in stopwords and len(word) > 3 and word in kamus_indonesia       # Filter stopwords & kamus
    ]
    return ' '.join(tokens)


def preprocessing_stopwords(text: str, stopwords: set) -> str:
    """Hapus stopwords saja (dipakai pada tahap klasterisasi)."""
    return ' '.join([word for word in text.split() if word not in stopwords])
