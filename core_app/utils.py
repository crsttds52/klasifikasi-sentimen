import os
import joblib
import pandas as pd
import re
import string
import numpy as np
from io import StringIO
from scipy import sparse
from google_play_scraper import Sort, reviews 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

def load_ml_artefacts():
    try:
        tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'final_xgb_model.pkl'))
        adaboost_model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_final.pkl'))
        kamus_path = os.path.join(MODEL_DIR, 'kamusalay.csv')
        
        try:
             kamus_df = pd.read_csv(kamus_path, encoding='utf-8') 
        except UnicodeDecodeError:
             kamus_df = pd.read_csv(kamus_path, encoding='latin-1')
             
        kamusalay_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))
        
        print("SEMUA MODEL BERHASIL DIMUAT")
        return tfidf, xgb_model, adaboost_model, kamusalay_dict
        
    except FileNotFoundError as e:
        print(f"\n!!! FILE TIDAK DITEMUKAN. Cek file di '{MODEL_DIR}'. Error: {e}\n") 
        return None, None, None, None
    except Exception as e:
        print(f"\n!!! FATAL ERROR LAINNYA SAAT STARTUP: {e}\n")
        return None, None, None, None

TFIDF, XGB_MODEL, ADABOOST_MODEL, KAMUSALAY_DICT = load_ml_artefacts()

def get_loaded_models():
    """Mengembalikan artefak ML yang dimuat."""
    return TFIDF, XGB_MODEL, ADABOOST_MODEL, KAMUSALAY_DICT

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')


def extract_package_id(url):
    """Mengambil package ID (com.nama.aplikasi) dari URL Play Store."""
    # Pola Regex mencari 'id=' diikuti oleh karakter non-& atau non-spasi
    match = re.search(r'id=([^& ]+)', url)
    if match:
        return match.group(1)
    return None

def run_scraper(package_id, count):
    """Menjalankan scraper untuk mengambil ulasan dari Play Store."""
    try:
        # Panggil scraper dengan Sort.NEWEST (sesuai keputusan Anda)
        result, _ = reviews(
            package_id,
            lang='id', # Bahasa Indonesia
            country='id',
            sort=Sort.NEWEST,
            count=count 
        )
        # Konversi hasil ke DataFrame
        df = pd.DataFrame(result)
        # Pilih kolom yang relevan
        df = df[['content', 'score']] 
        return df, None # Mengembalikan DataFrame dan tanpa error
    
    except Exception as e:
        return None, f"Gagal Scraping: Pastikan Package ID '{package_id}' benar atau jumlah data tidak terlalu banyak. Error: {e}"

#!!!!!! ... (Lanjutkan dengan Pre-processing yang Dipecah)!!!!!!!!!

def run_cleaning(text_series):
    """Applies case folding, removes links, numbers, and punctuation."""
    
    def clean_text_single(text):
        if pd.isna(text): # Handle NaN values which can come from the DataFrame
            return ""
        text = str(text).lower() # Case Folding
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Hapus link
        text = re.sub(r'[0-9]', '', text) # Hapus angka
        text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
        text = re.sub(r'[^\w\s]', '', text) # Hapus sisa karakter non-alfanumerik
        text = re.sub(r'\s\s+', ' ', text).strip() # Hapus spasi ganda
        return text

    # Apply cleaning function to the entire Series (column)
    return text_series.apply(clean_text_single)

def run_normalization_and_stopwords(text_series):
    """Tokenizes, applies slang normalization, and removes stopwords."""
    
    # Get necessary dictionaries/tools
    _, _, _, KAMUSALAY_DICT = get_loaded_models()
    
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(['yg', 'utk', 'aja', 'sih', 'dah', 'dong', 'kok', 'trs', 'dll']) 

    def process_token(text):
        if not text:
            return ""
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Normalization (using the globally loaded dictionary)
        # Ensure all tokens are lowercase for consistency
        normalized_tokens = []
        for word in tokens:
            word_lower = word.lower()
            # Get normalized word from dictionary, or use lowercase original
            normalized_word = KAMUSALAY_DICT.get(word_lower, word_lower)
            normalized_tokens.append(normalized_word)
        
        # Stopword Removal (all tokens are now lowercase)
        filtered_tokens = [word for word in normalized_tokens if word not in list_stopwords]
        
        # Join back temporarily for the next step or return tokens list
        return filtered_tokens 

    # Apply the token processing function
    return text_series.apply(process_token)


def run_stemming(token_series):
    """Applies stemming and joins tokens into the final string."""
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def perform_stemming_and_join(tokens):
        if not tokens:
            return ""
            
        # Stemming
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        # Join back into a single string for TF-IDF
        return " ".join(stemmed_tokens)

    # Apply the stemming function
    return token_series.apply(perform_stemming_and_join)

def prediksi_final(text_list, tfidf, xgb_model, adaboost_model):
    """
    Menjalankan Vektorisasi dan Prediksi Ganda (XGBoost & AdaBoost).
    Asumsi: text_list sudah bersih dan siap divetorisasi (sudah di-stem).
    """
    # Handle empty strings - replace with a space to ensure consistent vectorization
    # Empty strings can cause dimension mismatches with some vectorizers
    text_list_processed = [text if text and text.strip() else " " for text in text_list]
    
    # 1. Vektorisasi
    # Menggunakan objek tfidf yang diteruskan dari View
    text_vector = tfidf.transform(text_list_processed)
    
    # Get expected and actual feature counts
    expected_features_xgb = xgb_model.n_features_in_ if hasattr(xgb_model, 'n_features_in_') else None
    expected_features_ada = adaboost_model.n_features_in_ if hasattr(adaboost_model, 'n_features_in_') else None
    actual_features = text_vector.shape[1]
    
    # Get TF-IDF vocabulary size
    vocab_size = len(tfidf.vocabulary_) if hasattr(tfidf, 'vocabulary_') else actual_features
    max_features = getattr(tfidf, 'max_features', None)
    
    # Check for dimension mismatch
    if expected_features_xgb and actual_features != expected_features_xgb:
        # Try to diagnose the issue
        error_msg = (
            f"Feature dimension mismatch detected:\n"
            f"  - XGBoost model expects: {expected_features_xgb} features\n"
            f"  - TF-IDF vectorizer produced: {actual_features} features\n"
            f"  - TF-IDF vocabulary size: {vocab_size}\n"
            f"  - TF-IDF max_features setting: {max_features}\n"
        )
        
        # If difference is exactly 1, it might be a padding/offset issue
        if expected_features_xgb - actual_features == 1:
            # Try adding a zero column (workaround for potential offset issue)
            # This is a hack, but might work if there's a padding feature
            try:
                # Add a column of zeros at the end
                if sparse.issparse(text_vector):
                    # For sparse matrices, add a zero column
                    zero_col = sparse.csr_matrix((text_vector.shape[0], 1))
                    text_vector = sparse.hstack([text_vector, zero_col])
                else:
                    # For dense matrices
                    zero_col = np.zeros((text_vector.shape[0], 1))
                    text_vector = np.hstack([text_vector, zero_col])
                
                # Update actual_features
                actual_features = text_vector.shape[1]
                if actual_features == expected_features_xgb:
                    # Workaround successful, continue
                    pass
                else:
                    raise ValueError(error_msg + "Attempted padding workaround failed.")
            except Exception as e:
                raise ValueError(error_msg + f"Padding workaround also failed: {str(e)}")
        else:
            raise ValueError(
                error_msg + 
                "This usually means the preprocessing pipeline doesn't match the training pipeline. "
                "Please ensure tokenization, normalization, stopword removal, and stemming "
                "are identical to what was used during model training."
            )
    
    # 2. Prediksi Ganda
    xgb_pred = xgb_model.predict(text_vector)
    adaboost_pred = adaboost_model.predict(text_vector)
    
    # 3. Konversi label numerik (0/1) ke Label String
    label_map = {0: 'Negatif', 1: 'Positif'}
    
    # Gunakan list comprehension untuk konversi
    xgb_labels = [label_map[p] for p in xgb_pred]
    adaboost_labels = [label_map[p] for p in adaboost_pred]
    
    # Mengembalikan list label string
    return xgb_labels, adaboost_labels