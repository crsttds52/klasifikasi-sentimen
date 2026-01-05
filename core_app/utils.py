import os
import joblib
import pandas as pd
import re
import string
from io import StringIO
from google_play_scraper import Sort, reviews 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

def load_ml_artefacts():
    try:
        xgb_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer_xgboost.pkl'))
        adaboost_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer_adaboost.pkl'))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'final_xgb_model.pkl'))
        adaboost_model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_final.pkl'))
        kamus_path = os.path.join(MODEL_DIR, 'kamusalay.csv')
        
        try:
             kamus_df = pd.read_csv(kamus_path, encoding='utf-8') 
        except UnicodeDecodeError:
             kamus_df = pd.read_csv(kamus_path, encoding='latin-1')
             
        kamusalay_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))
        
        print("SEMUA MODEL BERHASIL DIMUAT")
        return xgb_vectorizer, adaboost_vectorizer, xgb_model, adaboost_model, kamusalay_dict
        
    except FileNotFoundError as e:
        print(f"\n!!! FILE TIDAK DITEMUKAN. Cek file di '{MODEL_DIR}'. Error: {e}\n") 
        return None, None, None, None, None
    except Exception as e:
        print(f"\n!!! FATAL ERROR LAINNYA SAAT STARTUP: {e}\n")
        return None, None, None, None, None

XGB_VECTORIZER, ADABOOST_VECTORIZER, XGB_MODEL, ADABOOST_MODEL, KAMUSALAY_DICT = load_ml_artefacts()

def get_loaded_models():
    """Mengembalikan artefak ML yang dimuat."""
    return XGB_VECTORIZER, ADABOOST_VECTORIZER, XGB_MODEL, ADABOOST_MODEL, KAMUSALAY_DICT

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
    match = re.search(r'id=([^& ]+)', url)
    if match:
        return match.group(1)
    return None

def run_scraper(package_id, count):
    """Menjalankan scraper untuk mengambil ulasan dari Play Store."""
    try:
        result, _ = reviews(
            package_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=count 
        )
        if not result:
            return None, (
                f"Tidak ada ulasan yang berhasil diambil untuk package '{package_id}'. "
                "Coba kurangi jumlah data atau pastikan aplikasi memiliki ulasan."
            )

        df = pd.DataFrame(result)

        required_columns = {'content', 'score'}
        if not required_columns.issubset(df.columns):
            missing = required_columns.difference(df.columns)
            return None, (
                "Struktur data dari Play Store berubah sehingga kolom yang dibutuhkan tidak tersedia. "
                f"Kolom hilang: {', '.join(missing)}"
            )

        df = df[['content', 'score']]
        if df.empty:
            return None, (
                f"Data yang diambil kosong untuk package '{package_id}'. "
                "Pastikan aplikasi memiliki ulasan publik."
            )

        return df, None
    
    except Exception as e:
        return None, f"Gagal Scraping: Pastikan Package ID '{package_id}' benar atau jumlah data tidak terlalu banyak. Error: {e}"

def run_cleaning(text_series):
    
    def clean_text_single(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[0-9]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s\s+', ' ', text).strip()
        return text

    return text_series.apply(clean_text_single)

def run_normalization_and_stopwords(text_series):
    
    *_, KAMUSALAY_DICT = get_loaded_models()
    
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(['yg', 'utk', 'aja', 'sih', 'dah', 'dong', 'kok', 'trs', 'dll']) 

    def process_token(text):
        if not text:
            return ""
        
        tokens = word_tokenize(text)
        
        normalized_tokens = []
        for word in tokens:
            word_lower = word.lower()
            normalized_word = KAMUSALAY_DICT.get(word_lower, word_lower)
            normalized_tokens.append(normalized_word)
        
        filtered_tokens = [word for word in normalized_tokens if word not in list_stopwords]
        
        return filtered_tokens 

    return text_series.apply(process_token)


def run_stemming(token_series):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def perform_stemming_and_join(tokens):
        if not tokens:
            return ""
            
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        return " ".join(stemmed_tokens)

    return token_series.apply(perform_stemming_and_join)

def prediksi_final(text_list, xgb_vectorizer, adaboost_vectorizer, xgb_model, adaboost_model):
    """
    Menjalankan Vektorisasi dan Prediksi Ganda (XGBoost & AdaBoost).
    Asumsi: text_list sudah bersih dan siap divetorisasi (sudah di-stem).
    """
    text_list_processed = [text if text and text.strip() else " " for text in text_list]
    
    xgb_vector = xgb_vectorizer.transform(text_list_processed)
    adaboost_vector = adaboost_vectorizer.transform(text_list_processed)

    expected_features_xgb = xgb_model.n_features_in_ if hasattr(xgb_model, 'n_features_in_') else None
    expected_features_ada = adaboost_model.n_features_in_ if hasattr(adaboost_model, 'n_features_in_') else None

    if expected_features_xgb and xgb_vector.shape[1] != expected_features_xgb:
        raise ValueError(
            "Dimensi fitur tidak sesuai (XGBoost):\n"
            f"  - Model XGBoost mengharapkan: {expected_features_xgb} fitur\n"
            f"  - Vectorizer XGBoost menghasilkan: {xgb_vector.shape[1]} fitur\n"
            "Vectorizer yang dimuat tidak sesuai dengan model."
        )

    if expected_features_ada and adaboost_vector.shape[1] != expected_features_ada:
        raise ValueError(
            "Dimensi fitur tidak sesuai (AdaBoost):\n"
            f"  - Model AdaBoost mengharapkan: {expected_features_ada} fitur\n"
            f"  - Vectorizer AdaBoost menghasilkan: {adaboost_vector.shape[1]} fitur\n"
            "Vectorizer yang dimuat tidak sesuai dengan model."
        )

    xgb_pred = xgb_model.predict(xgb_vector)
    adaboost_pred = adaboost_model.predict(adaboost_vector)
    
    label_map = {0: 'Negatif', 1: 'Positif'}
    
    xgb_labels = [label_map[p] for p in xgb_pred]
    adaboost_labels = [label_map[p] for p in adaboost_pred]
    
    return xgb_labels, adaboost_labels