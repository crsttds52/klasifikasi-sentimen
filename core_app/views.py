from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import run_scraper, run_cleaning, run_normalization_and_stopwords, run_stemming, extract_package_id, get_loaded_models, prediksi_final
from .utils import load_ml_artefacts # Perlu di-import jika digunakan di luar fungsi
import pandas as pd
import json

DATA_MENTAH_KEY = 'data_mentah_df'
DATA_BERSIH_KEY = 'data_bersih_df'
REVIEW_COLUMN = 'content' 


# --- Halaman 1: HOME / ABOUT ---
def home_view(request):
    """Menampilkan halaman Home/About statis dan tombol mulai."""
    # Menghapus data sesi lama saat user memulai dari awal
    if DATA_MENTAH_KEY in request.session:
        del request.session[DATA_MENTAH_KEY]
    if DATA_BERSIH_KEY in request.session:
        del request.session[DATA_BERSIH_KEY]
        
    return render(request, 'core_app/home.html')


# --- Halaman 2: INPUT LINK & SCRAPING ---
@csrf_exempt
def scrape_view(request):
    """Menerima link, menjalankan scraper, dan menampilkan data mentah."""
    context = {'results': None, 'error_message': None}
    
    if request.method == 'POST':
        url = request.POST.get('input_url')
        count_str = request.POST.get('input_count', '100') # Default 100 ulasan
        
        try:
            count = int(count_str)
            
            # 1. Ekstraksi Package ID
            package_id = extract_package_id(url)
            if not package_id:
                raise ValueError("URL Play Store tidak valid atau Package ID tidak ditemukan.")
            
            # 2. Jalankan Scraping
            # Menghilangkan variabel yang tidak perlu saat memanggil run_scraper
            df, error = run_scraper(package_id, count) 
            
            if error:
                context['error_message'] = f"Gagal Scraping: {error}"
            elif df is not None and not df.empty:
                
                # 3. Simpan Data Mentah ke Session (PENTING!)
                # Konversi DataFrame ke string JSON atau CSV untuk disimpan di sesi
                request.session[DATA_MENTAH_KEY] = df.to_json(orient='split')
                
                # 4. Siapkan Preview (Preview 5 baris untuk ditampilkan)
                result_list = df.head(5).to_dict('records')
                context['results'] = result_list
                context['success_message'] = f"Berhasil mengambil {len(df)} ulasan untuk '{package_id}'."
            
        except ValueError as e:
            context['error_message'] = f"Input tidak valid: {e}"
        except Exception as e:
            context['error_message'] = f"Error saat scraping: {e}"

    return render(request, 'core_app/scrape_result.html', context)

# core_app/views.py - LANJUTAN

# --- Halaman 3: PREPROCESSING & TAMPILAN DATA BERSIH ---
@csrf_exempt
def preprocess_view(request):
    """Menjalankan pipeline pre-processing dan menampilkan preview Data Bersih."""
    context = {'results_mentah': None, 'results_bersih': None, 'comparison_data': None, 'error_message': None}
    
    # 1. Ambil Data Mentah dari Session
    data_mentah_json = request.session.get(DATA_MENTAH_KEY)
    
    if not data_mentah_json:
        # Jika data hilang, redirect kembali ke halaman input/scrape
        return redirect('scrape') 
    
    try:
        # Baca kembali dari JSON ke DataFrame
        df_mentah = pd.read_json(data_mentah_json, orient='split')
        
        # 2. Lakukan Pre-processing Bertahap
        text_series = df_mentah[REVIEW_COLUMN].astype(str)
        
        # NOTE: Memanggil fungsi yang sudah diimplementasikan di utils.py
        df_mentah['Teks_Bersih'] = run_cleaning(text_series) 
        
        # Tokenize, Normalize, Stopwords
        df_mentah['Tokens_Filtered'] = run_normalization_and_stopwords(df_mentah['Teks_Bersih'])
        
        # Stemming dan Join
        df_mentah['Teks_Bersih_Final'] = run_stemming(df_mentah['Tokens_Filtered']) 
        
        # 3. Simpan Data Bersih ke Session (WAJIB)
        df_mentah['text_for_vector'] = df_mentah['Teks_Bersih_Final'] 
        request.session[DATA_BERSIH_KEY] = df_mentah.to_json(orient='split')
        
        # 4. Siapkan Preview (10 baris untuk perbandingan)
        preview_df = df_mentah.head(10)
        # Combine data for easier template rendering
        comparison_data = []
        for idx, row in preview_df.iterrows():
            comparison_data.append({
                'content': row['content'],
                'score': row['score'],
                'teks_bersih': row['Teks_Bersih_Final']
            })
        context['comparison_data'] = comparison_data
        context['results_mentah'] = preview_df[['content', 'score']].to_dict('records')
        context['results_bersih'] = preview_df[['Teks_Bersih_Final']].to_dict('records')
        
    except Exception as e:
        context['error_message'] = f"Error saat Pre-processing: {e}"
        return render(request, 'core_app/preprocess_result.html', context)
    
    # Render halaman preprocessing results (tidak redirect langsung)
    return render(request, 'core_app/preprocess_result.html', context)


# --- Halaman 4: KLASIFIKASI & HASIL AKHIR ---
@csrf_exempt
def results_view(request):
    """Menjalankan TF-IDF, prediksi ganda, dan menampilkan hasil akhir."""
    context = {'results': None, 'stats_xgb': None, 'stats_adaboost': None, 'error_message': None, 'metrics': None}
    
    # Handle download request first (needs to regenerate results)
    if request.method == 'POST' and request.POST.get('download_csv'):
        data_bersih_json = request.session.get(DATA_BERSIH_KEY)
        if not data_bersih_json:
            return redirect('preprocess')
        
        try:
            df_bersih = pd.read_json(data_bersih_json, orient='split')
            texts_to_predict = df_bersih['text_for_vector'].tolist()
            
            TFIDF, XGB_MODEL, ADABOOST_MODEL, _ = get_loaded_models()
            xgb_labels, adaboost_labels = prediksi_final(texts_to_predict, TFIDF, XGB_MODEL, ADABOOST_MODEL)
            
            results_df = df_bersih.copy()
            results_df['Prediksi_XGBoost'] = xgb_labels
            results_df['Prediksi_Adaboost'] = adaboost_labels
            
            # Generate CSV response
            response = HttpResponse(content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = 'attachment; filename="hasil_klasifikasi_sentimen.csv"'
            results_df.to_csv(response, index=False, encoding='utf-8-sig')
            return response
        except Exception as e:
            context['error_message'] = f"Error saat download: {e}"
            return render(request, 'core_app/final_result.html', context)
    
    # 1. Ambil Data Bersih dari Session
    data_bersih_json = request.session.get(DATA_BERSIH_KEY)
    
    if not data_bersih_json:
        return redirect('preprocess') # Jika data hilang, redirect kembali
        
    try:
        df_bersih = pd.read_json(data_bersih_json, orient='split')
        
        # 2. Klasifikasi Sentimen
        texts_to_predict = df_bersih['text_for_vector'].tolist()
        
        # Dapatkan model yang dimuat
        TFIDF, XGB_MODEL, ADABOOST_MODEL, _ = get_loaded_models()
        
        # Panggil fungsi prediksi final
        xgb_labels, adaboost_labels = prediksi_final(texts_to_predict, TFIDF, XGB_MODEL, ADABOOST_MODEL)
        
        # 3. Tambah Hasil ke DataFrame & Hitung Statistik
        results_df = df_bersih.copy()
        results_df['Prediksi_XGBoost'] = xgb_labels
        results_df['Prediksi_Adaboost'] = adaboost_labels
        
        # Hitung statistik Pie Chart untuk XGBoost dan AdaBoost
        total = len(xgb_labels)
        xgb_positif_count = xgb_labels.count('Positif')
        adaboost_positif_count = adaboost_labels.count('Positif')
        
        stats_xgb = {
            'Positif': round((xgb_positif_count / total) * 100, 1),
            'Negatif': round(((total - xgb_positif_count) / total) * 100, 1)
        }
        
        stats_adaboost = {
            'Positif': round((adaboost_positif_count / total) * 100, 1),
            'Negatif': round(((total - adaboost_positif_count) / total) * 100, 1)
        }
        
        # 4. Persiapan Output untuk display
        # Preview 10 baris untuk tabel hasil
        preview_df = results_df[['content', 'Prediksi_XGBoost', 'Prediksi_Adaboost']].head(10)
        context['results'] = preview_df.to_dict('records')
        context['stats_xgb'] = stats_xgb
        context['stats_adaboost'] = stats_adaboost
        context['download_data'] = True
        
        # Tambahkan Metrik Kinerja (Asumsi Metrik sudah dihitung di awal proyek)
        context['metrics'] = {
            'xgb_f1': 0.88, # Ganti dengan F1 Score XGBoost Anda
            'adaboost_f1': 0.85 # Ganti dengan F1 Score AdaBoost Anda
        }

    except Exception as e:
        context['error_message'] = f"Error saat Klasifikasi Final: {e}"
        
    return render(request, 'core_app/final_result.html', context)