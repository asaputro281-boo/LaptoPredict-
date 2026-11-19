import os
import json
import joblib
import requests
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from .models import RiwayatPrediksi

MODEL_DIR = os.path.join(settings.BASE_DIR, 'prediksi', 'ml_models')
model_reg = joblib.load(os.path.join(MODEL_DIR, 'model_regresi.pkl'))
model_dt = joblib.load(os.path.join(MODEL_DIR, 'model_dt.pkl'))

with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
    METRICS = json.load(f)

# API Key Groq
GROQ_API_KEY = ""


def _find_dataset_csv():
    candidates = [
        os.path.join(settings.BASE_DIR, 'laptop_price.csv'),
        os.path.join(settings.BASE_DIR, 'prediksi', 'laptop_price.csv'),
        os.path.join(MODEL_DIR, 'laptop_price.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_dataset_chart_data():
    fallback = {
        'ram': {
            'labels': ['4 GB', '8 GB', '16 GB', '32 GB', '64 GB'],
            'values': [420.0, 780.0, 1450.0, 2650.0, 4100.0],
        },
        'cpu': {
            'labels': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5'],
            'values': [360.0, 650.0, 1180.0, 1950.0, 3150.0],
        },
    }

    csv_path = _find_dataset_csv()
    if not csv_path:
        return fallback

    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return fallback

    price_col = next(
        (c for c in ['Price_euros', 'price_euros', 'Price_EUR', 'Harga_EUR', 'price'] if c in df.columns),
        None
    )
    ram_col = next(
        (c for c in ['Ram_GB', 'Ram', 'ram', 'RAM_GB', 'ram_gb'] if c in df.columns),
        None
    )
    cpu_col = next(
        (c for c in ['CPU_Tier', 'cpu_tier', 'Cpu_Tier'] if c in df.columns),
        None
    )

    if not price_col:
        return fallback

    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    result = {
        'ram': fallback['ram'],
        'cpu': fallback['cpu'],
    }

    if ram_col:
        ram_series = pd.to_numeric(
            df[ram_col].astype(str).str.replace('GB', '', regex=False).str.strip(),
            errors='coerce'
        )
        ram_df = pd.DataFrame({'ram': ram_series, 'price': df[price_col]}).dropna()
        if not ram_df.empty:
            grp_ram = ram_df.groupby('ram')['price'].mean().sort_index()
            result['ram'] = {
                'labels': [f"{int(k)} GB" if float(k).is_integer() else f"{k} GB" for k in grp_ram.index],
                'values': [round(float(v), 2) for v in grp_ram.values],
            }

    if cpu_col:
        cpu_series = pd.to_numeric(df[cpu_col], errors='coerce')
        cpu_df = pd.DataFrame({'cpu': cpu_series, 'price': df[price_col]}).dropna()
        if not cpu_df.empty:
            grp_cpu = cpu_df.groupby('cpu')['price'].mean().sort_index()
            result['cpu'] = {
                'labels': [f"Tier {int(k)}" if float(k).is_integer() else f"Tier {k}" for k in grp_cpu.index],
                'values': [round(float(v), 2) for v in grp_cpu.values],
            }

    return result


CHART_DATA = get_dataset_chart_data()


def get_active_groq_model(headers):
    models_res = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
    if models_res.status_code != 200:
        return None, f"Gagal mengambil daftar model Groq ({models_res.status_code}): {models_res.text}"

    available_models = [
        m['id'] for m in models_res.json().get('data', [])
        if not any(skip in m['id'].lower() for skip in ['whisper', 'tts', 'guard', 'playai', 'distil'])
    ]
    if not available_models:
        return None, "Tidak ada model chat aktif yang ditemukan di akun Groq kamu."

    chosen_model = available_models[0]
    for m in available_models:
        if any(k in m.lower() for k in ['llama', 'qwen', 'gemma', 'mixtral']):
            chosen_model = m
            break
    return chosen_model, None


def panggil_groq_ai(ram, storage, cpu_tier, harga_eur, harga_idr, kategori):
    api_key = GROQ_API_KEY.strip() if GROQ_API_KEY else ""

    if not api_key:
        return (
            f"Spesifikasi RAM {ram} GB, Storage {storage} GB, dan Prosesor Tier {cpu_tier} "
            f"diprediksi memiliki harga {harga_eur:.2f} EUR (sekitar Rp {harga_idr:,}). "
            f"Laptop ini masuk kategori '{kategori}' dan ideal untuk komputasi harian hingga menengah."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        chosen_model, err_msg = get_active_groq_model(headers)
        if err_msg:
            return err_msg
    except Exception as e:
        return f"Gagal memeriksa daftar model Groq: {e}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = (
        f"Kamu adalah asisten konsultan perangkat keras laptop. Berikan analisis singkat (3-4 kalimat) "
        f"dalam bahasa Indonesia yang jelas untuk calon pembeli laptop dengan spesifikasi:\n"
        f"- RAM: {ram} GB\n- Storage: {storage} GB\n- Tingkat Prosesor: Tier {cpu_tier} dari 5\n"
        f"- Estimasi Harga: {harga_eur:.2f} EUR (Rp {harga_idr:,})\n- Kategori Kelas: {kategori}\n"
        f"Jelaskan kecocokan penggunaannya dan kelayakan harganya."
    )

    payload = {
        "model": chosen_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            return f"Groq API Error ({response.status_code}) pada model {chosen_model}: {response.text}"
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Gagal terhubung ke Groq API: {e}"


def index(request):
    hasil_terbaru = None
    selected = {'ram': 8, 'storage': 512, 'cpu_tier': 3}

    if request.method == 'POST':
        ram = int(request.POST.get('ram', 8))
        storage = int(request.POST.get('storage', 512))
        cpu_tier = int(request.POST.get('cpu_tier', 3))
        selected = {'ram': ram, 'storage': storage, 'cpu_tier': cpu_tier}

        data_input = pd.DataFrame([{'Ram_GB': ram, 'Storage_GB': storage, 'CPU_Tier': cpu_tier}])

        estimasi_eur = round(float(model_reg.predict(data_input)[0]), 2)
        estimasi_idr = int(estimasi_eur * 17500)
        kategori = str(model_dt.predict(data_input)[0])
        penjelasan_ai = panggil_groq_ai(ram, storage, cpu_tier, estimasi_eur, estimasi_idr, kategori)

        hasil_terbaru = RiwayatPrediksi.objects.create(
            ram=ram,
            storage=storage,
            cpu_tier=cpu_tier,
            estimasi_harga_eur=estimasi_eur,
            estimasi_harga_idr=estimasi_idr,
            kategori_harga=kategori,
            analisis_groq=penjelasan_ai
        )

    # Menampilkan seluruh data riwayat dari SQLite
    riwayat = RiwayatPrediksi.objects.all().order_by('-waktu_prediksi')

    context = {
        'hasil': hasil_terbaru,
        'selected': selected,
        'riwayat': riwayat,
        'metrics': METRICS,
        'chart_data_json': json.dumps(CHART_DATA),
    }
    return render(request, 'prediksi/index.html', context)


def hapus_riwayat(request, pk):
    item = get_object_or_404(RiwayatPrediksi, pk=pk)
    item.delete()
    return redirect('index')


def chatbot_api(request):
    if request.method != 'POST':
        return JsonResponse({'reply': 'Metode tidak diizinkan.'}, status=405)

    try:
        body = json.loads(request.body)
        user_message = body.get('message', '').strip()
    except Exception:
        return JsonResponse({'reply': 'Format pesan tidak valid.'}, status=400)

    if not user_message:
        return JsonResponse({'reply': 'Silakan ketik pertanyaan seputar laptop atau aplikasi LaptoPredict AI.'})

    api_key = GROQ_API_KEY.strip() if GROQ_API_KEY else ""
    if not api_key:
        return JsonResponse({'reply': 'API Key Groq belum dikonfigurasi pada server.'})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        chosen_model, err_msg = get_active_groq_model(headers)
        if err_msg:
            return JsonResponse({'reply': err_msg})
    except Exception as e:
        return JsonResponse({'reply': f'Gagal memuat model AI: {e}'})

    system_prompt = (
        "Kamu adalah Asisten Virtual resmi untuk aplikasi web 'LaptoPredict AI'. "
        "TUGAS & BATASAN KETAT:\n"
        "1. Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan:\n"
        "   - Cara menggunakan fitur web LaptoPredict AI (Prediksi harga, Grafik & Metrik Dataset, Riwayat Database SQLite).\n"
        "   - Spesifikasi perangkat keras laptop (RAM, kapasitas Storage SSD/NVMe, Tier Prosesor Intel/Ryzen/Celeron).\n"
        "   - Penjelasan singkat cara kerja model Machine Learning di web ini (Linear Regression untuk estimasi harga EUR/IDR dan Decision Tree untuk klasifikasi Murah, Sedang, Mahal).\n"
        "   - Rekomendasi spesifikasi laptop sesuai kebutuhan (kuliah, desain, gaming, kantor).\n"
        "2. JIKA pengguna bertanya di luar topik tersebut (misalnya politik, resep masakan, curhat, matematika umum, coding bahasa lain, cuaca, selebriti, atau topik umum lainnya), "
        "TOLAK DENGAN SOPAN dalam 1-2 kalimat dengan mengatakan bahwa kamu hanya diprogram untuk membantu seputar LaptoPredict AI dan konsultasi spesifikasi laptop.\n"
        "3. Jawab dengan bahasa Indonesia yang ramah, ringkas, dan langsung ke inti (maksimal 3-4 kalimat)."
    )

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.4,
        "max_tokens": 250
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            return JsonResponse({'reply': f'Error Groq ({response.status_code}): {response.text}'})
        reply_text = response.json()['choices'][0]['message']['content']
        return JsonResponse({'reply': reply_text})
    except Exception as e:
        return JsonResponse({'reply': f'Terjadi gangguan koneksi: {e}'})