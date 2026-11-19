# LaptoPredict AI

Aplikasi web berbasis **Django** dan **Machine Learning** untuk memperkirakan harga laptop, mengklasifikasikan kelas harga (**Murah**, **Sedang**, **Mahal**), serta memberikan analisis rekomendasi menggunakan **Groq AI**.

## Fitur Utama

* **Prediksi Harga & Klasifikasi**: Hitung estimasi harga laptop (EUR dan Rupiah) beserta kategori kelas harga berdasarkan spesifikasi RAM, kapasitas penyimpanan (*storage*), dan kelas prosesor (*CPU Tier*).
* **Analisis Otomatis Groq AI**: Dapatkan ulasan singkat mengenai kecocokan penggunaan dan kelayakan harga spesifikasi yang dipilih.
* **Asisten Chatbot**: Tanya jawab seputar fitur aplikasi, spesifikasi laptop, dan cara kerja model *machine learning* di web ini.
* **Grafik & Evaluasi Model**: Lihat metrik evaluasi model (akurasi *Decision Tree*, skor *R²*, dan *MAE*) serta grafik perbandingan rata-rata harga dari dataset.
* **Riwayat Tersimpan (SQLite)**: Seluruh hasil prediksi dan analisis tersimpan otomatis di `db.sqlite3` serta dapat dilihat atau dihapus kapan saja.

## Teknologi yang Digunakan

* **Backend & Web Framework**: Python, Django
* **Machine Learning & Data**: Pandas, Scikit-learn, Joblib
* **AI / LLM**: Groq API
* **Database**: SQLite (`db.sqlite3`)
* **Frontend**: HTML, CSS, Bootstrap, Chart.js

## Struktur Folder

```text
LaptoPredict-/
├── config/             # Konfigurasi proyek Django
├── prediksi/           # Aplikasi utama (views, models, templates, ml_models)
├── db.sqlite3          # Basis data riwayat prediksi
├── manage.py           # Perintah manajemen Django
├── requirements.txt    # Daftar pustaka Python yang dibutuhkan
└── train_model.py      # Skrip pelatihan model Machine Learning
