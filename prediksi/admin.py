from django.contrib import admin
from .models import RiwayatPrediksi

@admin.register(RiwayatPrediksi)
class RiwayatPrediksiAdmin(admin.ModelAdmin):
    list_display = ('id', 'ram', 'storage', 'cpu_tier', 'estimasi_harga_eur', 'kategori_harga', 'waktu_prediksi')
    list_filter = ('kategori_harga', 'ram')