from django.db import models

class RiwayatPrediksi(models.Model):
    ram = models.IntegerField()
    storage = models.IntegerField()
    cpu_tier = models.IntegerField()
    estimasi_harga_eur = models.FloatField()
    estimasi_harga_idr = models.BigIntegerField()
    kategori_harga = models.CharField(max_length=20)
    analisis_groq = models.TextField()
    waktu_prediksi = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RAM {self.ram}GB | Storage {self.storage}GB - {self.kategori_harga}"