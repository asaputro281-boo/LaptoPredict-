from django.contrib import admin
from django.urls import path
from prediksi.views import index, hapus_riwayat, chatbot_api

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('hapus/<int:pk>/', hapus_riwayat, name='hapus_riwayat'),
    path('chat/', chatbot_api, name='chatbot_api'),
]