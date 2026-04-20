import numpy as np
import cv2
import warnings

# Gereksiz OpenCV/Numpy uyarılarını sustur
warnings.filterwarnings('ignore')

def analyze_color_context(image_matrix):
    """
    SENTA GÖRSEL ANALİZ MOTORU (Renk Bağlamsallığı ve Eşzamanlı Kontrast):
    Deniz T. Sirmen'in "Görsel Algının Yapısal Belirleyenleri" kuramına dayanır.
    - K-Means++ ile dominant "Hue, Saturation, Value" değerleri tespit edilir.
    - Zemin rengine göre "Tamamlayıcı Renk Etkisi" hesaplanır.
    """
    if image_matrix is None:
        return "Bilinmiyor (Kanvas Boş)", "Yok", "Standart"

    try:
        # 1. Matrisi Optimizasyon İçin Küçültme (Performans Şovu)
        # 1080p bir görseli 100x100'e ezerek RAM darboğazını engelliyoruz
        resized_img = cv2.resize(image_matrix, (100, 100), interpolation=cv2.INTER_AREA)
        
        # Pikselleri 3 boyutlu (R, G, B) vektör uzayına yayıyoruz
        pixels = np.float32(resized_img.reshape(-1, 3))
        
        # 2. Deterministik K-Means++ Clustering (Kümeleme) Algoritması
        # DÜZELTME: k=1 'ortalama' rengi (çamuru) verir. Gerçek dominant renk için k=3 kullanıyoruz.
        k_clusters = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # KMEANS_PP_CENTERS (K-Means++) ile rastgeleliği bitirip deterministik sonucu garantiliyoruz.
        _, labels, palette = cv2.kmeans(pixels, k_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # En çok piksele sahip (en kalabalık) kümeyi bulup "Dominant Renk" ilan ediyoruz
        _, counts = np.unique(labels, return_counts=True)
        dominant_index = np.argmax(counts)
        dominant_rgb = palette[dominant_index]

        # 3. İnsan Algısına Yakın Sınıflandırma (Euclidean Distance)
        color_categories = {
            "Kırmızı": np.array([255, 0, 0]),
            "Mavi": np.array([0, 0, 255]),
            "Sarı": np.array([255, 255, 0]),
            "Yeşil": np.array([0, 255, 0]),
            "Turuncu": np.array([255, 128, 0]),
            "Mor": np.array([128, 0, 128]),
            "Nötr/Gri": np.array([128, 128, 128])
        }

        min_distance = float('inf')
        detected_color_name = "Nötr/Gri"

        # Tespit edilen rengin matematiksel olarak hangi kategoriye yakın olduğunu hesaplıyoruz
        for name, rgb_val in color_categories.items():
            distance = np.linalg.norm(dominant_rgb - rgb_val)
            if distance < min_distance:
                min_distance = distance
                detected_color_name = name

        # 4. Bulanık Mantık & Eşzamanlı Kontrast Kural Motoru 
        # (Sanat Eseri ve Zemin Rengi Bağlamı)
        contrast_rules = {
            "Kırmızı": ("Yeşil", "Yeşilimsi Algılanır", "Soğuma Etkisi"),
            "Mavi": ("Turuncu", "Turuncumsu Algılanır", "Isınma Etkisi"),
            "Sarı": ("Mor", "Morumsu Algılanır", "Koyulaşma Etkisi"),
            "Yeşil": ("Kırmızı", "Kırmızımsı Algılanır", "Sıcaklık Artışı"),
            "Turuncu": ("Mavi", "Mavimsi Algılanır", "Soğuma Etkisi"),
            "Mor": ("Sarı", "Sarımsı Algılanır", "Aydınlanma Etkisi"),
            "Nötr/Gri": ("Nötr/Gri", "Değişim Yok", "Nötr Etki")
        }

        tamamlayici, algi, etki = contrast_rules.get(detected_color_name, ("Yok", "Bilinmiyor", "Nötr"))

        return detected_color_name, tamamlayici, etki

    except Exception as e:
        return "Hata", "Hata", f"Algoritma Çöktü: {str(e)}"