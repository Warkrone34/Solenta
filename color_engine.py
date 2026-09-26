import numpy as np
import cv2
import warnings

# Görüntü işleme kütüphanelerinin (OpenCV) derleme sürümü uyumsuzluklarından 
# kaynaklanabilecek uyarıları, arayüz stabilitesini korumak için işletim sistemi seviyesinde bastırır.
warnings.filterwarnings('ignore')

def analyze_color_context(image_matrix):
    """
    Neden Eklendi: Görselin yapısal zemin rengini tespit ederek, üzerine işlenecek 
    sanat eserinin renk paleti ile nasıl bir 'Eşzamanlı Kontrast' (Simultaneous Contrast) 
    yaratacağını matematiksel olarak öngörmek için kurgulanmıştır.
    """
    # Girdi matrisi boşsa, algoritmanın çökmesini engellemek için güvenli dönüş (Fallback) yapar.
    if image_matrix is None:
        return "Bilinmiyor (Kanvas Boş)", "Yok", "Standart"

    try:
        # =====================================================================
        # [1. BOYUT İNDİRGEME VE OPTİMİZASYON (Downsampling)]
        # =====================================================================
        # Neden Eklendi: Yüksek çözünürlüklü (örn. 1080p) bir matrisin milyonlarca pikselini 
        # tek tek kümelemek O(n^2) karmaşıklığına yol açar. Görüntü INTER_AREA interpolasyonu 
        # ile 100x100 boyutuna sıkıştırılarak veri seti küçültülür ve RAM darboğazı engellenir.
        resized_img = cv2.resize(image_matrix, (100, 100), interpolation=cv2.INTER_AREA)
        
        # 3 boyutlu tensörü (Yükseklik, Genişlik, Renk Kanalı), K-Means algoritmasının 
        # okuyabileceği 2 boyutlu (N adet piksel, 3 adet RGB değeri) bir vektör uzayına düzleştirir (Flattening).
        pixels = np.float32(resized_img.reshape(-1, 3))
        
        # =====================================================================
        # [2. GÖZETİMSİZ ÖĞRENME: K-MEANS++ CLUSTERING]
        # =====================================================================
        # Neden k=3: K=1 kullanılması durumunda algoritma tüm renklerin aritmetik ortalamasını 
        # (çamurlaşmış bir renk) döner. Gerçek istatistiksel mod (dominant renk) değerini 
        # izole etmek için uzay 3 ayrı kümeye bölünmüştür.
        k_clusters = 3
        
        # Algoritmanın durma kriteri: Ya 100 iterasyon tamamlanacak ya da kümeler arası merkez kayması 0.2'nin altına düşecek.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # K-Means++ (KMEANS_PP_CENTERS) başlatma algoritması kullanılarak, rastgele başlangıç noktalarından
        # kaynaklanan farklı sonuç (Non-deterministic) problemi çözülür. Her çalışmada aynı matematiksel sonucu vermesi garantilenir.
        _, labels, palette = cv2.kmeans(pixels, k_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Her bir kümeye atanan piksel sayısını hesaplayarak (Frekans), en yüksek yoğunluğa sahip 
        # kümenin indeksini (argmax) tespit eder ve ana rengin (dominant_rgb) vektörel karşılığını çeker.
        _, counts = np.unique(labels, return_counts=True)
        dominant_index = np.argmax(counts)
        dominant_rgb = palette[dominant_index]

        # =====================================================================
        # [3. SEMANTİK RENK HARİTALAMASI (L2 Norm / Euclidean Distance)]
        # =====================================================================
        # Neden Eklendi: Makinenin bulduğu vektörel RGB (Örn: 240, 10, 15) değerini,
        # insanın algılayabileceği anlamsal (Semantic) bir metin ifadeye (Örn: "Kırmızı") dönüştürmek için.
        color_categories = {
            "Kırmızı": np.array([255, 0, 0]),
            "Mavi": np.array([0, 0, 255]),
            "Sarı": np.array([255, 255, 0]),
            "Yeşil": np.array([0, 255, 0]),
            "Turuncu": np.array([255, 128, 0]),
            "Mor": np.array([128, 0, 128]),
            "Nötr/Gri": np.array([128, 128, 128])
        }

        # L2 Norm (Öklid Uzaklığı) algoritması için minimum mesafe eşiğini matematiksel sonsuzluk (inf) ile başlatır.
        min_distance = float('inf')
        detected_color_name = "Nötr/Gri"

        # Tespit edilen dominant rengin, sözlükteki hangi saf renge uzaysal olarak en yakın olduğunu bulur.
        for name, rgb_val in color_categories.items():
            distance = np.linalg.norm(dominant_rgb - rgb_val)
            if distance < min_distance:
                min_distance = distance
                detected_color_name = name

        # =====================================================================
        # [4. EŞZAMANLI KONTRAST (SIMULTANEOUS CONTRAST) KURAL MOTORU]
        # =====================================================================
        # Gestalt ve Bilişsel Algı kuramlarına dayanarak, hedeflenen dominant rengin
        # insan beyninde (Görsel Korteks) yaratacağı tamamlayıcı renk illüzyonunu hesaplar.
        contrast_rules = {
            "Kırmızı": ("Yeşil", "Yeşilimsi Algılanır", "Soğuma Etkisi"),
            "Mavi": ("Turuncu", "Turuncumsu Algılanır", "Isınma Etkisi"),
            "Sarı": ("Mor", "Morumsu Algılanır", "Koyulaşma Etkisi"),
            "Yeşil": ("Kırmızı", "Kırmızımsı Algılanır", "Sıcaklık Artışı"),
            "Turuncu": ("Mavi", "Mavimsi Algılanır", "Soğuma Etkisi"),
            "Mor": ("Sarı", "Sarımsı Algılanır", "Aydınlanma Etkisi"),
            "Nötr/Gri": ("Nötr/Gri", "Değişim Yok", "Nötr Etki")
        }

        # Bulunan ana renge karşılık gelen tamamlayıcı algıyı statik sözlükten çeker, eşleşme yoksa (Fallback) nötr döner.
        tamamlayici, algi, etki = contrast_rules.get(detected_color_name, ("Yok", "Bilinmiyor", "Nötr"))

        return detected_color_name, tamamlayici, etki

    except Exception as e:
        # İşlemci limitasyonu veya matris kilitlenmesi durumunda arayüzü çökertmeden log mesajı fırlatır.
        return "Hata", "Hata", f"Matris Analizi Basarisiz: {str(e)}"