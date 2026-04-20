import cv2
import numpy as np

def apply_timbre_driven_convolution(image_matrix, timbre, user_sharpness):
    """
    [SİNYAL SÜRÜCÜLÜ DİNAMİK KONVOLÜSYON MATRİSİ]
    Müziğin spektral ağırlık merkezini (Timbre) kullanarak piksellerin uzaysal (spatial) 
    türevini hesaplayan dinamik bir kernel (matris) üretir.
    
    Hocaya anlatılacak kısım: "Matrisin merkezindeki K değeri sabit değildir, 
    ses sinyalindeki yüksek frekanslı gürültü oranına göre saniye saniye hesaplanır."
    """
    if image_matrix is None:
        return None, ""
        
    # 1. Timbre (Tını) Sinyalinin Normalizasyonu
    # İnsan kulağının ve enstrümanların ortalama tını aralığı
    min_timbre, max_timbre = 200.0, 4000.0
    normalized_timbre = np.clip((timbre - min_timbre) / (max_timbre - min_timbre), 0.0, 1.0)
    
    # 2. Kernel (Matris) Ağırlık Hesaplaması
    # Müzik ne kadar "yırtıcı" ise (high timbre), w değeri o kadar artar.
    w = normalized_timbre * float(user_sharpness)
    
    # Eğer kullanıcı slider'ı 0 yaptıysa veya ses aşırı yumuşaksa matrisi pas geç
    if w == 0.0:
        return image_matrix, "🧮 Dinamik Konvolüsyon: Devre Dışı (K=1.0)"

    # 3. Dinamik Konvolüsyon Matrisi (Laplacian / Unsharp Mask Hibriti)
    # Merkez pikselin gücü (1 + 4w), komşuların negatif ağırlığı (-w)
    kernel = np.array([
        [ 0, -w,  0],
        [-w, 1 + 4*w, -w],
        [ 0, -w,  0]
    ], dtype=np.float32)
    
    # 4. Uzaysal Filtreleme (Spatial Filtering - O(N^2) Matris Çarpımı)
    filtered_image = cv2.filter2D(image_matrix, -1, kernel)
    
    # Renk sapmalarını engellemek için Kırpma (Clipping)
    final_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    log_msg = f"🧮 Dinamik Konvolüsyon Matrisi: K (Merkez) = {1 + 4*w:.2f} | Tını Çarpanı = {normalized_timbre:.2f}"
    
    return final_image, log_msg


def apply_kmeans_color_quantization(image_matrix, k_slider):
    """
    [VEKTÖREL RENK KÜMELEME - K-MEANS ALGORİTMASI]
    Görseldeki milyonlarca pikseli 3D RGB uzayında vektörler olarak alır,
    aralarındaki Öklid uzaklıklarını hesaplar ve tüm resmi
    kullanıcının belirlediği K (Cluster) sayısına indirger.
    
    Görsel etki: Devasa. Çizgi roman / Retro Poster estetiği yaratır.
    """
    if image_matrix is None:
        return None, ""
        
    k = int(k_slider)
    
    # K=0 veya çok yüksekse işlemi iptal et (Orijinal kalsın)
    # Slider 65'te duracak, 65 demek limitsiz renk demek.
    if k <= 1 or k >= 64:
        return image_matrix, f"🎨 Renk Kuantizasyonu: Devre Dışı (Sınırsız Renk)"

    # 1. Pikselleri 3 Boyutlu Vektör Uzayına Çevir (N x 3 Matrisi)
    # Satır ve sütunları düzleştirip sadece [R, G, B] vektörleri bırakıyoruz
    pixel_vectors = image_matrix.reshape((-1, 3))
    pixel_vectors = np.float32(pixel_vectors)
    
    # 2. K-Means Algoritması Kriterleri (Durdurma Koşulları)
    # Algoritma ya 10 iterasyon yapacak ya da pikseller merkezden 1.0 birimden fazla sapmayana kadar çalışacak
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # 3. Gözetimsiz Öğrenme Motorunu Ateşle
    # cv2.KMEANS_PP_CENTERS: Merkez noktalarını akıllıca dağıtır (K-Means++ algoritması)
    ret, labels, centers = cv2.kmeans(pixel_vectors, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # 4. Ağırlık Merkezlerini (Centroids) 8-bit Renk Kodlarına Çevir
    centers = np.uint8(centers)
    
    # 5. Tüm pikselleri, ait oldukları kümenin merkez rengine zorla
    quantized_vectors = centers[labels.flatten()]
    
    # 6. Düzleştirilmiş vektörleri yeniden 2 boyutlu resim matrisine çevir
    final_image = quantized_vectors.reshape(image_matrix.shape)
    
    log_msg = f"🎨 K-Means Kümeleme: Görüntü matrisi {k} algoritmik ağırlık merkezine (Centroid) indirgendi."
    
    return final_image, log_msg