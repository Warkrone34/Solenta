import cv2
import numpy as np

def apply_timbre_driven_convolution(image_matrix, timbre, user_sharpness):
    """
    Neden Eklendi: Görselin yapısal keskinliğini sabit bir değerle (hardcoded) değil, 
    işitsel sinyalin spektral ağırlık merkeziyle (Timbre) dinamik olarak bağdaştırmak için.
    Yüksek frekanslı (yırtıcı) sesler matrisin merkez katsayısını artırarak görselde 
    daha sert lokal kontrast (Spatial Sharpening) yaratır.
    """
    if image_matrix is None:
        return None, ""
        
    # =====================================================================
    # 1. Spektral Tını (Timbre) Normalizasyonu
    # =====================================================================
    # Müzikal frekansların ağırlık merkezi (Centroid) genellikle 200 Hz ile 4000 Hz arasında dalgalanır.
    # Min-Max normalizasyonu ile bu değer 0.0 ile 1.0 arasında standart bir çarpan haline getirilir.
    min_timbre, max_timbre = 200.0, 4000.0
    normalized_timbre = np.clip((timbre - min_timbre) / (max_timbre - min_timbre), 0.0, 1.0)
    
    # =====================================================================
    # 2. Kernel (Matris) Ağırlık Katsayısı Hesabı
    # =====================================================================
    # Normalizasyon sabiti, kullanıcının belirlediği keskinlik eşiği ile çarpılarak w değişkeni elde edilir.
    w = normalized_timbre * float(user_sharpness)
    
    # Matematiksel sıfır durumunda gereksiz işlemci maliyetini (O(N^2) Matris Çarpımı) önlemek için işlem baypas edilir.
    if w == 0.0:
        return image_matrix, "Dinamik Konvolusyon: Baypas Edildi (Katsayi = 0.0)"

    # =====================================================================
    # 3. Dinamik Konvolüsyon Matrisinin (Laplacian / Unsharp Mask) İnşası
    # =====================================================================
    # Merkez piksel (1 + 4w) gücünde aydınlatılırken, komşu pikseller (-w) ağırlığında karartılarak
    # pikseller arası türev (değişim hızı) yapay olarak artırılır ve yüksek frekanslı detaylar öne çıkar.
    kernel = np.array([
        [ 0, -w,  0],
        [-w, 1 + 4*w, -w],
        [ 0, -w,  0]
    ], dtype=np.float32)
    
    # =====================================================================
    # 4. Uzaysal Filtreleme (Spatial Filtering)
    # =====================================================================
    # Çekirdek matris (Kernel), ana matrisin tüm pikselleri üzerinde kaydırılarak (Convolution) yeni değerler hesaplanır.
    filtered_image = cv2.filter2D(image_matrix, -1, kernel)
    
    # Matris işlemlerinden doğan renk taşmalarını (Overflow/Underflow) 0-255 aralığına sıkıştırır (Clipping).
    final_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    log_msg = f"Dinamik Konvolusyon Matrisi: K (Merkez) = {1 + 4*w:.2f} | Spektral Carpan = {normalized_timbre:.2f}"
    
    return final_image, log_msg


def apply_kmeans_color_quantization(image_matrix, k_slider):
    """
    Neden Eklendi: Görseldeki kesintisiz (Continuous) renk uzayını, kullanıcının belirlediği
    K sayısında kesikli (Discrete) renklere indirgeyerek (Color Quantization) 
    vektörel ve posterize (Pop-Art/Retro) bir etki yaratmak için.
    """
    if image_matrix is None:
        return None, ""
        
    k = int(k_slider)
    
    # Kuantizasyon alt limiti (1) veya üst limiti (64+) aşıldığında matrisi orijinal halinde bırakır.
    if k <= 1 or k >= 64:
        return image_matrix, "Renk Kuantizasyonu: Baypas Edildi (Sinirsiz Renk Uzayi)"

    # =====================================================================
    # 1. Boyut Dönüşümü (Flattening)
    # =====================================================================
    # 3 Boyutlu tensör (Yükseklik, Genişlik, RGB), K-Means algoritmasının iteratif hesaplamalar 
    # yapabilmesi için N adet pikselden oluşan 3 boyutlu düz bir vektör uzayına indirgenir.
    pixel_vectors = image_matrix.reshape((-1, 3))
    pixel_vectors = np.float32(pixel_vectors)
    
    # =====================================================================
    # 2. Optimizasyon ve Durdurma Kriterleri
    # =====================================================================
    # K-Means algoritmasının sonsuz döngüye girmemesi için: Maksimum 10 iterasyon veya
    # merkez noktalarının (Centroids) kayma miktarının 1.0 epsilon değerinin altına düşmesi kuralı eklenir.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # =====================================================================
    # 3. K-Means++ Clustering (Gözetimsiz Öğrenme)
    # =====================================================================
    # cv2.KMEANS_PP_CENTERS: Merkez noktalarını başlangıçta akıllıca (birbirinden uzak) atayarak
    # algoritmanın zayıf lokal minimumlara (Poor Local Optima) takılmasını engeller.
    ret, labels, centers = cv2.kmeans(pixel_vectors, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # =====================================================================
    # 4. Yeniden İnşa (Reconstruction ve Vector Quantization)
    # =====================================================================
    # Hesaplanan K adet merkez noktası (Centroid), görüntü matrisine geri yazılabilmek için 8-bit formatına dönüştürülür.
    centers = np.uint8(centers)
    
    # Tüm pikseller, Öklid uzaklığına göre atandıkları (label) kümenin merkez rengiyle zorla değiştirilir.
    quantized_vectors = centers[labels.flatten()]
    
    # İşlenmiş düz vektör matrisi, orijinal fotoğrafın (Y, X, RGB) tensör boyutlarına geri katlanır.
    final_image = quantized_vectors.reshape(image_matrix.shape)
    
    log_msg = f"K-Means Kumeleme: Matris {k} adet algoritmik agirlik merkezine (Centroid) indirgendi."
    
    return final_image, log_msg