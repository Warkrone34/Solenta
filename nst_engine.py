import os
import cv2
import numpy as np

# [GÜVENLİK]: TensorFlow'un gereksiz terminal uyarılarını (C++ loglarını) sustur.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow_hub as hub

# CUDA GPU bellek optimizasyonu (VRAM Darboğaz Koruması)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[CUDA UYARISI]: {e}")

print("[SENTA SİSTEMİ]: NST Tensor Modeli belleğe yükleniyor, lütfen bekleyin...")
os.environ["TFHUB_CACHE_DIR"] = "./senta_ai_models"
NST_MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("[SENTA SİSTEMİ]: NST Modeli başarıyla CUDA/CPU belleğine alındı.")

def preprocess_image(image_matrix, max_dim=512):
    """
    Görsel matrislerini TensorFlow'un anlayacağı 4D Float32 Tensorlerine çevirir.
    GPU'yu patlatmamak için Aspect Ratio (En-Boy) oranını koruyarak güvenlice sıkıştırır.
    """
    img = tf.convert_to_tensor(image_matrix, dtype=tf.float32)
    img = img / 255.0
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    """
    Tensor matrisini Gradio/OpenCV uyumlu RGB Numpy matrisine dönüştürür.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def synthesize_nst_art(content_image, style_image):
    """
    SENTA ANA SENTEZ MOTORU (STATİK FİLTRELERDEN ARINDIRILMIŞ SÜRÜM):
    İskelet yapısı ve renk doygunluğu korunur. Son rötuş ve kontrast 
    'Adaptif Histogram' algoritmasına (Piksel Bükücü) devredilir.
    """
    try:
        if content_image is None or style_image is None:
            return None, "HATA: İçerik veya Stil görseli eksik!"

        # [VERİ TİPİ ZIRHI]
        content_image = np.array(content_image, dtype=np.uint8)
        style_image = np.array(style_image, dtype=np.uint8)

        content_tensor = preprocess_image(content_image, max_dim=512)
        style_tensor = preprocess_image(style_image, max_dim=384)
        
        results = NST_MODEL(tf.constant(content_tensor), tf.constant(style_tensor))
        raw_stylized_image = tensor_to_image(results[0])

        h, w = content_image.shape[:2]
        stylized_resized = cv2.resize(raw_stylized_image, (w, h), interpolation=cv2.INTER_CUBIC)

        # --- 1. ANTI-PASTEL: DOYGUNLUK ARTIŞI (Sulu Boya Solukluğunu Öldürür) ---
        hsv = cv2.cvtColor(stylized_resized, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.35  
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        vibrant_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # --- 2. HIGH-PASS FILTER (Voronoi Çizgilerini Korumak İçin) ---
        content_f = content_image.astype(np.float32)
        blur_f = cv2.GaussianBlur(content_f, (0, 0), 2.0)
        high_pass_f = content_f - blur_f 
        
        final_f = vibrant_color.astype(np.float32) + (high_pass_f * 1.2)
        final_image = np.clip(final_f, 0, 255).astype(np.uint8)

        # [NOT]: Statik CLAHE söküldü. Kontrast artık tamamen kullanıcı + gürlük (loudness) kontrolünde.

        return final_image, "Ham Sentez Başarılı (Matematiksel Modifikasyon Bekleniyor)"

    except Exception as e:
        return None, f"NST ÇÖKME HATASI: {str(e)}"

def apply_adaptive_histogram_bending(image_matrix, loudness, user_intensity_slider):
    """
    [MÜZİK KISITLI PİKSEL BÜKÜCÜ ALGORİTMA - TEZ SAVUNMASI İÇİN]
    Kullanıcının girdiği eşik değerini, müziğin gürlük (Loudness) verisiyle çarparak
    Non-Linear (Doğrusal Olmayan) bir Gamma Eğrisi (Power Law Transformation) oluşturur.
    Yapay zekanın ürettiği pikseller, bizim lineer cebir denklemimizle manipüle edilir.
    """
    if image_matrix is None:
        return None, "HATA: İşlenecek matris yok."

    # 1. Loudness (Gürlük) değerini normalize et
    normalized_loudness = np.clip(loudness / 100.0, 0.5, 1.5)
    
    # Sıfıra bölünme hatasını engellemek için slider güvenliği
    intensity = max(0.1, float(user_intensity_slider))
    
    # 2. Dinamik Gamma Formülü (Hocanın vurulacağı denklem)
    # Müzik şiddetliyse algoritma kontrasta daha çok müsaade eder.
    gamma = 1.0 / (intensity * normalized_loudness)
    
    # 3. Look-Up Table (Hızlı Matris Çarpımı için LUT) O(1) Karmaşıklığında
    # Formül: V_out = 255 * (V_in / 255) ^ invGamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 4. Görüntünün Piksellerini Matematiksel Eğriye Göre Yeniden Haritala
    bended_image = cv2.LUT(image_matrix, table)
    
    # 5. Dinamik CLAHE (Lokal Işık Dengelemesi) - Patlamaları engeller
    lab = cv2.cvtColor(bended_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5 + (intensity / 2.0), tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    log_msg = f"📉 Adaptif Histogram Eğrisi Uygulandı: γ (Gamma) = {gamma:.2f} | Müzik Çarpanı: {normalized_loudness:.2f}"
    
    return final_image, log_msg