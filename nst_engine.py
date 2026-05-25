import os
import cv2
import numpy as np
import hashlib # Benzersiz cache anahtarları (hash) oluşturmak için

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

# --- OTOMATİK İNDİRME VE ÖNBELLEK (CACHE) SİSTEMİ ---
print("[SENTA SİSTEMİ]: NST Tensor Modeli kontrol ediliyor...")
os.environ["TFHUB_CACHE_DIR"] = "./senta_ai_models"

try:
    NST_MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    print("[SENTA SİSTEMİ]: NST Modeli başarıyla CUDA/CPU belleğine alındı (Offline/Cache).")
except Exception as e:
    NST_MODEL = None
    print(f"[KRİTİK HATA]: Model indirilemedi veya okunamadı! Hata: {str(e)}")

# --- [YENİ] MEMOIZATION (SENTEZ ÖNBELLEĞİ) ---
# Niçin: Aynı görsel ve aynı stil tekrar istendiğinde VRAM'i yormadan anında sonucu RAM'den dönmek için.
SENTEZ_CACHE = {}
MAX_CACHE_SIZE = 10 # RAM'in şişmemesi için en fazla son 10 sentezi tutarız.

def generate_cache_key(content_image, style_image, style_weight):
    """
    Girdilerden benzersiz bir MD5 Hash üretir.
    Eğer matrislerde veya kaydırıcıda (weight) en ufak bir değişiklik olursa farklı hash çıkar.
    """
    # Görüntü matrislerini bayt dizisine çevir ve ağırlığı ekle
    content_bytes = content_image.tobytes()
    style_bytes = style_image.tobytes()
    weight_str = str(round(style_weight, 2)).encode('utf-8')
    
    # Hash oluştur
    m = hashlib.md5()
    m.update(content_bytes)
    m.update(style_bytes)
    m.update(weight_str)
    return m.hexdigest()

# ... (preprocess_image ve tensor_to_image fonksiyonları aynı kalacak) ...
def preprocess_image(image_matrix, max_dim=512):
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
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def apply_senta_signature_enhancement(raw_stylized_image, original_content_image):
    if raw_stylized_image.shape[:2] != original_content_image.shape[:2]:
        h, w = original_content_image.shape[:2]
        raw_stylized_image = cv2.resize(raw_stylized_image, (w, h), interpolation=cv2.INTER_CUBIC)

    lab_stylized = cv2.cvtColor(raw_stylized_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_original = cv2.cvtColor(original_content_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    l_style, a_style, b_style = cv2.split(lab_stylized)
    _, a_orig, b_orig = cv2.split(lab_original)
    
    a_blended = (a_style * 0.4) + (a_orig * 0.6)
    b_blended = (b_style * 0.4) + (b_orig * 0.6)
    
    merged_lab = cv2.merge((l_style, a_blended, b_blended))
    color_restored_img = cv2.cvtColor(np.clip(merged_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    gaussian_blur = cv2.GaussianBlur(color_restored_img, (0, 0), 2.0)
    sharpened_img = cv2.addWeighted(color_restored_img, 1.5, gaussian_blur, -0.5, 0)

    lab_final = cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2LAB)
    l_final, a_final, b_final = cv2.split(lab_final)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l_final)
    
    return cv2.cvtColor(cv2.merge((cl, a_final, b_final)), cv2.COLOR_LAB2RGB)


def synthesize_nst_art(content_image, style_image, style_weight=1.0):
    global SENTEZ_CACHE
    try:
        if NST_MODEL is None:
            return None, "SİSTEM HATASI: AI Modeli yüklenemedi."
        if content_image is None or style_image is None:
            return None, "HATA: İçerik veya Stil görseli eksik!"

        MAX_SAFE_PIXELS = 2048 * 2048
        if (content_image.shape[0] * content_image.shape[1] > MAX_SAFE_PIXELS) or (style_image.shape[0] * style_image.shape[1] > MAX_SAFE_PIXELS):
             return None, "GÜVENLİK HATASI: Görsel çok büyük."

        # Numpy Array'e çevir
        content_image = np.array(content_image, dtype=np.uint8)
        style_image = np.array(style_image, dtype=np.uint8)
        style_weight = max(0.0, min(float(style_weight), 1.0))

        # --- 1. ÖNBELLEK (CACHE) KONTROLÜ ---
        # Aynı girdi daha önce hesaplandıysa, doğrudan RAM'den döndür!
        cache_key = generate_cache_key(content_image, style_image, style_weight)
        if cache_key in SENTEZ_CACHE:
            print("[SENTA SİSTEMİ]: Sentez önbellekten (RAM) getirildi. (Süre: 0.0sn)")
            return SENTEZ_CACHE[cache_key], f"⚡ Hızlı Sentez (Önbellekten Yüklendi) - Stil Yoğunluğu: %{int(style_weight*100)}"

        # --- 2. EĞER ÖNBELLEKTE YOKSA ML İŞLEMİ YAP ---
        content_tensor = preprocess_image(content_image, max_dim=512)
        style_tensor = preprocess_image(style_image, max_dim=384)
        
        results = NST_MODEL(tf.constant(content_tensor), tf.constant(style_tensor))
        stylized_tensor = results[0]

        blended_tensor = (content_tensor * (1.0 - style_weight)) + (stylized_tensor * style_weight)
        raw_stylized_image = tensor_to_image(blended_tensor)

        h, w = content_image.shape[:2]
        resized_raw_image = cv2.resize(raw_stylized_image, (w, h), interpolation=cv2.INTER_CUBIC)

        final_image = apply_senta_signature_enhancement(resized_raw_image, content_image)

        # --- 3. SONUCU ÖNBELLEĞE KAYDET ---
        # Bellek şişmesini (OOM) önlemek için eski kayıtları sil
        if len(SENTEZ_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = list(SENTEZ_CACHE.keys())[0]
            del SENTEZ_CACHE[oldest_key]
            
        SENTEZ_CACHE[cache_key] = final_image

        return final_image, f"Sentez ve İyileştirme Başarılı (GPU ile işlendi) - Stil Yoğunluğu: %{int(style_weight*100)}"

    except Exception as e:
        return None, f"NST ÇÖKME HATASI: {str(e)}"

# ... (apply_adaptive_histogram_bending fonksiyonu aynı kalacak) ...
def apply_adaptive_histogram_bending(image_matrix, loudness, user_intensity_slider):
    if image_matrix is None:
        return None, "HATA: İşlenecek matris yok."
    if not isinstance(image_matrix, np.ndarray):
        return None, "HATA: Girdi formatı geçersiz, Numpy matrisi bekleniyor."

    normalized_loudness = np.clip(loudness / 100.0, 0.5, 1.5)
    intensity = max(0.1, float(user_intensity_slider))
    gamma = 1.0 / (intensity * normalized_loudness)
    invGamma = 1.0 / gamma
    
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bended_image = cv2.LUT(image_matrix, table)
    
    lab = cv2.cvtColor(bended_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5 + (intensity / 2.0), tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    log_msg = f"📉 Adaptif Histogram Eğrisi Uygulandı: γ (Gamma) = {gamma:.2f} | Müzik Çarpanı: {normalized_loudness:.2f}"
    
    return final_image, log_msg