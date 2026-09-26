import os
import sys
import cv2
import numpy as np
import hashlib
import warnings

# =====================================================================
# [SİSTEM YAPILANDIRMASI]: Çevrimdışı Dizin Çözümleyici ve Uyarılar
# =====================================================================
def get_base_dir():
    """
    Neden Eklendi: Projenin taşınabilir (Portable) yapısını korumak için. 
    PyInstaller ile derlenmiş yürütülebilir (.exe) dosya ile geliştirme 
    ortamındaki (.py) yol (path) hiyerarşisi farkını dinamik olarak tolere eder.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()

# C++ seviyesindeki donanım uyarılarını ve log kirliliğini bastırır.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub

# CUDA GPU Bellek Yönetimi (Memory Growth)
# Neden Eklendi: TensorFlow'un GPU VRAM'inin tamamını rezerve edip (OOM - Out of Memory)
# Gradio arayüzünü kitlemesini engellemek için bellek büyümesi dinamik (True) hale getirildi.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[DONANIM UYARISI]: CUDA Bellek optimizasyonu yapılandırılamadı. {e}")

# =====================================================================
# [MODEL İLKLEME]: Çevrimdışı Magenta NST Ağı
# =====================================================================
# [KRİTİK ZIRH]: Göreceli yol yerine mutlak yol (Absolute Path) kullanıldı!
# Kısayoldan çalıştırmalarda modelin Masaüstünde aranmasını engeller.
cache_dir = os.path.join(BASE_DIR, "solenta_ai_models") if os.path.exists(os.path.join(BASE_DIR, "solenta_ai_models")) else os.path.join(BASE_DIR, "senta_ai_models")
os.environ["TFHUB_CACHE_DIR"] = cache_dir

try:
    # Arbitrary Image Stylization ağının VRAM'e kalıcı olarak yüklenmesi.
    NST_MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
except Exception as e:
    NST_MODEL = None
    print(f"[KRİTİK HATA]: Nöral Ağ yüklenemedi. Bağımlılıkları kontrol ediniz: {str(e)}")

# =====================================================================
# [BELLEK YÖNETİMİ]: Deterministik Önbellek (Memoization) Sistemi
# =====================================================================
# Neden Eklendi: Aynı piksel matrislerinin tensör ağından defalarca geçirilmesini 
# engelleyerek inferans süresini O(1) seviyesine çekmek.
SENTEZ_CACHE = {}
MAX_CACHE_SIZE = 10 

def generate_cache_key(content_image, style_image, style_weight):
    """
    Matris girdilerinden benzersiz bir MD5 özet değeri (Hash) üretir.
    Girdi matrisindeki tek bir piksel veya ağırlık değiştiğinde algoritmik çığ etkisiyle 
    (Avalanche Effect) yeni bir anahtar oluşturur.
    """
    content_bytes = content_image.tobytes()
    style_bytes = style_image.tobytes()
    weight_str = str(round(style_weight, 2)).encode('utf-8')
    
    m = hashlib.md5()
    m.update(content_bytes)
    m.update(style_bytes)
    m.update(weight_str)
    return m.hexdigest()

def preprocess_image(image_matrix, max_dim=512):
    """
    Neden Eklendi: OpenCV matrisini (NumPy) TensorFlow'un işleyebileceği 
    4 boyutlu (Batch, Height, Width, Channels) normalize edilmiş bir tensöre çevirir.
    """
    img = tf.convert_to_tensor(image_matrix, dtype=tf.float32)
    img = img / 255.0  # [0, 255] aralığından [0, 1] aralığına Min-Max Normalizasyonu
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # Batch boyutu ekleme
    return img

def tensor_to_image(tensor):
    """ TensorFlow tensörünü yeniden standart 8-bit RGB görüntü matrisine geri katlar. """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def apply_solenta_signature_enhancement(raw_stylized_image, original_content_image):
    """
    Neden Eklendi: NST modeli fırça darbelerini mükemmel aktarır ancak orijinal renklerin
    solmasına (Color Shift) neden olur. Bu modül, orijinal fotoğrafın Luminance (Işık) 
    değerlerini koruyarak stil transferinin renklerini Alpha Blending ile geri yükler.
    """
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
    """
    [NÖRAL STİL TRANSFER (NST) MOTORU]
    Arbitrary Image Stylization modeli kullanılarak, içerik görüntüsünün uzaysal
    yapısı (Spatial Structure) ile stil görüntüsünün düşük seviyeli özellikleri (Low-level features,
    fırça darbeleri, doku) derin konvolüsyonel katmanlarda (VGG-19) sentezlenir.
    """
    global SENTEZ_CACHE
    try:
        if NST_MODEL is None:
            return None, "SİSTEM HATASI: Yapay Sinir Ağı yüklenemedi."
        if content_image is None or style_image is None:
            return None, "İSTİSNA: Girdi matrisleri eksik."

        # VRAM Taşmasını (Out of Memory - OOM) Önleme Sınırı
        MAX_SAFE_PIXELS = 2048 * 2048
        if (content_image.shape[0] * content_image.shape[1] > MAX_SAFE_PIXELS) or (style_image.shape[0] * style_image.shape[1] > MAX_SAFE_PIXELS):
             return None, "BELLEK İSTİSNASI: Matris boyutu güvenlik sınırlarını aşıyor."

        content_image = np.array(content_image, dtype=np.uint8)
        style_image = np.array(style_image, dtype=np.uint8)
        style_weight = max(0.0, min(float(style_weight), 1.0))

        # 1. Bellek Çözümleme (Memoization) Kontrolü
        cache_key = generate_cache_key(content_image, style_image, style_weight)
        if cache_key in SENTEZ_CACHE:
            return SENTEZ_CACHE[cache_key], f"RAM Önbellek İsabeti (Hit) - Inferans Bypass Edildi | Stil Yoğunluğu: %{int(style_weight*100)}"

        # 2. Tensör Dönüşümleri ve İleri Besleme (Feed-Forward)
        content_tensor = preprocess_image(content_image, max_dim=512)
        style_tensor = preprocess_image(style_image, max_dim=384)
        
        # Yapay Sinir Ağının tetiklenmesi
        results = NST_MODEL(tf.constant(content_tensor), tf.constant(style_tensor))
        stylized_tensor = results[0]

        # =====================================================================
        # [BAŞ MİMAR MÜDAHALESİ]: Boyut Hizalama Zırhı
        # VGG-19 Max Pooling katmanlarının küsuratlı pikselleri yuvarlamasından 
        # kaynaklanan (AddV2) boyut uyumsuzluğu hatasını engeller. Modelden çıkan 
        # matrisi zorla orijinal matrisin boyutlarına ezer.
        # =====================================================================
        target_shape = tf.shape(content_tensor)[1:3]
        stylized_tensor = tf.image.resize(stylized_tensor, target_shape)

        # 3. İstatistiksel Karışım (Interpolation)
        blended_tensor = (content_tensor * (1.0 - style_weight)) + (stylized_tensor * style_weight)
        raw_stylized_image = tensor_to_image(blended_tensor)

        # 4. Yeniden Ölçeklendirme ve SOLENTA Rötüşü
        h, w = content_image.shape[:2]
        resized_raw_image = cv2.resize(raw_stylized_image, (w, h), interpolation=cv2.INTER_CUBIC)
        final_image = apply_solenta_signature_enhancement(resized_raw_image, content_image)

        # 5. FIFO (First-In-First-Out) prensibiyle önbellek temizliği ve kayıt
        if len(SENTEZ_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = list(SENTEZ_CACHE.keys())[0]
            del SENTEZ_CACHE[oldest_key]
            
        SENTEZ_CACHE[cache_key] = final_image

        return final_image, f"Nöral Ağ İleri Beslemesi Başarılı (TF Hub) - Stil Yoğunluğu: %{int(style_weight*100)}"

    except Exception as e:
        return None, f"NST ÇÖKME HATASI: {str(e)}"

def apply_adaptive_histogram_bending(image_matrix, loudness, user_intensity_slider):
    """
    Neden Eklendi: Ses sinyalinin gürlüğünü (Loudness), piksellerin ışık şiddetine 
    matematiksel olarak bağlamak için. Doğrusal olmayan (Non-linear) Gamma Bükülmesi 
    ve CLAHE kullanarak bölgesel kontrastı optimize eder.
    """
    if image_matrix is None:
        return None, "İSTİSNA: İşlenecek matris boş."
    if not isinstance(image_matrix, np.ndarray):
        return None, "İSTİSNA: Geçersiz veri tipi, NumPy matrisi gereklidir."

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
    log_msg = f"Adaptif Histogram Modifikasyonu: Gamma = {gamma:.2f} | Ses Çarpanı: {normalized_loudness:.2f}"
    
    return final_image, log_msg