import os
import sys
import cv2
import json
import uuid
from datetime import datetime

# =====================================================================
# [SİSTEM MİMARİSİ 1]: Dinamik Kök Dizin ve Güvenlik İzolasyonu
# =====================================================================
def get_base_dir():
    """
    Neden Eklendi: Uygulama Inno Setup ile kapalı devre bir .exe haline getirildiğinde, 
    göreceli yollar (relative paths, örn: './klasor') işletim sisteminin çalışma dizinine 
    (Working Directory) göre hata verebilir veya kritik sistem dizinlerine yazma 
    girişiminde bulunarak Path Traversal (Dizin Atlatma) açıklarına yol açabilir.
    Bu fonksiyon, yürütülebilir dosyanın mutlak (absolute) yolunu donanımsal olarak kilitler.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "senta_export_logs")

# I/O İşlemleri için dizin ağacının asenkron olmayan, güvenli inşası
# Dizin yoksa FileNotFoundError çökmesini önlemek için hiyerarşik olarak üretilir.
if not os.path.exists(TRAINING_DATA_DIR):
    os.makedirs(TRAINING_DATA_DIR)
    os.makedirs(os.path.join(TRAINING_DATA_DIR, "images"))
    os.makedirs(os.path.join(TRAINING_DATA_DIR, "metadata"))

def log_training_data(final_image, state_data, target_concept, intensity_val, spatial_val, kmeans_val):
    """
    [VERİ GÖLÜ (DATA LAKE) YÖNETİMİ VE ETL PROTOKOLÜ]
    Neden Eklendi: SENTA'nın ürettiği her başarılı çıktıyı deterministik parametreleriyle 
    (BPM, Timbre, Slider verileri) eşleştirerek gelecekteki Fine-Tuning (İnce Ayar) ve 
    Gözetimli Öğrenme (Supervised Learning) süreçleri için yapılandırılmamış bir 
    veri ambarı (Data Lake) oluşturur.
    """
    try:
        # =====================================================================
        # 1. Veri Bütünlüğü Doğrulaması (Sanitization)
        # =====================================================================
        # Neden Eklendi: Hatalı, boş veya None tipindeki matrislerin diske yazılarak 
        # eğitim setini (Dataset) zehirlemesini (Data Poisoning) engellemek için.
        if final_image is None or not state_data:
            return False

        # =====================================================================
        # 2. Kriptografik Dosya Kimliklendirme (UUIDv4)
        # =====================================================================
        # Neden Eklendi: Klasik ardışık isimlendirmeler, ardışık asenkron üretimlerde 
        # Race Condition (Aynı dosyaya yazma çarpışması) hatalarına yol açar.
        # Zaman damgası ve UUIDv4 birleştirilerek astronomik çarpışma direnci (Collision Resistance) sağlanır.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]
        safe_filename = f"senta_{timestamp}_{unique_id}"

        image_path = os.path.join(TRAINING_DATA_DIR, "images", f"{safe_filename}.jpg")
        json_path = os.path.join(TRAINING_DATA_DIR, "metadata", f"{safe_filename}.json")

        # =====================================================================
        # 3. Renk Uzayı Dönüşümü ve Matris İhracı
        # =====================================================================
        # Neden Eklendi: SENTA iç mimaride algısal doğruluk için RGB çalışırken,
        # OpenCV'nin disk yazma motoru (imwrite) donanımsal olarak BGR dizilimi bekler.
        # Renk spektrumunun tersine dönmesini engellemek için kanal yer değişimi yapılır.
        bgr_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, bgr_image)

        # =====================================================================
        # 4. JSON Formatında Deterministik Etiketleme (Labeling)
        # =====================================================================
        # Neden Eklendi: Pikseller tek başına makine için anlamsızdır. Bu sözlük,
        # görüntüyü var eden ses frekanslarını ve matematiksel filtre katsayılarını
        # ilişkisel olmayan (NoSQL) yapıda birleştirerek tensör modellerine girdi (Feature) sağlar.
        metadata = {
            "id": safe_filename,
            "created_at": timestamp,
            "audio_determinants": {
                "bpm": state_data.get("bpm"),
                "timbre_hz": state_data.get("timbre"),
                "hue_degree": state_data.get("hue_val"),
                "loudness": state_data.get("loudness")
            },
            "art_style_reference": state_data.get("active_brush"),
            "focal_concept": target_concept if target_concept else "Pure Abstract",
            "mathematical_modifiers": {
                "adaptive_histogram_intensity": intensity_val,
                "spatial_convolution_sharpness": spatial_val,
                "kmeans_color_clusters": kmeans_val
            }
        }

        # =====================================================================
        # 5. Güvenli Disk Yazma (I/O) Operasyonu
        # =====================================================================
        # Neden Eklendi: Python'ın varsayılan CP1254 (Windows) karakter kodlaması, 
        # sanatçı isimlerindeki aksanlı harflerde çökmeye neden olur.
        # ensure_ascii=False ve utf-8 zorlamasıyla veri bütünlüğü korunur.
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        return True

    except Exception as e:
        # Fatal Exception Handling: Disk dolması veya yetki ihlali gibi kritik
        # I/O hatalarının ana arayüzü (Gradio) kitlemesini engellemek için izole edilir.
        print(f"[I/O İSTİSNASI]: Veri gölüne yazma işlemi başarısız. Hata Kodu: {str(e)}")
        return False