import os
import sys
import atexit
from pathlib import Path

# İşletim sistemi seviyesinde programın kapatılmasını yakalar.
# Neden: Tarayıcı kapansa bile arka planda asılı kalan (ghost process) GPU işlemlerini öldürüp VRAM sızıntısını engellemek için.
atexit.register(os._exit, 0)

# =====================================================================
# [SISTEM YAPILANDIRMASI 1]: Standart Cikti (Stdout/Stderr) İzolasyonu
# Neden Eklendi: PyInstaller --windowed argumani ile derlendiginde, Gradio
# log yazacak bir konsol bulamadigi icin Fatal Error verip cokmektedir.
# Cokmeyi onlemek amaciyla tum terminal ciktilari null (hiclik) yonlendirilir.
# =====================================================================
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
    
# =====================================================================
# [SISTEM YAPILANDIRMASI 2]: Çevrimdisi (Air-Gapped) Calisma Ortami
# Neden Eklendi: Uygulamanin dış aglara baglanmasini engellemek ve yalnizca
# yerel (local) _internal dizinindeki agirlik (weight) dosyalarini okumaya zorlamak.
# =====================================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =====================================================================
# [SISTEM YAPILANDIRMASI 3]: PyInstaller Bağımlılık Onarimi (Self-Healing)
# Neden Eklendi: Fastapi ve Pydantic gibi kutuphaneler derleme (compile)
# asamasinda version.txt dosyalarini kaybetmektedir. Bu durum calisma
# aninda modullerin cokmesine yol actigindan, eksik dosyalar sanal olarak uretilir.
# =====================================================================
if getattr(sys, 'frozen', False):
    # Programın çalıştığı ana dizini tespit eder.
    base_dir = os.path.dirname(sys.executable)
    # Paketlenmiş kütüphanelerin bulunduğu _internal dizininin yolunu oluşturur.
    internal_dir = os.path.join(base_dir, "_internal")
    
    # Sıkıştırma sırasında version.txt dosyalarını kaybedip çökmeye meyilli kütüphanelerin listesi.
    problematic_libs = ["groovy", "safehttpx", "gradio_client", "pydantic", "fastapi"]
    
    # Listelenen her bir kütüphane için sanal onarım döngüsü başlatır.
    for lib in problematic_libs:
        lib_dir = os.path.join(internal_dir, lib)
        # Kütüphane klasörü yoksa zorla oluşturur.
        os.makedirs(lib_dir, exist_ok=True)
        version_file = os.path.join(lib_dir, "version.txt")
        
        # Eğer version.txt dosyası gerçekten silinmişse, içine sahte bir 1.0.0 sürümü yazar.
        # Neden: Kutüphanelerin versiyon kontrolü yaparken FileNotFoundError fırlatıp sistemi kitlemesini önler.
        if not os.path.exists(version_file):
            try:
                with open(version_file, "w", encoding="utf-8") as f:
                    f.write("1.0.0")
            except:
                pass

# Arayüz ve web sunucusu yönetimi için
import gradio as gr
# İşitsel sinyal analizi ve spektral hesaplamalar için
import librosa
# Yüksek boyutlu matris ve tensör işlemleri için
import numpy as np
# Görsel işleme ve K-Means, CLAHE, filtreleme algoritmaları için
import cv2 
# Akustik matrisin koordinatlarını hesaplamak için Voronoi şeması
from scipy.spatial import Voronoi

# Gevşek bağlı (loosely coupled) mimari prensibiyle yazılmış yerel modüllerin projeye dahil edilmesi
from audio_engine import analyze_audio_determinants, find_optimal_audio_segment
from color_engine import analyze_color_context
from nst_engine import synthesize_nst_art, apply_adaptive_histogram_bending
from filter_engine import apply_timbre_driven_convolution, apply_kmeans_color_quantization 
from solenta_logger import log_training_data
from data_manager import (
    get_all_artists,
    get_artworks_by_artist,
    get_artwork_image,
    get_ranked_top_10_artworks, 
    get_top_3_artworks_from_audio, 
    recolor_image_to_music_hue, 
    get_preset_audio_list,
    safe_image_read,
    calculate_image_features 
)

# PyTorch ve Difüzyon kütüphanelerinin donanımsal eksikliklerde (GPU yokluğu) ana sistemi 
# tamamen çökertmesini engellemek için hata yalıtımı (Try-Except) uygulanmıştır.
try:
    from diffusion_engine import synthesize_fusion_art
except ImportError:
    # Modül yüklenemezse Pro Mod sentetik üretim fonksiyonu devre dışı bırakılır.
    synthesize_fusion_art = None

def enforce_1080p_quality(image_matrix):
    """
    Neden Eklendi: Yerel donanim limitleri nedeniyle (VRAM) uretim 512px veya 768px'de
    yapilmaktadir. Bu fonksiyon, Lanczos-4 ve Unsharp Mask algoritmalarini kullanarak
    goruntuyu kalite kaybi yasamadan 1080p piksel yogunluguna (Super Resolution) ulastirir.
    """
    # Eğer girdi matrisi boşsa işlemi iptal edip geri döner (hata önleme).
    if image_matrix is None: return None
    # Matrisin mevcut yükseklik (h) ve genişlik (w) değerlerini çeker.
    h, w = image_matrix.shape[:2]
    
    # Sabitlenecek hedef çözünürlük yüksekliğini 1080 piksel olarak tanımlar.
    target_h = 1080
    
    # Eğer görsel zaten 1080p veya daha yüksek bir çözünürlüğe sahipse, işlemciyi yormamak için doğrudan geri yollar.
    if h >= target_h:
        return image_matrix
        
    # Görselin en-boy (aspect ratio) oranını korumak için gerekli matematiksel katsayıyı (ratio) hesaplar.
    ratio = target_h / float(h)
    # Orijinal genişliği bu katsayı ile çarparak bozulma yaratmayacak yeni genişliği tespit eder.
    target_w = int(w * ratio)
    
    # Lanczos-4 algoritması ile pikseller arası komşuluk değerlerini hesaplayarak görseli kalite kaybı olmadan büyütür (Upscaling).
    upscaled = cv2.resize(image_matrix, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Büyütme işlemi sonucu oluşan mikro düzeydeki piksel bulanıklığını Unsharp Mask algoritmasıyla keskinleştirir.
    blur = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    # Keskinleştirilmiş görüntüyü orijinal büyütülmüş görüntüyle ağırlıklı matris çarpımı (addWeighted) yaparak birleştirir.
    crisp_1080p = cv2.addWeighted(upscaled, 1.25, blur, -0.25, 0)
    
    return crisp_1080p

def generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness):
    """
    Neden Eklendi: Muzigin yapi taslarini (Ritim, Gurluk) rastgele olmayan, 
    matematiksel bir geometrik iskelete oturtmak icin kullanilir. 
    Voronoi semasi, BPM degerine bagli olarak hucrelere bolunur.
    """
    # 512x512 boyutlarında 3 kanallı (RGB) boş bir numpy matrisi (tuval) oluşturur.
    width, height = 512, 512
    # Tuvali tamamen siyah yapmak yerine estetik bir zemin için hafif koyu gri (20) ile doldurur.
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 20
    
    # BPM ve Gürlük (Loudness) değerlerini kullanarak Voronoi şemasında oluşacak poligon (hücre) sayısını hesaplar.
    # Neden: Hızlı müziklerde karmaşık, yavaş müziklerde sade iskeletler elde etmek için.
    num_points = int(max(12, (bpm / 2.0) + (loudness / 10.0)))
    
    # Rastgeleliği tamamen ortadan kaldırıp, deterministik (hep aynı sese aynı şekil) bir üretim için tüm verileri tohum (seed) yapar.
    seed_value = int(bpm + timbre + hue_val + loudness)
    rng = np.random.default_rng(seed_value)
    
    # Tohumlanmış matematiksel motor ile tuval üzerinde belirtilen sayıda rastgele (ama tekrarlanabilir) X,Y koordinatları üretir.
    points = rng.integers(0, 512, size=(num_points, 2))
    
    # Şemanın tuval dışına taşıp algoritmayı çökertmemesi için sınır (boundary) noktaları tanımlar.
    boundary_points = np.array([[-100, -100], [-100, 612], [612, -100], [612, 612]])
    # Merkez noktalar ile sınır noktalarını dikey olarak birleştirir (vstack).
    all_points = np.vstack([points, boundary_points])
    
    # Scipy kütüphanesi ile koordinatlardan Voronoi alan diyagramını oluşturur.
    vor = Voronoi(all_points)
    
    # Referans renge göre doygunluk (saturation) değerini sabitler.
    sat_val = 200 
    # Gürlük verisini baz alarak parlaklık (lightness) değerini 60 ile 255 aralığına dinamik olarak çeker (interp).
    val_lightness = int(np.interp(loudness, [20, 150], [60, 255]))
    
    # OpenCV'nin işleyebilmesi için rengi önce HSV formatında bir tensöre çevirir.
    hsv_color = np.uint8([[[hue_val, sat_val, val_lightness]]])
    # HSV tensörünü standart RGB formatına geri döndürür.
    base_rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]

    # Voronoi şemasındaki her bir bölge (poligon) için döngü başlatır.
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        # Geçersiz veya sonsuza giden (-1) bölgeleri işlememek için atlar (continue).
        if not region or -1 in region: continue
            
        # Bölgenin köşelerini temsil eden koordinatları bir liste haline getirir.
        polygon = [vor.vertices[i] for i in region]
        # Çizim algoritmasının anlayabileceği int32 formatında bir matrise şekillendirir (reshape).
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        # Ana RGB renginin üzerine, her poligon için algoritmik varyasyonlar (hafif ton farkları) ekleyerek rengi hesaplar.
        # Neden: İskeletin tekdüze (monoton) olmasını engelleyip sanatsal bir derinlik katmak için.
        color = (
            int(np.clip(base_rgb[0] + rng.integers(-45, 45), 0, 255)),
            int(np.clip(base_rgb[1] + rng.integers(-35, 35), 0, 255)),
            int(np.clip(base_rgb[2] + rng.integers(-55, 55), 0, 255))
        )
        # Hesaplanan köşeleri ve rengi baz alarak çokgeni tuval üzerine çizer ve içini boyar.
        cv2.fillPoly(canvas, [pts], color)
        
        # Poligonların kenar çizgilerinin kalınlığını, sesin yüksekliğine (loudness) göre oranlayarak hesaplar.
        edge_thickness = max(1, int(loudness / 25))
        # Kenar çizgilerinin rengini, ana rengin biraz daha koyu tonu olacak şekilde matematiksel olarak karartır.
        edge_color = (max(0, color[0]-60), max(0, color[1]-60), max(0, color[2]-60))
        # Köşeleri belirginleştirmek için Anti-Aliasing (kenar yumuşatma) özelliği ile poligon çerçevesini çizer.
        cv2.polylines(canvas, [pts], isClosed=True, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

    # Görselin pürüzsüzlüğünü kırıp analog bir doku (grain) katmak için gürlükle orantılı Gaussian gürültüsü üretir.
    noise = rng.normal(0, int(loudness/10), canvas.shape).astype(np.int16)
    # Gürültü matrisini ana tuvale ekler ve değerlerin 0-255 sınırları dışına çıkmasını np.clip ile engeller.
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return canvas

def generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro):
    """
    Neden Eklendi: AI tarafindan alinan deterministik kararlari sistemin kullanicisina
    ve log kayitlarina dogrudan seffaf (white-box) sekilde gostermek amaciyla olusturulmustur.
    """
    # AI tarafından alınan analitik kararların arayüzde okunabilir bir string (metin) raporuna dönüştürülmesi işlemini yapar.
    log_text = f"SOLENTA URETIM SURECI ANALIZI\n\n"
    log_text += f"1. ISITSEL-GORSEL HARITALAMA (DETERMINANTLAR)\n"
    log_text += f"|- Ritim (BPM)    : {bpm:.1f} (Voronoi tohum noktalarinin sayisini belirledi)\n"
    log_text += f"|- Tini (Timbre)  : {timbre:.1f} Hz (Dinamik Konvolusyon Matrisi carpani)\n"
    log_text += f"|- Perde (Hue)    : {hue_val} derece (Referans eseri tayf rengine senkronize etti)\n"
    log_text += f"|- Gurluk (Vol)   : {loudness:.1f} (Adaptif Histogram Gamma carpani)\n\n"
    
    log_text += f"2. TOPOLOJIK SENTEZ\n"
    log_text += f"|- Iskelet Algoritmasi: Akustik Voronoi Semasi\n"
    log_text += f"|- Odak Nesnesi: {target_concept_tr if (is_pro and target_concept_tr != 'Yok (Saf Soyut)') else 'Saf Soyut (Vektorel)'}\n"
    log_text += f"|- Cikti Cozunurlugu: 1080p (Super Resolution Entegre Edildi)\n"
    log_text += f"|- Kullanilan Motor: {engine_used}\n"
    log_text += f"|- Sistem Durumu: {status}\n"
        
    return log_text

def process_primary_generation(audio_upload, audio_preset, start_time_sec, synthesis_mode, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider):
    """
    Neden Eklendi: Ana orkestrasyon fonksiyonudur. Sinyal analizini baslatir, stil 
    referansini ceker ve uretim tipine gore (Light vs Pro) gorseli insa eder.
    """
    try:
        # Kullanıcının kendi dosyasını mı yoksa hazır kütüphaneyi mi kullandığını tespit eder.
        audio_file = audio_upload if audio_upload else audio_preset
        # Her ikisi de yoksa işlemi keser ve kullanıcıya arayüz üzerinden hata mesajı fırlatır.
        if not audio_file:
            return None, "HATA: Lutfen isitsel veri yukleyiniz.", gr.update(visible=False), gr.update(visible=False), None

        # Zaman aralığını float tipine zorlar, boşsa 0.0 saniyeden başlatır.
        safe_start = float(start_time_sec) if start_time_sec else 0.0
        
        # audio_engine modülünü çağırarak ses dosyasındaki fiziksel belirleyicileri (determinantları) analiz eder.
        bpm, timbre, hue_val, loudness = analyze_audio_determinants(audio_file, start_time=safe_start, duration=10.0)
        
        # Analiz çuvallarsa BPM None döner, bu durumda hatayı yakalayıp ekrana basar.
        if bpm is None: return None, f"Sinyal Analiz Hatasi: {timbre}", gr.update(), gr.update(), None

        # BPM hızına göre sanat akımı için kural bazlı (If-Then) bir Bulanık Mantık (Fuzzy Logic) kararı alır.
        if bpm > 120: akim_karari = "Analitik Kubizm (Yuksek Form Dominansi)"
        elif bpm > 80: akim_karari = "Sentetik Kubizm (Genis Renk Yuzeyleri)"
        else: akim_karari = "Proto-Kubizm (Sadelestirilmis Dogrusal Indeks)"

        # Veritabanından müzikal tınıya ve seçilen sanat akımına en uygun 3 eseri hiyerarşik olarak filtreler.
        top_3 = get_top_3_artworks_from_audio(timbre, akim_karari, target_concept_tr)
        if not top_3: return None, "HATA: Eser havuzu yetersiz.", gr.update(), gr.update(), None

        # En uygun (birinci) eserin diskteki yolunu (path) ve ismini değişkenlere atar.
        best_path = top_3[0][0]
        style_name = top_3[0][1].split(": ")[-1]
        
        # Matris okuma hatalarına karşı güvenli fonksiyonla resmi NumPy array olarak belleğe alır.
        raw_img = safe_image_read(best_path)
        # Seçilen referans resmin renk tonunu (Hue), müziğin ana frekans rengiyle matematiksel olarak eşitler.
        style_img = recolor_image_to_music_hue(raw_img, hue_val)
        # Voronoi fonksiyonunu çağırarak müziğin fiziksel parametrelerinden geometrik bir altyapı (canvas) çıkarır.
        structural_canvas = generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness)

        # Arayüzdeki Türkçe konsept isimlerini, yapay zekanın (LCM) anlayacağı İngilizce promptlara (komutlara) çevirir.
        concept_map = {
            "Yok (Saf Soyut)": None, 
            "İnsan Yüzü": "A dramatic close-up of a human face", 
            "Göz": "A highly detailed close-up of a human eye", 
            "Kedi": "A majestic cat silhouette",
            "Uçan Kuş": "A flying bird with spread wings", 
            "Yaşlı Ağaç": "An ancient solitary tree with sprawling branches", 
            "Çiçek (Lotus)": "A blooming lotus flower"
        }

        # Kullanıcının arayüzden seçtiği motor tipinde 'Pro Mod' kelimesi geçip geçmediğini boolean (True/False) olarak kaydeder.
        is_pro = "Pro Mod" in synthesis_mode
        
        # Post-Processing (filtreleme) aşamasında modeli baştan çalıştırmamak için sistemin anlık hafızasını (State) oluşturur.
        state_data = {
            "bpm": bpm, "timbre": timbre, "hue_val": hue_val, "loudness": loudness,
            "akim": akim_karari, "top_3": top_3, "safe_start": safe_start,
            "active_brush": top_3[0][1]
        }

        # Pro Mod (LCM Turbo) Seçildiyse:
        if is_pro:
            target_en = concept_map.get(target_concept_tr, None)
            # Difüzyon kütüphaneleri eksikse işlemi durdurur.
            if synthesize_fusion_art is None: return None, "SISTEM HATASI: Pro Mod kutuphaneleri (CUDA/Diffusers) eksik.", gr.update(), gr.update(), None
            # GPU üzerinde çalışan asıl yapay zeka sentez fonksiyonunu çağırarak iskelet ve stili birleştirir.
            final_image, status = synthesize_fusion_art(structural_canvas, style_img, is_music_mode=True, target_concept=target_en, style_name=style_name, loudness=loudness, intensity_slider=intensity_slider)
            
            # Hafızaya ham çıktıyı kaydeder, NST çıktısını None yapar.
            state_data["pro_base_output"] = final_image.copy() if final_image is not None else None
            state_data["raw_light_output"] = None
            engine_used = "LCM Turbo + MiDaS Depth"
            # Sentezden dönen uzun log metnini satır sonlarından (\n) ayırarak temizler.
            hist_log = status.split("\n")[-1] if "\n" in status else ""
            status = status.split("\n")[0]
        
        # Light Mod (Neural Style Transfer) Seçildiyse:
        else:
            # Klasik algoritmik stil transferi fonksiyonunu çağırarak işlemi CPU/hafif GPU ile tamamlar.
            raw_final_image, status = synthesize_nst_art(structural_canvas, style_img, style_weight=1.0)
            
            # Hafızaya NST çıktısını kaydeder, Pro çıktısını None yapar.
            state_data["raw_light_output"] = raw_final_image.copy() if raw_final_image is not None else None
            state_data["pro_base_output"] = None
            engine_used = "Neural Style Transfer"
            # NST sonucunun ışık değerini müziğin gürlüğüne (loudness) göre Gamma filtresiyle büker.
            final_image, hist_log = apply_adaptive_histogram_bending(raw_final_image, loudness, intensity_slider)

        # Sentez başarıyla tamamlandıysa, arayüzden gelen slider değerleriyle son rötüşleri yapar.
        if final_image is not None:
            # Tınıya duyarlı matris konvolüsyonu uygulayarak keskinliği artırır.
            final_image, spatial_log = apply_timbre_driven_convolution(final_image, timbre, spatial_slider)
            # K-Means algoritması ile renk uzayını daraltarak vektörel (posterize) bir görünüm sağlar.
            final_image, kmeans_log = apply_kmeans_color_quantization(final_image, kmeans_slider)
            
            # Son çıktıyı ekrana vermeden önce Lanczos-4 ile 1080p çözünürlüğe ölçekler.
            final_image = enforce_1080p_quality(final_image)
            
            # Başarılı üretimi ve kullanılan parametreleri JSON veri gölüne (Data Lake) kaydeder.
            log_training_data(final_image, state_data, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider)
        else:
            spatial_log = "Konvolusyon Pas Gecildi."
            kmeans_log = "K-Means Pas Gecildi."

        # Ekran logu için metin formatlama fonksiyonunu çağırır.
        base_log = generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro)
        state_data["base_log"] = base_log 
        
        # Log metnine filtreleme adımlarını (K-Means vb.) ekleyerek nihai raporu oluşturur.
        full_log = base_log + f"\n3. MATEMATIKSEL MODIFIKASYON (ALGORITMA RAPORU)\n|- {hist_log}\n|- {spatial_log}\n|- {kmeans_log}"

        # Kullanıcıya sunulacak alternatif fırça (stil) seçenekleri için listeyi düzenler.
        choices = [item[1] for item in top_3]
        
        # Görseli, log metnini, alternatif doku listesini ve sistem hafızasını (State) arayüze döndürür.
        return final_image, full_log, gr.update(choices=choices, value=choices[0], visible=True), gr.update(visible=True), state_data

    except Exception as e:
        # Kritik bir göçme anında hatayı yakalar ve arayüze metin olarak fırlatır, arayüz elemanlarını gizler.
        return None, f"HATA: {str(e)}", gr.update(visible=False), gr.update(visible=False), None

def process_alternative_generation(state_data, selected_brush, synthesis_mode, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider):
    """
    Neden Eklendi: Kullanıcı ilk sentezlenen eserin dokusunu (style) beğenmezse, analizi 
    baştan yapmadan (hafızadaki determinantları kullanarak) sadece stil referansını değiştirip otonom yeniden üretim yapar.
    """
    try:
        # Hafıza yoksa veya fırça seçilmemişse işlemi iptal eder.
        if not state_data or not selected_brush: return gr.update(), gr.update(), state_data
        # Seçilen fırça halihazırda ekrandakiyle aynıysa işlem yapmadan geri döner.
        if state_data.get("active_brush") == selected_brush: return gr.update(), gr.update(), state_data
            
        # Yeni fırçayı (stili) hafızaya aktif olarak kaydeder.
        state_data["active_brush"] = selected_brush 

        # Sistem hafızasından (State) daha önce analiz edilmiş olan fiziksel ses verilerini geri çağırır.
        bpm, timbre, hue_val, loudness, akim_karari, top_3 = state_data["bpm"], state_data["timbre"], state_data["hue_val"], state_data["loudness"], state_data["akim"], state_data["top_3"]
        
        # Kullanıcının seçtiği yeni isme göre, top_3 listesindeki dosya yolunu (path) bulur.
        best_path = next(path for path, name in top_3 if name == selected_brush)
        
        # Yeni stil referansını okur, frekans rengine senkronize eder ve Voronoi iskeletini tekrar çizer.
        raw_img = safe_image_read(best_path)
        style_img = recolor_image_to_music_hue(raw_img, hue_val)
        structural_canvas = generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness)
        
        style_name = selected_brush.split(": ")[-1]
        
        # Arayüzdeki string ifadeyi AI modelinin anlayacağı İngilizce prompta haritalar.
        concept_map = {
            "Yok (Saf Soyut)": None, 
            "İnsan Yüzü": "A dramatic close-up of a human face", 
            "Göz": "A highly detailed close-up of a human eye", 
            "Kedi": "A majestic cat silhouette",
            "Uçan Kuş": "A flying bird with spread wings", 
            "Yaşlı Ağaç": "An ancient solitary tree with sprawling branches", 
            "Çiçek (Lotus)": "A blooming lotus flower"
        }

        is_pro = "Pro Mod" in synthesis_mode

        # Pro veya Light Mod seçimine göre yukarıdaki process_primary_generation ile aynı sentez adımlarını işletir.
        if is_pro:
            target_en = concept_map.get(target_concept_tr, None)
            final_image, status = synthesize_fusion_art(structural_canvas, style_img, is_music_mode=True, target_concept=target_en, style_name=style_name, loudness=loudness, intensity_slider=intensity_slider)
            
            state_data["pro_base_output"] = final_image.copy() if final_image is not None else None
            state_data["raw_light_output"] = None
            engine_used = "LCM Turbo + MiDaS Depth"
            hist_log = status.split("\n")[-1] if "\n" in status else ""
            status = status.split("\n")[0]
        else:
            raw_final_image, status = synthesize_nst_art(structural_canvas, style_img, style_weight=1.0)
            
            state_data["raw_light_output"] = raw_final_image.copy() if raw_final_image is not None else None
            state_data["pro_base_output"] = None
            engine_used = "Neural Style Transfer"
            final_image, hist_log = apply_adaptive_histogram_bending(raw_final_image, loudness, intensity_slider)

        if final_image is not None:
            final_image, spatial_log = apply_timbre_driven_convolution(final_image, timbre, spatial_slider)
            final_image, kmeans_log = apply_kmeans_color_quantization(final_image, kmeans_slider)
            
            final_image = enforce_1080p_quality(final_image)
            
            log_training_data(final_image, state_data, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider)
        else:
            spatial_log = "Konvolusyon Pas Gecildi."
            kmeans_log = "K-Means Pas Gecildi."

        # Yeni üretim sürecine ait log raporunu günceller ve arayüze yansıtır.
        base_log = generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro)
        state_data["base_log"] = base_log 
        full_log = base_log + f"\n3. MATEMATIKSEL MODIFIKASYON (ALGORITMA RAPORU)\n|- {hist_log}\n|- {spatial_log}\n|- {kmeans_log}"
        
        return final_image, full_log, state_data
    except Exception as e:
        return None, f"SENTEZ HATASI: {str(e)}", state_data


def apply_all_post_processing(state_data, intensity_slider, spatial_slider, kmeans_slider):
    """
    Neden Eklendi: Arayüzdeki kaydırıcılar (slider) değiştirildiğinde bütün AI motorunu (GPU) 
    baştan çalıştırmamak için, hafızada (state_data) tutulan saf üretim matrisi üzerinde 
    canlı (live) matematiksel filtreleme (Post-Processing) yapar.
    """
    # Hafıza boşsa işlemi pas geçer.
    if not state_data: return gr.update(), gr.update()
    
    log_additions = []
    
    # Orijinal saf üretimi, seçilen motor tipine göre hafızadan çeker.
    if state_data.get("raw_light_output") is not None:
        # Işık şiddetini Gamma bükücü ile değiştirir.
        current_image, hist_log = apply_adaptive_histogram_bending(state_data["raw_light_output"], state_data["loudness"], intensity_slider)
        log_additions.append(hist_log)
    else:
        current_image = state_data.get("pro_base_output")
        if current_image is None: return gr.update(), gr.update()
        
    # Çıkan resmin üzerine sırasıyla Konvolüsyon (Keskinlik) ve K-Means (Renk Kümeleme) filtrelerini uygular.
    img_conv, spatial_log = apply_timbre_driven_convolution(current_image, state_data["timbre"], spatial_slider)
    log_additions.append(spatial_log)
    
    final_image, kmeans_log = apply_kmeans_color_quantization(img_conv, kmeans_slider)
    log_additions.append(kmeans_log)
    
    # Filtrelenmiş resmi son olarak 1080p kalitesine zorlar.
    final_image = enforce_1080p_quality(final_image)
    
    # Filtre loglarını birleştirerek ekrandaki analiz raporunu günceller.
    new_log_text = "\n|- ".join(log_additions)
    final_log = state_data["base_log"] + f"\n3. MATEMATIKSEL MODIFIKASYON (CANLI)\n|- {new_log_text}"
    
    return final_image, gr.update(value=final_log)


# =====================================================================
# [ARAYÜZ KONTROL YAPILARI]
# Modüllerin birbirleriyle senkronize çalışması için yazılan Gradio Olay Tetikleyicileri (Event Listeners)
# =====================================================================
def initialize_auto_scan(content_img, max_scan, top_k):
    # Fotoğraf yüklenmediyse kullanıcıyı uyarır.
    if content_img is None: return gr.update(value=[]), [], "Lutfen once fotograf yukleyin."
    
    # Yüklenen fotoğrafın RGB uzayındaki zemin ve genel renk analizini yapar.
    zemin_rengi, _, _ = analyze_color_context(content_img)
    
    # Zemin rengine göre if-else karar mekanizmasıyla en uygun sanatsal akımı atar.
    if zemin_rengi in ["Kırmızı", "Turuncu", "Sarı"]: akim = "Sentetik Kubizm (Genis Renk Yuzeyleri)"
    elif zemin_rengi in ["Mavi", "Mor", "Yeşil"]: akim = "Analitik Kubizm (Yuksek Form Dominansi)"
    else: akim = "Proto-Kubizm (Sadelestirilmis Dogrusal Indeks)"
    
    # Belirlenen akım ve tarama limitlerine göre veritabanından en uygun eserleri (Top-K) çeker.
    top_results = get_ranked_top_10_artworks(content_image=content_img, akim_karari=akim, max_scan=max_scan, top_k=top_k)
    
    # Eserleri arayüzdeki galeriye yansıtır ve log bilgisini döner.
    return gr.update(value=top_results), top_results, f"Bulanik Mantik taramasi tamamlandi. Secilen alt kume: {top_k} eser."

def toggle_auto_mode(is_auto):
    # Otomatik Mod Checkbox'ına tıklandığında, arayüzdeki manuel seçim panellerini gizler, tarama panelini açar (veya tam tersi).
    if is_auto: return gr.update(visible=False), gr.update(visible=True)
    else: return gr.update(visible=True), gr.update(visible=False)

def save_gallery_selection(evt: gr.SelectData):
    # Kullanıcı AI taramasından dönen galerideki bir resme tıkladığında, o resmin indeksini (sırasını) hafızaya alır.
    return evt.index, gr.update(interactive=True)

def live_image_post_processing(img_state, sharpness, saturation, contrast):
    """
    Neden Eklendi: Görselin frekans uzayında işlenerek (CLAHE, Unsharp Mask, Vibrance) 
    fotoğraf manipülasyonunun son kullanıcı tarafından anlık ve dinamik olarak yapılabilmesi için.
    """
    if img_state is None:
        return gr.update()
    
    # Orijinal matrisi bozmamak için bir kopyasını oluşturur.
    img = img_state.copy()
    
    # Eğer doygunluk (saturation) kaydırıcısı değiştirildiyse:
    if saturation != 1.0:
        # Rengi doygunluk kanalını izole edebilmek için RGB'den HSV uzayına çevirir.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        # S (Saturation) kanalını katsayı ile çarpar ve 0-255 sınırını aşmamasını sağlar (np.clip).
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        # Değiştirilmiş matrisi yeniden RGB uzayına döndürür.
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    # Eğer keskinlik (sharpness) kaydırıcısı değiştirildiyse:
    if sharpness > 0.0:
        # Gaussian filtresiyle görselin mikro-bulanık bir versiyonunu çıkarır.
        blur = cv2.GaussianBlur(img, (0, 0), 3.0)
        # Bulanık versiyonu orijinal versiyondan çıkararak (Unsharp Masking) piksellerdeki kenar (edge) geçişlerini sertleştirir.
        img = cv2.addWeighted(img, 1.0 + (sharpness / 2.0), blur, -(sharpness / 2.0), 0)
        
    # Eğer kontrast kaydırıcısı değiştirildiyse:
    if contrast > 1.0:
        # Işığı ayrıştırmak için RGB'den LAB (Lightness-A-B) uzayına çevirir.
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) algoritmasıyla sadece Lightness (L) kanalının kontrastını bölgesel olarak artırır.
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
        cl = clahe.apply(l)
        # İşlenmiş L kanalıyla renk kanallarını (A ve B) geri birleştirip RGB'ye döner.
        img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        
    return img

def process_image_to_image(content_image, is_auto, artist, artwork, top_10_state, selected_idx, synthesis_mode, style_weight):
    # 2. Sekme olan "Fotoğrafa Sanat İşle" kısmının ana işlem bloğudur.
    try:
        if content_image is None: return None, "HATA: Gorsel veri saptanamadi.", None
        
        # Yüklenen fotoğrafın yapısal özelliklerini (Işık ve Kenar yoğunluğu) hesaplar.
        user_lum, user_edge = calculate_image_features(content_image)
        # Renk analizini yapar.
        zemin_rengi, tamamlayici, etki = analyze_color_context(content_image)
        
        # Kullanıcı Otomatik AI Taramasını seçtiyse:
        if is_auto:
            # Galeriden seçim yapılmadıysa hata döndürür.
            if selected_idx is None: return None, "Gorsel secimi zorunludur.", None
            # Galeride tıklanan (indeksi alınan) görselin disk yolunu çekip matrise dönüştürür.
            style_img = safe_image_read(top_10_state[selected_idx][0])
            style_name = top_10_state[selected_idx][1]
            secim_yontemi = "AI Bulanik Mantik Taramasi"
        # Kullanıcı Manuel Seçimi tercih ettiyse:
        else:
            if not artist or not artwork: return None, "Sanatci ve Eser parametreleri zorunludur.", None
            # Dropdown (Açılır menü) üzerinden seçilen sanatçının eserini çeker.
            style_img = get_artwork_image(artist, artwork)
            style_name = f"{artist} - {artwork}"
            secim_yontemi = "Manuel Determinasyon"

        # Pro Mod (LCM Turbo) motoru seçildiyse:
        if "Pro Mod" in synthesis_mode:
            # Fotoğrafı, seçilen eserle birlikte Depth-to-Image difüzyon hattına (pipeline) sokar.
            final_image, status = synthesize_fusion_art(content_image, style_img, is_music_mode=False, style_name=style_name, style_weight=style_weight)
            engine_used = "LCM Turbo + MiDaS Depth"
        # Light Mod (NST) motoru seçildiyse:
        else:
            # Geleneksel nöral ağ üzerinden stil matrisini içeriğe geçirir.
            final_image, status = synthesize_nst_art(content_image, style_img, style_weight=style_weight)
            engine_used = "Neural Style Transfer Modulu"

        # Arayüze basılacak analiz raporunu (log) hazırlar.
        log_text = f"SOLENTA GORSEL SENTEZ RAPORU\n\n"
        log_text += f"1. GORSEL ALGI\n|- Isik (Luminance): {user_lum:.2f}\n|- Kenar (Edge): {user_edge:.2f}\n|- Zemin: {zemin_rengi}\n\n"
        log_text += f"2. REFERANS\n|- Karar Mekanizmasi: {secim_yontemi}\n|- Eser: {style_name}\n\n"
        log_text += f"3. SENTEZ\n|- Cikti Cozunurlugu: 1080p (Lanczos-4)\n|- Motor : {engine_used}\n|- Agirlik: %{int(style_weight * 100)}\n|- Durum: {status}"

        # İşlenmiş matrisi son olarak 1080p boyutlarına kilitler.
        final_image = enforce_1080p_quality(final_image)
        # Canlı filtreleme (sliderlar) kullanılabilmesi için saf çıktıyı State (hafıza) değişkenine kopyalar.
        final_state = final_image.copy() if final_image is not None else None
        
        # Görseli, logu ve hafızayı arayüze döndürür.
        return final_image, log_text, final_state
    
    except Exception as e:
        return None, f"HATA: {str(e)}", None

def handle_preset_selection(preset_name):
    # Kullanıcı hazır listeden şarkı seçtiyse, özel dosya yükleme alanını arayüzden gizler.
    if preset_name:
        return None, gr.update(visible=False, value=0)
    return gr.update(), gr.update(visible=True)

def handle_user_upload(file_path):
    """
    Neden Eklendi: Kullanıcı bir ses dosyası yüklediğinde, şarkının neresinden işlem 
    yapılacağını otonom olarak (insan seçimine bırakmadan) bulmak içindir. Sesin 
    spektral enerjisinin (Onset) en yüksek olduğu saniyeyi 'Drop / Kreşendo' noktası 
    olarak tespit edip arayüze basar.
    """
    if file_path:
        optimal_start = find_optimal_audio_segment(file_path)
        return None, gr.update(visible=True, value=optimal_start)
    return gr.update(), gr.update(visible=True, value=0)

# =====================================================================
# [ARAYÜZ TASARIMI VE CSS]
# Uygulamanın son kullanıcı (Frontend) mimarisini resmeder.
# CSS blokları, Gradio'nun standart beyaz/gri temasını tamamen Karanlık Mod'a (Dark Mode) dönüştürür.
# Neden: Dijital sanat üretiminde parlak zemin renkleri, kullanıcının "Eşzamanlı Kontrast" algısını bozar.
# =====================================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
.gradio-container { font-family: 'Montserrat', sans-serif !important; }
h1 { color: #ff6b00 !important; text-transform: uppercase; letter-spacing: 2px; text-align: center; }
textarea { background-color: #050505 !important; color: #ff8c00 !important; font-family: 'JetBrains Mono', monospace !important; border-radius: 8px !important; overflow-y: auto !important; max-height: 380px !important;}
button.primary { background: linear-gradient(135deg, #ff4500, #ff8c00) !important; border: none !important; color: white !important; font-weight: 800 !important; letter-spacing: 1px !important;}
button.secondary { background: #2b2b2b !important; border: 1px solid #ff8c00 !important; color: #ff8c00 !important; font-weight: 600 !important;}
footer { display: none !important; visibility: hidden !important; opacity: 0 !important; }
.gradio-container-4-36-1 a.svelte-1ijrhy9 { display: none !important; }
.radio-group { padding: 10px; border-radius: 8px; border-left: 4px solid #ff6b00; margin-bottom: 15px;}
.scan-btn { background: #2b2b2b !important; color: #ff8c00 !important; border: 1px solid #ff8c00 !important; }
.alt-box { padding: 15px; border-radius: 8px; background-color: #1a1a1a; margin-top: 15px;}
"""

# Arayüz inşa iskeletini (Blocks) başlatır.
with gr.Blocks(css=custom_css, title="Solenta Laboratuvari") as solenta_app:
    gr.Markdown("# SOLENTA PROJE LABORATUVARI")
    gr.Markdown("<br>")

    # Ana sekmeleri (Tabs) oluşturur.
    with gr.Tabs():
        # -------- BİRİNCİ SEKME: MÜZİKTEN GÖRSEL ÜRET --------
        with gr.TabItem("Muzikten Gorsel Uret (Generative)"):
            with gr.Row():
                # Sol Sütun: Girdi Parametreleri
                with gr.Column(scale=1):
                    gr.Markdown("### ISITSEL GIRDI")
                    audio_mode_radio = gr.Radio(choices=["Light Mod (NST)", "Pro Mod (LCM Turbo)"], value="Light Mod (NST)", label="Motor Secimi", elem_classes="radio-group")
                    
                    target_concept_dropdown = gr.Dropdown(choices=["Yok (Saf Soyut)", "İnsan Yüzü", "Göz", "Kedi", "Uçan Kuş", "Yaşlı Ağaç", "Çiçek (Lotus)"], value="Yok (Saf Soyut)", label="Opsiyonel Odak Nesnesi (Focal Point)", visible=False)
                    
                    audio_input = gr.Audio(type="filepath", label="Muzik Yukle (Kendi Dosyaniz)")
                    preset_audio = gr.Dropdown(choices=get_preset_audio_list(), label="Veya Veritabanindan Muzik Secin") 
                    
                    start_time_input = gr.Number(value=0, label="Baslangic Saniyesi (Smart Crop Tespiti)", precision=0)
                    
                    with gr.Group():
                        gr.Markdown("#### Sinyal-Piksel Bukucu Algoritmalar")
                        intensity_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Adaptif Histogram Egrisi (Isik Bukucu)")
                        spatial_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Mekansal Konvolusyon (Matris)")
                        kmeans_slider = gr.Slider(minimum=2, maximum=65, value=65, step=1, label="Vektorel Renk Kumeleme (K-Means)")
                    
                    btn_audio = gr.Button("Sentez Islemini Baslat", variant="primary")
                    audio_state = gr.State()
                    
                # Sağ Sütun: Çıktı ve Raporlar
                with gr.Column(scale=1):
                    gr.Markdown("### SENTETIK CIKTI")
                    audio_output_img = gr.Image(label="Sentezlenen Laboratuvar Ciktisi")
                    audio_log = gr.Textbox(label="Uretim Sureci Analizi", lines=15)
                    
                    # İlk üretimden sonra görünür hale gelecek olan Alternatif Patern (Fırça) paneli
                    with gr.Column(visible=False, elem_classes="alt-box") as alt_column:
                        gr.Markdown("#### Alternatif Doku Uretimi")
                        brush_options = gr.Dropdown(label="Veritabanindan farkli bir patern seciniz:", choices=[], interactive=True)
                        
            # Olay Tetikleyicileri (Event Listeners) - Bileşenler arası etkileşimi sağlar
            preset_audio.change(fn=handle_preset_selection, inputs=[preset_audio], outputs=[audio_input, start_time_input])
            audio_input.change(fn=handle_user_upload, inputs=[audio_input], outputs=[preset_audio, start_time_input])
            audio_mode_radio.change(fn=lambda mode: gr.update(visible="Pro Mod" in mode), inputs=[audio_mode_radio], outputs=[target_concept_dropdown], show_progress="hidden")
            
            # Asıl sentez işlemini başlatan tetikleyici (Buton Tıklaması)
            btn_audio.click(fn=process_primary_generation, inputs=[audio_input, preset_audio, start_time_input, audio_mode_radio, target_concept_dropdown, intensity_slider, spatial_slider, kmeans_slider], outputs=[audio_output_img, audio_log, brush_options, alt_column, audio_state])
            # Alternatif fırça seçimi değiştiğinde yeniden sentezleyen tetikleyici
            brush_options.change(fn=process_alternative_generation, inputs=[audio_state, brush_options, audio_mode_radio, target_concept_dropdown, intensity_slider, spatial_slider, kmeans_slider], outputs=[audio_output_img, audio_log, audio_state])
            
            # Slider'lar serbest bırakıldığında (release) canlı filtrelemeyi tetikler
            intensity_slider.release(fn=apply_all_post_processing, inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider], outputs=[audio_output_img, audio_log], show_progress="hidden")
            spatial_slider.release(fn=apply_all_post_processing, inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider], outputs=[audio_output_img, audio_log], show_progress="hidden")
            kmeans_slider.release(fn=apply_all_post_processing, inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider], outputs=[audio_output_img, audio_log], show_progress="hidden")

        # -------- İKİNCİ SEKME: FOTOĞRAFA SANAT İŞLE --------
        with gr.TabItem("Fotografa Sanat Isle (Stylization)"):
            with gr.Row():
                # Sol Sütun: Girdi Parametreleri
                with gr.Column(scale=1):
                    gr.Markdown("### GORSEL GIRDI")
                    img_mode_radio = gr.Radio(choices=["Light Mod (NST)", "Pro Mod (LCM Turbo)"], value="Light Mod (NST)", label="Motor Secimi", elem_classes="radio-group")
                    img_content = gr.Image(type="numpy", label="Ham Gorsel Verisi")
                    
                    with gr.Group():
                        gr.Markdown("#### Makine Ogrenmesi Parametreleri")
                        style_weight_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Stil Yogunlugu (Interpolasyon)")

                    with gr.Group():
                        auto_mode_check = gr.Checkbox(label="Otonom Tarama Protokolu")
                        
                        with gr.Column(visible=True) as manual_selection_col:
                            artist_dropdown = gr.Dropdown(choices=get_all_artists(), label="Sanatci Belirle")
                            artwork_dropdown = gr.Dropdown(choices=[], label="Eser Belirle")
                        
                        with gr.Column(visible=False) as auto_selection_col:
                            gr.Markdown("##### Veritabani Tarama Parametreleri")
                            max_scan_slider = gr.Slider(minimum=500, maximum=2127, value=1000, step=1, label="Taranacak Maksimum Eser")
                            top_k_slider = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Dondurulecek Alt Kume (Top-K)")
                            
                            btn_scan = gr.Button("Sorguyu Baslat", elem_classes="scan-btn")
                            top_10_gallery = gr.Gallery(label="Analiz Sonuclari", columns=5, height="auto", allow_preview=False)
                            top_10_state = gr.State([])
                            selected_gallery_idx = gr.State(None)
                    
                    btn_img = gr.Button("Gorseli Isleyerek Sentezle", variant="primary", interactive=True)
                    
                # Sağ Sütun: Çıktı ve Raporlar
                with gr.Column(scale=1):
                    gr.Markdown("### STILIZE CIKTI")
                    img_output = gr.Image(label="Manipule Edilmis Nihai Veri")
                    img_state = gr.State() 
                    
                    with gr.Group():
                        gr.Markdown("#### Dinamik Iyilestirme")
                        sharpness_slider = gr.Slider(minimum=0.0, maximum=3.0, value=0.0, step=0.1, label="Yapisal Keskinlik (Unsharp Mask)")
                        saturation_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Renk Doygunlugu (Vibrance)")
                        contrast_slider = gr.Slider(minimum=1.0, maximum=4.0, value=1.0, step=0.1, label="Lokal Kontrast (CLAHE)")

                    img_log = gr.Textbox(label="Sentez Raporu", lines=10)

            # Olay Tetikleyicileri (Event Listeners)
            artist_dropdown.change(fn=lambda x: gr.update(choices=get_artworks_by_artist(x), value=None), inputs=artist_dropdown, outputs=artwork_dropdown, show_progress="hidden")
            auto_mode_check.change(fn=toggle_auto_mode, inputs=[auto_mode_check], outputs=[manual_selection_col, auto_selection_col], show_progress="hidden")
            
            btn_scan.click(
                fn=initialize_auto_scan, 
                inputs=[img_content, max_scan_slider, top_k_slider], 
                outputs=[top_10_gallery, top_10_state, img_log]
            )
            
            top_10_gallery.select(fn=save_gallery_selection, outputs=[selected_gallery_idx, btn_img])
            
            # Asıl sentez işlemini başlatan tetikleyici
            btn_img.click(
                fn=process_image_to_image, 
                inputs=[img_content, auto_mode_check, artist_dropdown, artwork_dropdown, top_10_state, selected_gallery_idx, img_mode_radio, style_weight_slider], 
                outputs=[img_output, img_log, img_state]
            )

            style_weight_slider.release(
                fn=process_image_to_image, 
                inputs=[img_content, auto_mode_check, artist_dropdown, artwork_dropdown, top_10_state, selected_gallery_idx, img_mode_radio, style_weight_slider], 
                outputs=[img_output, img_log, img_state],
                show_progress="hidden"
            )
            
            sharpness_slider.release(fn=live_image_post_processing, inputs=[img_state, sharpness_slider, saturation_slider, contrast_slider], outputs=[img_output], show_progress="hidden")
            saturation_slider.release(fn=live_image_post_processing, inputs=[img_state, sharpness_slider, saturation_slider, contrast_slider], outputs=[img_output], show_progress="hidden")
            contrast_slider.release(fn=live_image_post_processing, inputs=[img_state, sharpness_slider, saturation_slider, contrast_slider], outputs=[img_output], show_progress="hidden")

    # =====================================================================
    # [KRITIK GUVENLIK] Isletim Sistemi Duzeyinde Kapatma Protokolu
    # Neden Eklendi: Web tarayicisi kapatildiginda arka planda asili kalan 
    # Python portlarini ve GPU VRAM alanini serbest birakarak bellek tasmalarini (leak) onler.
    # =====================================================================
    gr.Markdown("---")
    def kill_system():
        # os._exit komutu programi derhal, acimasizca ve kalici olarak sistem belleginden siler.
        os._exit(0)
    btn_kill = gr.Button("SISTEMI GUVENLI KAPAT (Arka Plani Temizler)", variant="primary")
    btn_kill.click(fn=kill_system, inputs=None, outputs=None)

# Ana Calistirma Blogu (Entry Point)
if __name__ == "__main__":
    # Gradio web sunucusunu yerel (127.0.0.1) agda ve 7860 portunda baslatir.
    solenta_app.launch(
        inbrowser=True, # Sunucu calistiginda varsayilan tarayiciyi otomatik olarak acar.
        server_name="127.0.0.1", 
        server_port=7860,
        prevent_thread_lock=False # Paralel islem is parcaciklarinin birbirini kitlemesini onler.
    )