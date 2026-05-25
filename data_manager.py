import os
import cv2
import numpy as np
import sys  
import glob 

DATASET_PATH = "dataset"
CATEGORIES = {
    "Analitik Kübizm (Yüksek Form Dominansı)": "analitik",
    "Sentetik Kübizm (Geniş Renk Yüzeyleri)": "sentetik",
    "Proto-Kübizm (Sadeleştirilmiş Doğrusal İndeks)": "proto"
}

_ARTIST_CATALOG = None 
_FEATURES_CACHE = {} 

def safe_image_read(img_path):
    try:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bozuk veya okunamayan görsel formatı.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[VERİ AMBARI HATASI]: {img_path} okunamadı -> {e}")
        return np.ones((512, 512, 3), dtype=np.uint8) * 128

def _get_or_build_catalog():
    global _ARTIST_CATALOG
    if _ARTIST_CATALOG is not None: return _ARTIST_CATALOG
    catalog = {}
    if not os.path.exists(DATASET_PATH): return catalog
        
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                clean_name = os.path.splitext(file)[0]
                if '_' in clean_name:
                    artist = clean_name.split('_')[0].strip().title()
                    artwork = clean_name.split('_', 1)[1].replace('_', ' ').title()
                elif '-' in clean_name:
                    artist = clean_name.split('-')[0].strip().title()
                    artwork = clean_name.split('-', 1)[1].replace('-', ' ').title()
                else:
                    artist = "Diğer Sanatçılar"
                    artwork = clean_name.title()
                    
                full_path = os.path.join(root, file)
                if artist not in catalog: catalog[artist] = {}
                catalog[artist][artwork] = full_path
                
    _ARTIST_CATALOG = catalog
    return _ARTIST_CATALOG

def get_all_artists():
    catalog = _get_or_build_catalog()
    artists = list(catalog.keys())
    artists.sort()
    return artists

def get_artworks_by_artist(artist_name):
    catalog = _get_or_build_catalog()
    if artist_name in catalog:
        artworks = list(catalog[artist_name].keys())
        artworks.sort()
        return artworks
    return []

def get_artwork_image(artist_name, artwork_name):
    catalog = _get_or_build_catalog()
    if artist_name in catalog and artwork_name in catalog[artist_name]:
        path = catalog[artist_name][artwork_name]
        return safe_image_read(path)
    return None

def calculate_image_features(img_matrix):
    small_img = cv2.resize(img_matrix, (100, 100), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    luminance = np.mean(l_channel)
    gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0 
    return luminance, edge_density

def calculate_image_timbre_surrogate(img_matrix):
    small_img = cv2.resize(img_matrix, (100, 100), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    visual_timbre = np.mean(magnitude_spectrum) * 100 
    return visual_timbre

# [GÜNCELLENDİ]: Dinamik Havuz ve Top-K Parametreleri
def get_ranked_top_10_artworks(content_image, akim_karari, max_scan=2127, top_k=10):
    global _FEATURES_CACHE
    category_folder = CATEGORIES.get(akim_karari, "proto")
    target_path = os.path.join(DATASET_PATH, category_folder)

    if not os.path.exists(target_path): return []
    images = [f for f in os.listdir(target_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images: return []

    images.sort()
    seed = abs(hash(akim_karari)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    
    # Kullanıcının belirlediği havuza göre tarama yapar
    pool_size = min(max_scan, len(images)) 
    candidate_images = rng.choice(images, pool_size, replace=False).tolist() if len(images) > pool_size else images

    user_lum, user_edge = calculate_image_features(content_image)
    scores = []
    print(f"\n[SENTA SİSTEMİ]: Bulanık Mantık Eşleştirmesi Başladı ({pool_size} aday eser taranıyor...)")
    
    total_candidates = len(candidate_images)
    update_interval = max(1, total_candidates // 10)

    for i, img_name in enumerate(candidate_images):
        if (i + 1) % update_interval == 0 or (i + 1) == total_candidates:
            percent = int(((i + 1) / total_candidates) * 100)
            sys.stdout.write(f"\r\t> Algısal Çıkarım %{percent} tamamlandı... ({i + 1} / {total_candidates} eser tarandı)")
            sys.stdout.flush()

        img_path = os.path.join(target_path, img_name)
        if img_path not in _FEATURES_CACHE:
            try:
                img_mat = safe_image_read(img_path)
                _FEATURES_CACHE[img_path] = calculate_image_features(img_mat)
            except Exception:
                _FEATURES_CACHE[img_path] = (128.0, 0.5) 
        
        style_lum, style_edge = _FEATURES_CACHE[img_path]
        lum_diff = abs(user_lum - style_lum) / 255.0
        edge_diff = abs(user_edge - style_edge)
        scores.append((lum_diff + edge_diff, img_path, img_name))
        
    print(f"\n[SENTA SİSTEMİ]: Tarama Tamamlandı. En iyi {top_k} eser UI'a aktarılıyor.")
    scores.sort(key=lambda x: x[0])
    
    results = []
    # Dinamik Top-K
    for rank, (score, path, img_name) in enumerate(scores[:int(top_k)], start=1):
        clean_name = os.path.splitext(img_name)[0].replace('_', ' ').replace('-', ' ').title()
        display_name = f"{rank}. Eşleşme: {clean_name}"
        results.append((path, display_name))
    return results

def recolor_image_to_music_hue(img_matrix, target_hue_deg):
    if img_matrix is None: return np.ones((512, 512, 3), dtype=np.uint8) * 128
    hsv_image = cv2.cvtColor(img_matrix, cv2.COLOR_RGB2HSV).astype(np.float32)
    opencv_hue = int(target_hue_deg / 2.0) % 180
    hsv_image[:, :, 0] = opencv_hue
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.2, 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

def get_top_3_artworks_from_audio(timbre_centroid, akim_karari, target_concept="Yok (Saf Soyut)"):
    global _FEATURES_CACHE
    category_folder = CATEGORIES.get(akim_karari, "proto")
    target_path = os.path.join(DATASET_PATH, category_folder)

    if not os.path.exists(target_path): return []
    images = [f for f in os.listdir(target_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images: return []

    images.sort()
    seed = abs(hash(str(timbre_centroid) + akim_karari + str(target_concept))) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    
    pool_size = min(150, len(images))
    candidate_images = rng.choice(images, pool_size, replace=False).tolist() if len(images) > pool_size else images

    scores = []
    print(f"\n[SENTA İŞİTSEL MOTOR]: Müzik Tınısı ({timbre_centroid} Hz) Eşleştiriliyor...")

    for i, img_name in enumerate(candidate_images):
        img_path = os.path.join(target_path, img_name)
        cache_key = img_path + "_audio" 
        
        if cache_key not in _FEATURES_CACHE:
            try:
                img_mat = safe_image_read(img_path)
                _FEATURES_CACHE[cache_key] = calculate_image_timbre_surrogate(img_mat)
            except:
                _FEATURES_CACHE[cache_key] = 1000.0 
                
        style_timbre = _FEATURES_CACHE[cache_key]
        timbre_diff = abs(timbre_centroid - style_timbre)
        scores.append((timbre_diff, img_path, img_name))
        
    scores.sort(key=lambda x: x[0])
    
    top_3_results = []
    for rank, match in enumerate(scores[:3], start=1):
        path = match[1]
        img_name = match[2]
        clean_name = os.path.splitext(img_name)[0].replace('_', ' ').replace('-', ' ').title()
        top_3_results.append((path, f"Stil {rank}: {clean_name}"))
        
    print(f"[SENTA İŞİTSEL MOTOR]: En Uyumlu 3 Eser (Fırça Seçeneği) Bulundu.")
    return top_3_results

def get_preset_audio_list():
    audio_dir = os.path.join("dataset", "audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        return []
        
    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3")) + glob.glob(os.path.join(audio_dir, "*.wav"))
    dropdown_choices = []
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        clean_name = os.path.splitext(file_name)[0].replace("_", " ")
        dropdown_choices.append((clean_name, file_path))
    return dropdown_choices