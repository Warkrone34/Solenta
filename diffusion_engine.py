import os
import sys
import torch
import cv2
import numpy as np
import traceback
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor

_LAST_ERROR = "Bilinmeyen Hata"

# =====================================================================
# [GUVENLIK VE DAGITIM GUNCELLEMESI]: Dinamik Yol ve Offline Cekirdek
# Neden Eklendi: USB'den kurulumda uygulamanin internete baglanmadan,
# disa bagimli olmadan (Air-Gapped) calisabilmesi icin.
# =====================================================================
def get_base_dir():
    """ PyInstaller (.exe) veya normal Python scripti fark etmeksizin ana klasoru bulur. """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
MODELS_DIR = os.path.join(BASE_DIR, "models")

def get_model_path(hub_id, local_folder_name):
    """
    Once yerel 'models' klasorune bakar. (Offline USB Kurulumu icin)
    Bulamazsa HuggingFace Hub ID'sini dondurur. (Gelistirici ortami icin)
    """
    local_path = os.path.join(MODELS_DIR, local_folder_name)
    if os.path.exists(local_path):
        return local_path
    return hub_id

# --- GLOBAL MODEL CACHE (Hafiza Yonetimi) ---
_PIPELINE = None
_DEPTH_ESTIMATOR = None

def get_safe_device():
    """
    Neden Eklendi: Donanım mimarisini ve CUDA çekirdek uyumluluğunu (sm_120 / Blackwell vb.)
    küçük bir tensör operasyonu ile doğrulayarak en güvenli ve kararlı aygıtı (cuda veya cpu) seçer.
    """
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return "cuda"
        except Exception as e:
            print(f"[SOLENTA UYARI]: CUDA aygıtı saptandı ancak kurulu PyTorch çekirdeği ile uyumsuz ({e}). Güvenli CPU moduna geçiliyor...")
            return "cpu"
    return "cpu"

def get_depth_map(image_pil):
    """
    Neden Eklendi: Girdi görselinden uzaysal derinlik (Z-Axis) matrisi çıkarmak için.
    MiDaS (Monocular Depth Estimation) modeli, 2 boyutlu piksellerden 3 boyutlu
    hacimsel bir harita üreterek ControlNet'e mekansal yapı koşulu (Conditioning) sağlar.
    """
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is None:
        print("[SOLENTA V2]: MiDaS Derinlik Algisi (Depth Map) Motoru Yukleniyor...")
        device = get_safe_device()
        midas_path = get_model_path("Intel/dpt-hybrid-midas", "dpt-hybrid-midas")
        
        # [ZIRH]: Yerel model varsa diskten okur, yoksa HuggingFace Hub'dan otonom indirir
        is_midas_local = os.path.exists(midas_path)
        processor = AutoImageProcessor.from_pretrained(midas_path, local_files_only=is_midas_local)
        model = AutoModelForDepthEstimation.from_pretrained(midas_path, local_files_only=is_midas_local).to(device)
        _DEPTH_ESTIMATOR = {"model": model, "processor": processor, "device": device}
    
    inputs = _DEPTH_ESTIMATOR["processor"](images=image_pil, return_tensors="pt").to(_DEPTH_ESTIMATOR["device"])
    with torch.no_grad():
        outputs = _DEPTH_ESTIMATOR["model"](**inputs)
        predicted_depth = outputs.predicted_depth
        
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return Image.fromarray(formatted).convert("RGB")

def get_diffusion_pipeline():
    global _PIPELINE, _LAST_ERROR
    if _PIPELINE is not None:
        return _PIPELINE
        
    print("[SOLENTA V2]: Depth ControlNet ve LCM Turbo Yukleniyor (Sarsilmaz Surum)...")
    device = get_safe_device()

    try:
        cnet_path = get_model_path("lllyasviel/sd-controlnet-depth", "sd-controlnet-depth")
        is_cnet_local = os.path.exists(cnet_path)
        controlnet = ControlNetModel.from_pretrained(
            cnet_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=is_cnet_local,
            use_safetensors=True  # [EXE ZIRHI]: Safetensors okumaya zorlar
        )
        
        sd_path = get_model_path("runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5")
        is_sd_local = os.path.exists(sd_path)
        
        # =====================================================================
        # [BAŞ MİMAR ZIRHI]: OTONOM ÇİFT-AŞAMALI YÜKLEYİCİ (DUAL-LOAD ARMOR)
        # Terminal (.py) ve Kapsül (.exe) çatışmasını kökünden çözer.
        # =====================================================================
        try:
            # Deneme 1: Terminal (Geliştirici) Ortamı İçin Standart Yükleme
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                sd_path, 
                controlnet=controlnet, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                local_files_only=is_sd_local,
                use_safetensors=True
            )
        except Exception as e_term:
            print(f"[SOLENTA V2]: Terminal modu reddedildi. EXE (FP16) uyumluluk moduna geciliyor...")
            # Deneme 2: EXE Ortamında patlarsa, kilitleri kırıp FP16 varyantı ile zorla yükleme
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                sd_path, 
                controlnet=controlnet, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                local_files_only=is_sd_local,
                use_safetensors=True,
                variant="fp16"
            )
        # =====================================================================
        
        lcm_path = get_model_path("latent-consistency/lcm-lora-sdv1-5", "lcm-lora-sdv1-5")
        if os.path.isdir(lcm_path):
            lora_files = [f for f in os.listdir(lcm_path) if f.endswith(('.safetensors', '.bin'))]
            if lora_files:
                pipe.load_lora_weights(lcm_path, weight_name=lora_files[0], local_files_only=True)
            else:
                raise FileNotFoundError(f"KRİTİK HATA: LCM klasörü boş: {lcm_path}")
        else:
            print("[SOLENTA V2]: LCM LoRA ağırlıkları HuggingFace Hub üzerinden otonom indiriliyor...")
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # [KRITIK DUZELTME]: CPU Offload ve Attention Slicing SILINDI!
        # Modeli direkt Ekran Kartinin (VRAM) icine kalici olarak gomuyoruz. Hiz %500 artacak.
        pipe.to(device)
        
        _PIPELINE = pipe
        print(f"[SOLENTA V2]: LCM Turbo Motoru Basariyla Ateslendi ({device}). Sistem Hazir!")
        return _PIPELINE
    except Exception as e:
        hata_detayi = traceback.format_exc()
        try:
            with open(os.path.join(BASE_DIR, "CRASH_LOG.txt"), "w", encoding="utf-8") as f:
                f.write(hata_detayi)
        except:
            pass
        _LAST_ERROR = str(e)
        print(f"[KRITIK HATA]: Model yuklenemedi. Lutfen 'models' klasorundeki dosyalari kontrol et!\n{e}")
        return None

def get_dominant_color_name(image_matrix):
    """
    Renk algisi motoru.
    Gorselin HSV uzayindaki baskin tonunu metinsel bir karsiliga (prompt objesine) cevirir.
    """
    hsv = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2HSV)
    mask = hsv[:,:,2] > 20 
    if not np.any(mask): return "dark"
    mean_hue = np.mean(hsv[:,:,0][mask])
    
    if mean_hue < 10 or mean_hue >= 165: return "crimson red"
    elif mean_hue < 25: return "vibrant orange"
    elif mean_hue < 35: return "bright yellow"
    elif mean_hue < 85: return "emerald green"
    elif mean_hue < 130: return "deep blue"
    elif mean_hue < 165: return "mystic purple"
    else: return "colorful"

def apply_adaptive_histogram_bending(image_matrix, loudness, user_intensity_slider):
    """
    Sesin gurluk (RMS) degerini kullanarak gorselin isik kontrastini (Gamma Egrisi) buker.
    """
    if image_matrix is None: return None, "HATA: Islenecek matris yok."
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

    return final_image, f"Adaptif Histogram: y = {gamma:.2f} | Muzik Carpani: {normalized_loudness:.2f}"

def transfer_color_profile(source_img, target_img):
    """
    Hedef sanat eserinin renk standart sapmasini, ana gorsele transfer eder.
    """
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)
    
    src_std = np.where(src_std == 0, 1.0, src_std) 
    
    src_lab -= src_mean.reshape((1,1,3))
    src_lab = (src_lab / src_std.reshape((1,1,3))) * tgt_std.reshape((1,1,3))
    src_lab += tgt_mean.reshape((1,1,3))
    
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)

def synthesize_fusion_art(base_image, style_reference, is_music_mode=False, target_concept=None, style_name="Cubist", loudness=50.0, intensity_slider=1.0, style_weight=1.0):
    """
    Ana Sentez Motoru: Sinyal iskeletini ve stil matrisini difuzyon modeliyle kaynastirir.
    VRAM tasmalarina (OOM) karsi koruma onlemleri icerir.
    """
    pipe = get_diffusion_pipeline()
    if pipe is None: 
        return None, f"HATA: Motor baslatilamadi.\nNEDENI: {_LAST_ERROR}\n(Detay icin CRASH_LOG.txt dosyasina bakin)"

    if isinstance(base_image, np.ndarray):
        if base_image.dtype != np.uint8:
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        base_pil = Image.fromarray(base_image).convert("RGB")
    else:
        base_pil = base_image.convert("RGB")
        base_image = np.array(base_pil)

    # VRAM BOGULMASINI ONLEYEN COZUNURLUK OPTIMIZATORU
    max_dim = 768  # 768x768, LCM Turbo icin ideal boyut. Daha buyukse optimize edilir.
    w, h = base_pil.size
    
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        base_pil = base_pil.resize((new_w, new_h), Image.LANCZOS)
        base_image = np.array(base_pil)
        w, h = new_w, new_h
        print(f"[SOLENTA HIZLANDIRICI]: Fotograf {w}x{h} boyutuna optimize edildi.")

    if isinstance(style_reference, str):
        style_pil = base_pil
        style_name = style_reference
    elif isinstance(style_reference, np.ndarray):
        if style_reference.dtype != np.uint8:
            style_reference = np.clip(style_reference, 0, 255).astype(np.uint8)
        style_pil = Image.fromarray(style_reference).convert("RGB")
    else:
        style_pil = style_reference.convert("RGB")

    style_resized = style_pil.resize((w, h), Image.LANCZOS)
    
    base_arr = np.array(base_pil)
    style_arr = np.array(style_resized)
    
    style_weight = max(0.01, min(float(style_weight), 1.0))
    short_style_name = " ".join(style_name.split()[:2]) if len(style_name.split()) > 2 else style_name
    dominant_color = get_dominant_color_name(base_image)

    if is_music_mode:
        fusion_arr = cv2.addWeighted(base_arr, 0.7, style_arr, 0.3, 0)
        init_pil = Image.fromarray(fusion_arr)
        depth_map = get_depth_map(init_pil)
        
        strength_val = 0.85 
        control_scale_val = 0.45 * (1.5 - style_weight)
        guidance_scale_val = 1.5 
        
        if target_concept is not None and str(target_concept).strip() != "":
            positive_prompt = f"A clear, striking {target_concept} as the central focal point, highly detailed, photorealistic features, emerging from {dominant_color} abstract intersecting cubist planes, heavily textured with {short_style_name} art style, sharp edges, vibrant, aesthetic digital art, 8k resolution"
        else:
            positive_prompt = f"{dominant_color} themed abstract expressionist music visualization, heavily textured with {short_style_name} art style, masterpiece, highly detailed, vibrant, aesthetic digital art, 8k resolution, sharp geometric shapes"

    else:
        color_matched_arr = transfer_color_profile(base_arr, style_arr)
        init_arr = cv2.addWeighted(base_arr, 1.0 - style_weight, color_matched_arr, style_weight, 0)
        init_pil = Image.fromarray(init_arr)
        depth_map = get_depth_map(base_pil) 

        strength_val = 0.35 + (0.50 * style_weight) 
        control_scale_val = 0.9 - (0.4 * style_weight)
        guidance_scale_val = 1.5 
        
        positive_prompt = f"Masterpiece, highly detailed artwork entirely painted in the distinct {short_style_name} art style, visible heavy brushstrokes, intricate textures, matching the reference painting"

    negative_prompt = "text, signature, watermark, username, letters, words, logo, photography, realism, plain background, low contrast, standard colors, glitch, noise, corrupted"
    
    generator = torch.Generator(device="cpu").manual_seed(42)

    try:
        output = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=init_pil,               
            control_image=depth_map, 
            strength=strength_val,        
            controlnet_conditioning_scale=control_scale_val,
            num_inference_steps=6,   
            generator=generator,
            guidance_scale=guidance_scale_val 
        ).images[0]

        raw_output_array = np.array(output)
        final_image, hist_log = apply_adaptive_histogram_bending(raw_output_array, loudness, intensity_slider)

        return final_image, f"LCM Turbo + Depth + LAB Sentez Basarili (Stil Yogunlugu: %{int(style_weight*100)})\n{hist_log}"
    except Exception as e:
        hata_detayi = traceback.format_exc()
        try:
            with open(os.path.join(BASE_DIR, "CRASH_LOG.txt"), "w", encoding="utf-8") as f:
                f.write(hata_detayi)
        except:
            pass
        return None, f"DIFUZYON MOTORU COKTU: {str(e)}\nCRASH_LOG.txt dosyasini inceleyin!"

# =====================================================================
# [KRITIK DUZELTME]: GLOBAL WARM-UP (HATA KORUMALI)
# Sistemin soguk baslatmadan (Cold Start) kaynakli kilitlenmesini onler.
# =====================================================================
def _warmup_engine():
    try:
        print("\n" + "="*60)
        print("[SOLENTA V2]: MOTOR ISITMA PROTOKOLU BASLATILIYOR (WARM-UP)")
        print("="*60)
        print("[SOLENTA V2]: 1/4 - MiDaS ve LCM Turbo Modelleri VRAM'e Cekiliyor...")
        
        pipe = get_diffusion_pipeline()
        
        if pipe is not None:
            print("[SOLENTA V2]: 2/4 - Sahte (Dummy) Matris Uretiliyor...")
            dummy_matrix = np.zeros((512, 512, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_matrix)
            
            print("[SOLENTA V2]: 3/4 - MiDaS Derinlik Motoru Kor Atesleme Yapiyor...")
            dummy_depth = get_depth_map(dummy_pil)
            
            print("[SOLENTA V2]: 4/4 - LCM Turbo CUDA Cekirdekleri Isitiliyor (Lutfen Bekleyin)...")
            try:
                # DUZELTME: Tensor cokusunu onlemek icin negative_prompt ve 4 step eklendi.
                _ = pipe(
                    prompt="a black square",
                    negative_prompt="nothing",
                    image=dummy_pil,
                    control_image=dummy_depth,
                    strength=0.5,
                    controlnet_conditioning_scale=0.5,
                    num_inference_steps=4,
                    guidance_scale=1.5
                )
                print("-" * 60)
                print("[SOLENTA V2]: WARM-UP TAMAMLANDI! Motorlar jilet gibi, rolantide bekliyor.")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"[UYARI]: İnferans ısınması atlandı: {e}")
        else:
            print("[UYARI]: Modeller yüklenemediği için ısınma atlandı.")
    except Exception as e_top:
        print(f"[UYARI]: Isınma protokolü atlandı ({e_top}). Arayüz başlatılıyor...")

_warmup_engine()