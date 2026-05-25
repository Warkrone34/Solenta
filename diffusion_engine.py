import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
from transformers import pipeline

# --- GLOBAL MODEL CACHE (Hafıza Yönetimi) ---
_PIPELINE = None
_DEPTH_ESTIMATOR = None

def get_depth_map(image_pil):
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is None:
        print("[SENTA V2]: MiDaS Derinlik Algısı (Depth Map) Motoru Yükleniyor...")
        device_id = 0 if torch.cuda.is_available() else -1
        _DEPTH_ESTIMATOR = pipeline('depth-estimation', model="Intel/dpt-hybrid-midas", device=device_id)
    
    depth_image = _DEPTH_ESTIMATOR(image_pil)['depth']
    return depth_image.convert("RGB")

def get_diffusion_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
        
    print("[SENTA V2]: Depth ControlNet ve LCM Turbo Yükleniyor (Sarsılmaz Sürüm)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None 
        )
        
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # [KRİTİK DÜZELTME]: CPU Offload ve Attention Slicing SİLİNDİ!
        # Modeli direkt Ekran Kartının (VRAM) içine kalıcı olarak gömüyoruz. Hız %500 artacak.
        pipe.to(device)
        
        _PIPELINE = pipe
        print(f"[SENTA V2]: LCM Turbo Motoru Başarıyla Ateşlendi ({device}). Sistem Hazır!")
        return _PIPELINE
    except Exception as e:
        print(f"[KRİTİK HATA]: Model yüklenemedi.\n{e}")
        return None

def get_dominant_color_name(image_matrix):
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
    if image_matrix is None: return None, "HATA: İşlenecek matris yok."
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

    return final_image, f"📉 Adaptif Histogram: γ = {gamma:.2f} | Müzik Çarpanı: {normalized_loudness:.2f}"

def transfer_color_profile(source_img, target_img):
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
    pipe = get_diffusion_pipeline()
    if pipe is None: return None, "HATA: Motor başlatılamadı."

    if isinstance(base_image, np.ndarray):
        if base_image.dtype != np.uint8:
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        base_pil = Image.fromarray(base_image).convert("RGB")
    else:
        base_pil = base_image.convert("RGB")
        base_image = np.array(base_pil)

    # VRAM BOĞULMASINI ÖNLEYEN ÇÖZÜNÜRLÜK OPTİMİZATÖRÜ
    max_dim = 768  # 768x768, LCM Turbo için ideal boyut. Daha büyükse optimize edilir.
    w, h = base_pil.size
    
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        base_pil = base_pil.resize((new_w, new_h), Image.LANCZOS)
        base_image = np.array(base_pil)
        w, h = new_w, new_h
        print(f"[SENTA HIZLANDIRICI]: Fotoğraf {w}x{h} boyutuna optimize edildi.")

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
        
        if target_concept and target_concept.strip() != "":
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

        return final_image, f"LCM Turbo + Depth + LAB Sentez Başarılı (Stil Yoğunluğu: %{int(style_weight*100)})\n{hist_log}"
    except Exception as e:
        return None, f"DİFÜZYON MOTORU ÇÖKTÜ: {str(e)}"

# =====================================================================
# [KRİTİK DÜZELTME]: GLOBAL WARM-UP (HATA KORUMALI)
# =====================================================================
def _warmup_engine():
    print("\n" + "█"*60)
    print("🚀 [SENTA V2]: MOTOR ISITMA PROTOKOLÜ BAŞLATILIYOR (WARM-UP)")
    print("█"*60)
    print("[SENTA V2]: 1/4 - MiDaS ve LCM Turbo Modelleri VRAM'e Çekiliyor...")
    
    pipe = get_diffusion_pipeline()
    
    if pipe is not None:
        print("[SENTA V2]: 2/4 - Sahte (Dummy) Matris Üretiliyor...")
        dummy_matrix = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_matrix)
        
        print("[SENTA V2]: 3/4 - MiDaS Derinlik Motoru Kör Ateşleme Yapıyor...")
        dummy_depth = get_depth_map(dummy_pil)
        
        print("[SENTA V2]: 4/4 - LCM Turbo CUDA Çekirdekleri Isıtılıyor (Lütfen Bekleyin)...")
        try:
            # DÜZELTME: Tensör çöküşünü önlemek için negative_prompt ve 4 step eklendi.
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
            print("[SENTA V2]: ✅ WARM-UP TAMAMLANDI! Motorlar jilet gibi, rölantide bekliyor.")
            print("█" * 60 + "\n")
        except Exception as e:
            print(f"[KRİTİK HATA]: Isıtma sırasında motor çöktü! Hata: {e}")
    else:
        print("[KRİTİK HATA]: Modeller yüklenemediği için ısıtma iptal edildi.")

_warmup_engine()