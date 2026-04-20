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
    """
    [DORSAL AKIŞ - WHERE SİSTEMİ 2.0 (HACİMSEL ALGI)]
    Senta nesnelerin 3 boyutlu hacmini algılar, form bozulmaları engellenir.
    """
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is None:
        print("[SENTA V2]: MiDaS Derinlik Algısı (Depth Map) Motoru Yükleniyor...")
        device_id = 0 if torch.cuda.is_available() else -1
        _DEPTH_ESTIMATOR = pipeline('depth-estimation', model="Intel/dpt-hybrid-midas", device=device_id)
    
    depth_image = _DEPTH_ESTIMATOR(image_pil)['depth']
    return depth_image.convert("RGB")

def get_diffusion_pipeline():
    """
    [SİSTEM BAŞLATICI - LCM TURBO + DEPTH MİMARİSİ]
    """
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
        
        if device == "cuda":
            pipe.enable_model_cpu_offload() 
            pipe.enable_attention_slicing() 
        else:
            pipe.to("cpu")
        
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

# [YENİ]: Akademik Savunma İçin Müzik Kısıtlı Piksel Bükücü (Adaptif Histogram)
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

# [GÜNCELLENDİ]: slider ve loudness parametreleri eklendi.
def synthesize_fusion_art(base_image, style_reference, is_music_mode=False, target_concept=None, style_name="Cubist", loudness=50.0, intensity_slider=1.0):
    pipe = get_diffusion_pipeline()
    if pipe is None: return None, "HATA: Motor başlatılamadı."

    if isinstance(base_image, np.ndarray):
        if base_image.dtype != np.uint8:
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        base_pil = Image.fromarray(base_image).convert("RGB")
    else:
        base_pil = base_image.convert("RGB")
        base_image = np.array(base_pil)

    w, h = base_pil.size

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

    if is_music_mode:
        fusion_arr = cv2.addWeighted(base_arr, 0.7, style_arr, 0.3, 0)
    else:
        fusion_arr = cv2.addWeighted(base_arr, 0.4, style_arr, 0.6, 0)

    init_pil = Image.fromarray(fusion_arr)
    depth_map = get_depth_map(init_pil) 
    dominant_color = get_dominant_color_name(base_image)
    
    control_scale_val = 0.85
    strength_val = 0.80

    short_style_name = " ".join(style_name.split()[:2]) if len(style_name.split()) > 2 else style_name

    if is_music_mode:
        if target_concept and target_concept.strip() != "":
            control_scale_val = 0.45 
            strength_val = 0.85 
            positive_prompt = f"A clear, striking {target_concept} as the central focal point, highly detailed, photorealistic features, emerging from {dominant_color} abstract intersecting cubist planes, heavily textured with {short_style_name} art style, sharp edges, vibrant, aesthetic digital art, 8k resolution"
        else:
            positive_prompt = f"{dominant_color} themed abstract expressionist music visualization, heavily textured with {short_style_name} art style, masterpiece, highly detailed, vibrant, aesthetic digital art, 8k resolution, sharp geometric shapes"
    else:
        positive_prompt = f"Masterpiece, highly detailed, exact physical structure retained, heavily stylized textures matching the {short_style_name} reference art style"
        strength_val = 0.65 

    negative_prompt = "text, signature, watermark, username, letters, words, logo, watercolor, washed out, pale, blurry, abstract blobs, ugly, deformed identity, bad anatomy, sloppy blending, low contrast"
    
    generator = torch.Generator(device="cpu").manual_seed(42)

    try:
        output = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=init_pil,               
            control_image=depth_map, 
            strength=strength_val,        
            controlnet_conditioning_scale=control_scale_val,
            num_inference_steps=5,   
            generator=generator,
            guidance_scale=2.0
        ).images[0]

        raw_output_array = np.array(output)
        
        # [ALGORİTMİK İMZA]: AI üretti, biz büküyoruz.
        final_image, hist_log = apply_adaptive_histogram_bending(raw_output_array, loudness, intensity_slider)

        return final_image, f"LCM Turbo + Depth Map Başarılı\n{hist_log}"
    except Exception as e:
        return None, f"DİFÜZYON MOTORU ÇÖKTÜ: {str(e)}"