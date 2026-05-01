import gradio as gr
import librosa
import numpy as np
import cv2 

from scipy.spatial import Voronoi

from audio_engine import analyze_audio_determinants
from color_engine import analyze_color_context
from nst_engine import synthesize_nst_art, apply_adaptive_histogram_bending
from filter_engine import apply_timbre_driven_convolution, apply_kmeans_color_quantization 
from senta_logger import log_training_data
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

try:
    from diffusion_engine import synthesize_fusion_art
except ImportError:
    synthesize_fusion_art = None


# --- 1. AŞAMA: BİLGİSAYAR BİLİMLERİ (AKUSTİK VORONOI İSKELETİ) ---
def generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness):
    width, height = 512, 512
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 20
    
    num_points = int(max(12, (bpm / 2.0) + (loudness / 10.0)))
    seed_value = int(bpm + timbre + hue_val + loudness)
    rng = np.random.default_rng(seed_value)
    points = rng.integers(0, 512, size=(num_points, 2))
    
    boundary_points = np.array([[-100, -100], [-100, 612], [612, -100], [612, 612]])
    all_points = np.vstack([points, boundary_points])
    
    vor = Voronoi(all_points)
    
    sat_val = 200 
    val_lightness = int(np.interp(loudness, [20, 150], [60, 255]))
    hsv_color = np.uint8([[[hue_val, sat_val, val_lightness]]])
    base_rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]

    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not region or -1 in region: continue
            
        polygon = [vor.vertices[i] for i in region]
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        color = (
            int(np.clip(base_rgb[0] + rng.integers(-45, 45), 0, 255)),
            int(np.clip(base_rgb[1] + rng.integers(-35, 35), 0, 255)),
            int(np.clip(base_rgb[2] + rng.integers(-55, 55), 0, 255))
        )
        cv2.fillPoly(canvas, [pts], color)
        
        edge_thickness = max(1, int(loudness / 25))
        edge_color = (max(0, color[0]-60), max(0, color[1]-60), max(0, color[2]-60))
        cv2.polylines(canvas, [pts], isClosed=True, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

    noise = rng.normal(0, int(loudness/10), canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return canvas


# --- 2. İŞLEM FONKSİYONLARI (MÜZİK) ---
def generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro):
    log_text = f"███ SENTA ÜRETİM SÜRECİ ANALİZİ ███\n\n"
    log_text += f"🎵 1. İŞİTSEL-GÖRSEL HARİTALAMA (DETERMİNANTLAR)\n"
    log_text += f" ├─ Ritim (BPM)    : {bpm:.1f} ➔ Voronoi tohum noktalarının sayısını belirledi.\n"
    log_text += f" ├─ Tını (Timbre)  : {timbre:.1f} Hz ➔ Dinamik Konvolüsyon Matrisi çarpanı ve Doku referansı.\n"
    log_text += f" ├─ Perde (Hue)    : {hue_val}° ➔ Referans eseri müzikal tayf rengine senkronize etti.\n"
    log_text += f" └─ Gürlük (Vol)   : {loudness:.1f} ➔ Adaptif Histogram Gamma çarpanı olarak kullanıldı.\n\n"
    
    log_text += f"🎨 2. TOPOLOJİK SENTEZ\n"
    log_text += f" ├─ İskelet Algoritması: Akustik Voronoi Şeması (O(n log n))\n"
    log_text += f" ├─ Odak Nesnesi: {target_concept_tr if (is_pro and target_concept_tr != 'Yok (Saf Soyut)') else 'Saf Soyut (Sadece Vektörler)'}\n"
    log_text += f" ├─ Motor: {engine_used}\n"
    log_text += f" └─ Sistem Durumu: {status}\n"
        
    return log_text

def process_primary_generation(audio_upload, audio_preset, start_time_sec, synthesis_mode, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider):
    try:
        audio_file = audio_upload if audio_upload else audio_preset
        if not audio_file:
            return None, "[HATA]: Lütfen müzik yükleyin.", gr.update(visible=False), gr.update(visible=False), None

        safe_start = float(start_time_sec) if start_time_sec else 0.0
        bpm, timbre, hue_val, loudness = analyze_audio_determinants(audio_file, start_time=safe_start, duration=10.0)
        
        if bpm is None: return None, f"[HATA]: {timbre}", gr.update(), gr.update(), None

        if bpm > 120: akim_karari = "Analitik Kübizm (Yüksek Form Dominansı)"
        elif bpm > 80: akim_karari = "Sentetik Kübizm (Geniş Renk Yüzeyleri)"
        else: akim_karari = "Proto-Kübizm (Sadeleştirilmiş Doğrusal İndeks)"

        top_3 = get_top_3_artworks_from_audio(timbre, akim_karari, target_concept_tr)
        if not top_3: return None, "[HATA]: Eser bulunamadı.", gr.update(), gr.update(), None

        best_path = top_3[0][0]
        style_name = top_3[0][1].split(": ")[-1]
        
        raw_img = safe_image_read(best_path)
        style_img = recolor_image_to_music_hue(raw_img, hue_val)
        structural_canvas = generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness)

        concept_map = {
            "Yok (Saf Soyut)": None, 
            "İnsan Yüzü": "A dramatic close-up of a human face", 
            "Göz": "A highly detailed close-up of a human eye", 
            "Kedi": "A majestic cat silhouette",
            "Uçan Kuş": "A flying bird with spread wings", 
            "Yaşlı Ağaç": "An ancient solitary tree with sprawling branches", 
            "Çiçek (Lotus)": "A blooming lotus flower"
        }

        is_pro = "Pro" in synthesis_mode
        state_data = {
            "bpm": bpm, "timbre": timbre, "hue_val": hue_val, "loudness": loudness,
            "akim": akim_karari, "top_3": top_3, "safe_start": safe_start,
            "active_brush": top_3[0][1]
        }

        if is_pro:
            target_en = concept_map.get(target_concept_tr, None)
            if synthesize_fusion_art is None: return None, "[SİSTEM HATASI] Pro Mod kütüphaneleri eksik.", gr.update(), gr.update(), None
            final_image, status = synthesize_fusion_art(structural_canvas, style_img, is_music_mode=True, target_concept=target_en, style_name=style_name, loudness=loudness, intensity_slider=intensity_slider)
            
            state_data["pro_base_output"] = final_image.copy() 
            state_data["raw_light_output"] = None
            engine_used = "LCM Turbo + MiDaS Depth"
            hist_log = status.split("\n")[-1] if "\n" in status else ""
            status = status.split("\n")[0]
        else:
            raw_final_image, status = synthesize_nst_art(structural_canvas, style_img)
            state_data["raw_light_output"] = raw_final_image.copy() 
            state_data["pro_base_output"] = None
            engine_used = "Neural Style Transfer"
            final_image, hist_log = apply_adaptive_histogram_bending(raw_final_image, loudness, intensity_slider)

        final_image, spatial_log = apply_timbre_driven_convolution(final_image, timbre, spatial_slider)
        final_image, kmeans_log = apply_kmeans_color_quantization(final_image, kmeans_slider)

        base_log = generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro)
        state_data["base_log"] = base_log 
        
        full_log = base_log + f"\n⚙️ 3. MATEMATİKSEL MODİFİKASYON (ALGORİTMA RAPORU)\n ├─ {hist_log}\n ├─ {spatial_log}\n └─ {kmeans_log}"

        log_training_data(final_image, state_data, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider)

        choices = [item[1] for item in top_3]
        return final_image, full_log, gr.update(choices=choices, value=choices[0], visible=True), gr.update(visible=True), state_data

    except Exception as e:
        return None, f"HATA: {str(e)}", gr.update(visible=False), gr.update(visible=False), None

def process_alternative_generation(state_data, selected_brush, synthesis_mode, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider):
    try:
        if not state_data or not selected_brush: return gr.update(), gr.update(), state_data
        if state_data.get("active_brush") == selected_brush: return gr.update(), gr.update(), state_data
            
        state_data["active_brush"] = selected_brush 

        bpm, timbre, hue_val, loudness, akim_karari, top_3 = state_data["bpm"], state_data["timbre"], state_data["hue_val"], state_data["loudness"], state_data["akim"], state_data["top_3"]
        
        best_path = next(path for path, name in top_3 if name == selected_brush)
        raw_img = safe_image_read(best_path)
        style_img = recolor_image_to_music_hue(raw_img, hue_val)
        structural_canvas = generate_acoustic_voronoi_canvas(bpm, timbre, hue_val, loudness)
        
        style_name = selected_brush.split(": ")[-1]
        
        concept_map = {
            "Yok (Saf Soyut)": None, 
            "İnsan Yüzü": "A dramatic close-up of a human face", 
            "Göz": "A highly detailed close-up of a human eye", 
            "Kedi": "A majestic cat silhouette",
            "Uçan Kuş": "A flying bird with spread wings", 
            "Yaşlı Ağaç": "An ancient solitary tree with sprawling branches", 
            "Çiçek (Lotus)": "A blooming lotus flower"
        }

        is_pro = "Pro" in synthesis_mode

        if is_pro:
            target_en = concept_map.get(target_concept_tr, None)
            final_image, status = synthesize_fusion_art(structural_canvas, style_img, is_music_mode=True, target_concept=target_en, style_name=style_name, loudness=loudness, intensity_slider=intensity_slider)
            
            state_data["pro_base_output"] = final_image.copy()
            state_data["raw_light_output"] = None
            engine_used = "LCM Turbo + MiDaS Depth"
            hist_log = status.split("\n")[-1] if "\n" in status else ""
            status = status.split("\n")[0]
        else:
            raw_final_image, status = synthesize_nst_art(structural_canvas, style_img)
            state_data["raw_light_output"] = raw_final_image.copy()
            state_data["pro_base_output"] = None
            engine_used = "Neural Style Transfer"
            final_image, hist_log = apply_adaptive_histogram_bending(raw_final_image, loudness, intensity_slider)

        final_image, spatial_log = apply_timbre_driven_convolution(final_image, timbre, spatial_slider)
        final_image, kmeans_log = apply_kmeans_color_quantization(final_image, kmeans_slider)

        base_log = generate_detailed_log(bpm, timbre, hue_val, loudness, akim_karari, style_name, engine_used, status, target_concept_tr, is_pro)
        state_data["base_log"] = base_log 
        
        full_log = base_log + f"\n⚙️ 3. MATEMATİKSEL MODİFİKASYON (ALGORİTMA RAPORU)\n ├─ {hist_log}\n ├─ {spatial_log}\n └─ {kmeans_log}"
        
        log_training_data(final_image, state_data, target_concept_tr, intensity_slider, spatial_slider, kmeans_slider)

        return final_image, full_log, state_data
    except Exception as e:
        return None, f"SENTEZ HATASI: {str(e)}", state_data


def apply_all_post_processing(state_data, intensity_slider, spatial_slider, kmeans_slider):
    if not state_data: return gr.update(), gr.update()
    
    log_additions = []
    
    if state_data.get("raw_light_output") is not None:
        current_image, hist_log = apply_adaptive_histogram_bending(state_data["raw_light_output"], state_data["loudness"], intensity_slider)
        log_additions.append(hist_log)
    else:
        current_image = state_data.get("pro_base_output")
        if current_image is None: return gr.update(), gr.update()
        
    img_conv, spatial_log = apply_timbre_driven_convolution(current_image, state_data["timbre"], spatial_slider)
    log_additions.append(spatial_log)
    
    final_image, kmeans_log = apply_kmeans_color_quantization(img_conv, kmeans_slider)
    log_additions.append(kmeans_log)
    
    new_log_text = "\n ├─ ".join(log_additions)
    final_log = state_data["base_log"] + f"\n⚙️ 3. MATEMATİKSEL MODİFİKASYON (CANLI)\n ├─ {new_log_text}"
    
    return final_image, gr.update(value=final_log)


# --- 3. GÖRSEL SEKME UX KONTROLLERİ ---
def initialize_auto_scan(content_img):
    if content_img is None: return gr.update(value=[]), [], "Lütfen önce fotoğraf yükleyin."
    zemin_rengi, _, _ = analyze_color_context(content_img)
    if zemin_rengi in ["Kırmızı", "Turuncu", "Sarı"]: akim = "Sentetik Kübizm (Geniş Renk Yüzeyleri)"
    elif zemin_rengi in ["Mavi", "Mor", "Yeşil"]: akim = "Analitik Kübizm (Yüksek Form Dominansı)"
    else: akim = "Proto-Kübizm (Sadeleştirilmiş Doğrusal İndeks)"
    top_10 = get_ranked_top_10_artworks(content_img, akim)
    return gr.update(value=top_10), top_10, "AI taraması tamamlandı! Lütfen listeden bir eser seçiniz."

def toggle_auto_mode(is_auto):
    if is_auto: return gr.update(visible=False), gr.update(visible=True)
    else: return gr.update(visible=True), gr.update(visible=False)

def save_gallery_selection(evt: gr.SelectData):
    return evt.index, gr.update(interactive=True)

def process_image_to_image(content_image, is_auto, artist, artwork, top_10_state, selected_idx, synthesis_mode):
    try:
        if content_image is None: return None, "[HATA]: Fotoğraf yükleyin."
        
        user_lum, user_edge = calculate_image_features(content_image)
        zemin_rengi, tamamlayici, etki = analyze_color_context(content_image)
        
        if is_auto:
            if selected_idx is None: return None, "Eser seçin!"
            style_img = safe_image_read(top_10_state[selected_idx][0])
            style_name = top_10_state[selected_idx][1]
            secim_yontemi = "AI Bulanık Mantık Taraması (Top-10)"
        else:
            if not artist or not artwork: return None, "Sanatçı ve Eser seçin."
            style_img = get_artwork_image(artist, artwork)
            style_name = f"{artist} - {artwork}"
            secim_yontemi = "Kullanıcı Manuel Seçimi"

        if "Pro" in synthesis_mode:
            final_image, status = synthesize_fusion_art(content_image, style_img, is_music_mode=False, style_name=style_name)
            engine_used = "LCM Turbo + MiDaS Depth"
            fusion_text = "Algısal Harmanlama: Referans sanat eserinin dokusu, yüklenen fotoğrafın MiDaS ile çıkarılan hacim haritasına (Depth Map) işlendi."
        else:
            final_image, status = synthesize_nst_art(content_image, style_img)
            engine_used = "Neural Style Transfer"
            fusion_text = "Matris Transferi: Referans sanat eserinin dokusal pikselleri doğrudan içerik matrisinin üzerine Neural Network ile işlendi."

        log_text = f"███ SENTA GÖRSEL SENTEZ RAPORU ███\n\n"
        log_text += f"🔍 1. GÖRSEL ALGI (ÖZNİTELİK ÇIKARIMI)\n"
        log_text += f" ├─ Işık Haritası (Luminance): {user_lum:.2f} (Piksel Parlaklık Ortalaması)\n"
        log_text += f" ├─ Kenar Yoğunluğu (Edges)  : {user_edge:.2f} (Canny Edge Algoritması Puanı)\n"
        log_text += f" └─ Zemin Rengi Analizi      : {zemin_rengi} ({etki})\n\n"

        log_text += f"🧠 2. REFERANS HARİTALAMA\n"
        log_text += f" ├─ Karar Yöntemi: {secim_yontemi}\n"
        log_text += f" └─ Referans Eser: {style_name}\n\n"

        log_text += f"⚙️ 3. TOPOLOJİK SENTEZ\n"
        log_text += f" ├─ Sentez Motoru : {engine_used}\n"
        log_text += f" ├─ Sentez Tekniği: {fusion_text}\n"
        log_text += f" └─ Motor Durumu  : {status}"

        return final_image, log_text
    except Exception as e:
        return None, f"HATA: {str(e)}"

# --- YENİ UX DENEYİMİ: DİNAMİK MÜZİK GİRDİSİ ---
def handle_preset_selection(preset_name):
    # Hazır şarkı seçildiyse: Kullanıcının yüklediği dosyayı sil, saniye kutusunu Kapat ve 0'a sabitle.
    if preset_name:
        return None, gr.update(visible=False, value=0)
    return gr.update(), gr.update(visible=True)

def handle_user_upload(file_path):
    # Kullanıcı dosya yüklediyse: Hazır şarkı seçimini sil, saniye kutusunu Aç.
    if file_path:
        return None, gr.update(visible=True)
    return gr.update(), gr.update(visible=True)

# --- 4. CSS VE ARAYÜZ ---
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

with gr.Blocks(css=custom_css, title="SENTA Laboratuvarı") as senta_app:
    gr.Markdown("# SENTA PROJE LABORATUVARI")
    gr.Markdown("<br>")

    with gr.Tabs():
        with gr.TabItem("🎵 Müzikten Görsel Üret (Generative)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### İŞİTSEL GİRDİ")
                    audio_mode_radio = gr.Radio(choices=["⚡ Light Mod (NST)", "🚀 Pro Mod (LCM Turbo)"], value="⚡ Light Mod (NST)", label="Motor Seçimi", elem_classes="radio-group")
                    
                    target_concept_dropdown = gr.Dropdown(
                        choices=["Yok (Saf Soyut)", "İnsan Yüzü", "Göz", "Kedi", "Uçan Kuş", "Yaşlı Ağaç", "Çiçek (Lotus)"],
                        value="Yok (Saf Soyut)",
                        label="Opsiyonel Odak Nesnesi (Focal Point)",
                        visible=False
                    )
                    
                    audio_input = gr.Audio(type="filepath", label="Müzik Yükle (Kendi Dosyan)")
                    preset_audio = gr.Dropdown(choices=get_preset_audio_list(), label="Veya Hazır Müzik Seç (Otomatik Kırpılmış)") 
                    
                    # [UX GÜNCELLEMESİ]: Bu kutu artık dinamik olarak kaybolup belirecek
                    start_time_input = gr.Number(value=0, label="Hangi Saniyeden Başlasın? (Sadece Yüklenen Dosyalar İçin)", precision=0)
                    
                    with gr.Group():
                        gr.Markdown("#### 🧮 Sinyal-Piksel Bükücü Algoritmalar")
                        intensity_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Adaptif Histogram Eğrisi (Logaritmik Işık Bükücü)", info="Sesin gürlüğüne oranla karanlık pikselleri ezer.")
                        spatial_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Mekansal Konvolüsyon (Tını Sürücülü Matris)", info="Müziğin Tınısına bağlı olarak Kenar Bulucu çekirdeğini tetikler.")
                        kmeans_slider = gr.Slider(minimum=2, maximum=65, value=65, step=1, label="Vektörel Renk Kümeleme (K-Means Algoritması)", info="Pikselleri 3D uzayda kümeleyerek renk sayısını azaltır. (65 = Limitsiz/Devre Dışı)")
                    
                    btn_audio = gr.Button("🚀 Playlist Kapağını Sentezle", variant="primary")
                    audio_state = gr.State()
                    
                with gr.Column(scale=1):
                    gr.Markdown("### SENTETİK ÇIKTI")
                    audio_output_img = gr.Image(label="Müzikten Sentezlenen Yeni Eser")
                    audio_log = gr.Textbox(label="Üretim Süreci Analizi", lines=15)
                    
                    with gr.Column(visible=False, elem_classes="alt-box") as alt_column:
                        gr.Markdown("#### 🎨 Çıkan Dokuyu Beğenmedin mi?")
                        brush_options = gr.Dropdown(label="Müziğine uygun diğer 2 dokudan birini seç:", choices=[], interactive=True)
                        
            # --- YENİ UX ETKİLEŞİMLERİ ---
            preset_audio.change(
                fn=handle_preset_selection, 
                inputs=[preset_audio], 
                outputs=[audio_input, start_time_input]
            )
            
            audio_input.change(
                fn=handle_user_upload, 
                inputs=[audio_input], 
                outputs=[preset_audio, start_time_input]
            )
            # -----------------------------

            audio_mode_radio.change(
                fn=lambda mode: gr.update(visible="Pro" in mode),
                inputs=[audio_mode_radio],
                outputs=[target_concept_dropdown],
                show_progress="hidden"
            )
            
            btn_audio.click(
                fn=process_primary_generation,
                inputs=[audio_input, preset_audio, start_time_input, audio_mode_radio, target_concept_dropdown, intensity_slider, spatial_slider, kmeans_slider],
                outputs=[audio_output_img, audio_log, brush_options, alt_column, audio_state]
            )
            
            brush_options.change(
                fn=process_alternative_generation,
                inputs=[audio_state, brush_options, audio_mode_radio, target_concept_dropdown, intensity_slider, spatial_slider, kmeans_slider],
                outputs=[audio_output_img, audio_log, audio_state]
            )
            
            intensity_slider.release(
                fn=apply_all_post_processing,
                inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider],
                outputs=[audio_output_img, audio_log],
                show_progress="hidden"
            )
            
            spatial_slider.release(
                fn=apply_all_post_processing,
                inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider],
                outputs=[audio_output_img, audio_log],
                show_progress="hidden"
            )
            
            kmeans_slider.release(
                fn=apply_all_post_processing,
                inputs=[audio_state, intensity_slider, spatial_slider, kmeans_slider],
                outputs=[audio_output_img, audio_log],
                show_progress="hidden"
            )

        with gr.TabItem("🖼️ Fotoğrafa Sanat İşle (Stylization)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### GÖRSEL GİRDİ")
                    img_mode_radio = gr.Radio(choices=["⚡ Light Mod (NST)", "🚀 Pro Mod (LCM Turbo)"], value="⚡ Light Mod (NST)", label="Motor Seçimi", elem_classes="radio-group")
                    img_content = gr.Image(type="numpy", label="Fotoğrafını Yükle")
                    
                    with gr.Group():
                        auto_mode_check = gr.Checkbox(label="AI En İyi 10 Eseri Önersin (Otomatik Mod)")
                        
                        with gr.Column(visible=True) as manual_selection_col:
                            artist_dropdown = gr.Dropdown(choices=get_all_artists(), label="Sanatçı Seç")
                            artwork_dropdown = gr.Dropdown(choices=[], label="Eser Seç")
                        
                        with gr.Column(visible=False) as auto_selection_col:
                            btn_scan = gr.Button("🔍 500 Eser Arasında AI Taramasını Başlat", elem_classes="scan-btn")
                            top_10_gallery = gr.Gallery(label="AI Önerileri (Seçmek için resme tıklayın)", columns=5, height="auto", allow_preview=False)
                            top_10_state = gr.State([])
                            selected_gallery_idx = gr.State(None)
                    
                    btn_img = gr.Button("⚡ Patternleri İşle", variant="primary", interactive=True)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### STİLİZE ÇIKTI")
                    img_output = gr.Image(label="İşlenmiş Görsel")
                    img_log = gr.Textbox(label="Sentez Analizi", lines=16)

            artist_dropdown.change(
                fn=lambda x: gr.update(choices=get_artworks_by_artist(x), value=None), 
                inputs=artist_dropdown, 
                outputs=artwork_dropdown,
                show_progress="hidden"
            )
            auto_mode_check.change(
                fn=toggle_auto_mode, 
                inputs=[auto_mode_check], 
                outputs=[manual_selection_col, auto_selection_col],
                show_progress="hidden"
            )
            btn_scan.click(fn=initialize_auto_scan, inputs=[img_content], outputs=[top_10_gallery, top_10_state, img_log])
            top_10_gallery.select(fn=save_gallery_selection, outputs=[selected_gallery_idx, btn_img])
            btn_img.click(fn=process_image_to_image, inputs=[img_content, auto_mode_check, artist_dropdown, artwork_dropdown, top_10_state, selected_gallery_idx, img_mode_radio], outputs=[img_output, img_log])

if __name__ == "__main__":
    senta_app.launch()