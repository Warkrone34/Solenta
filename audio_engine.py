import librosa
import numpy as np
import warnings
import functools

# Terminal kirliliğini önlemek için Librosa'nın gereksiz uyarılarını gizliyoruz
warnings.filterwarnings('ignore')

# [1. GEÇİCİ HAFIZA - MEMOIZATION BUFFER]
# Hoca İsteği: Aynı processing'i yapma, geçici olarak tut.
@functools.lru_cache(maxsize=32)
def fetch_audio_buffer(audio_path, start_time, duration):
    """Ses dosyasını diskten okuma işlemini önbelleğe alır. I/O ve Matris yükünü sıfırlar."""
    return librosa.load(audio_path, offset=start_time, duration=duration)

# =====================================================================
# [YENİ]: AKILLI KESİM (SMART CROP) - ALTIN 10 SANİYEYİ BULMA ALGORİTMASI
# =====================================================================
def find_optimal_audio_segment(audio_path, window_duration=10.0):
    """
    Kullanıcının yüklediği şarkının tamamını çok hızlı bir şekilde (low sample rate) tarar.
    RMS (Gürlük) ve Onset (Vuruş) varyansının en yüksek olduğu (Drop/Kreşendo) noktayı bulur.
    Jüri için tam bir "Magic UX" (Sihirli Kullanıcı Deneyimi) sağlar.
    """
    try:
        # Hız için şarkıyı sadece 8000 Hz'de okutuyoruz (Analiz için yeterli, RAM'i boğmaz)
        sr_fast = 8000
        y, sr = librosa.load(audio_path, sr=sr_fast)
        
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Eğer şarkı zaten belirlediğimiz süreden kısaysa, hiç hesap yapma direkt baştan başla
        if audio_duration <= window_duration:
            return 0.0
            
        # 1. RMS Enerjisi (Gürlük - Hacim)
        rms = librosa.feature.rms(y=y)[0]
        rms_norm = librosa.util.normalize(rms)
        
        # 2. Onset Zarfi (Vuruş Yoğunluğu - Ritim Patlamaları)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_norm = librosa.util.normalize(onset_env)
        
        # 3. Nihai SENTA Skoru (İki metriğin birleşimi)
        senta_score = rms_norm + onset_norm
        
        # 4. Kayan Pencere (Rolling Window) Matematiği
        hop_length = 512 # Librosa varsayılanı
        frames_per_sec = sr / hop_length
        window_frames = int(window_duration * frames_per_sec)
        
        # Convolve ile 10 saniyelik çerçeveyi tüm şarkı üzerinde kaydırıp skorları topluyoruz
        window = np.ones(window_frames)
        rolling_sum = np.convolve(senta_score, window, mode='valid')
        
        # En yüksek skoru alan pencerenin (çerçevenin) başlangıç indeksini bul
        best_frame_idx = np.argmax(rolling_sum)
        
        # Bulduğumuz matematiksel indeksi saniyeye (time) çevir
        best_time_sec = librosa.frames_to_time(best_frame_idx, sr=sr, hop_length=hop_length)
        
        # Güvenlik Duvarı: Seçilen saniye + 10 sn, şarkının toplam süresini taşıyorsa geri çek
        if best_time_sec + window_duration > audio_duration:
            best_time_sec = max(0.0, audio_duration - window_duration)
            
        print(f"[SENTA ZEKASI]: Şarkı analiz edildi. En yüksek enerjili 'Drop' noktası {round(best_time_sec, 2)}. saniyede bulundu!")
        return round(best_time_sec, 2)
        
    except Exception as e:
        print(f"[AKILLI KESİM HATASI]: Motor tarama yapamadı, varsayılan saniye (0.0) atandı. Detay: {e}")
        return 0.0

# =====================================================================
# [2. SENTA İŞİTSEL ANALİZ MOTORU]
# =====================================================================
def analyze_audio_determinants(audio_path, start_time=0.0, duration=10.0):
    """
    SENTA İŞİTSEL ANALİZ MOTORU (Akademik Revizyon v3.0 - Onset Tabanlı):
    Sesi körlemesine değil, Transient (Saldırı/Vuruş) noktalarında analiz eder.
    """
    try:
        # --- GÜVENLİK KATMANI ---
        if not isinstance(start_time, (int, float)) or start_time < 0:
            start_time = 0.0
        if not isinstance(duration, (int, float)) or duration <= 0 or duration > 10.0:
            duration = 10.0
            
        y, sr = fetch_audio_buffer(audio_path, start_time, duration)
        
        if len(y) == 0:
             return None, "SİSTEM HATASI: Seçilen saniyede analiz edilecek yeterli ses verisi bulunamadı.", None, None

        # [2. NYQUIST VE DİNAMİK BLOK BÜYÜKLÜĞÜ (WINDOWING)]
        # Hoca İsteği: Blok büyüklükleri nasıl hesaplanıyor?
        # Yaklaşık 46ms'lik transient (saldırı) blokları yakalamak için örnekleme frekansına göre dinamik hesap.
        target_window_ms = 0.046  
        n_fft_ideal = int(sr * target_window_ms)
        n_fft = 2 ** int(np.log2(n_fft_ideal)) # İşlemci optimizasyonu için en yakın 2'nin kuvveti (Örn: 1024)
        hop_length = n_fft // 4 # %75 örtüşme (overlap) ile ani ritimleri kaçırmamak için

        # [3. TRANSIENT / ONSET DETECTION (Olay Tabanlı Bölütleme)]
        # Hoca İsteği: 10 saniyeyi neye göre bölüyoruz?
        # Sadece vuruşların (kick, snare, gitar atağı) olduğu frame indekslerini buluruz.
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        if len(onset_frames) == 0: 
            onset_frames = np.array([0]) # Sessiz kısımlar için fallback
            
        # 1. RİTİM VE TEMPO (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        # 2. TINI (Timbre / Spectral Centroid) - [SADECE VURUŞ ANLARINDA]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        # Spektral veriyi sadece müziğin patladığı "Onset" karelerinde alıyoruz, gürültüyü elliyoruz.
        valid_frames = [f for f in onset_frames if f < len(cent)]
        timbre_centroid = float(np.mean(cent[valid_frames])) if valid_frames else float(np.mean(cent))
        
        # 3. PERDE (Pitch -> Hue Mapping) - [SADECE VURUŞ ANLARINDA]
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        # STFT matrisinden sadece vuruş (onset) anlarındaki frekans sütunlarını filtrele
        onset_stft = stft[:, valid_frames] if valid_frames else stft
        power_spectrum = np.sum(onset_stft, axis=1)
        power_spectrum[0] = 0  # DC Offset hatası koruması
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        dominant_pitch = float(frequencies[np.argmax(power_spectrum)])
        
        hue_value = int(dominant_pitch % 360)
        
        # 4. GÜRLÜK (Loudness / RMS Energy) - [SADECE VURUŞ ANLARINDA]
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        valid_rms = [f for f in onset_frames if f < len(rms)]
        loudness = float(np.mean(rms[valid_rms])) if valid_rms else float(np.mean(rms))
        normalized_loudness = min(max(int((loudness / 0.3) * 150), 20), 150)
        
        return round(bpm, 2), round(timbre_centroid, 2), hue_value, normalized_loudness
    
    except Exception as e:
        return None, f"KRİTİK İŞİTSEL HATA: {str(e)}", None, None

# --- SİSTEM TEST BLOĞU ---
if __name__ == "__main__":
    import os
    test_file = "test_sarkisi.mp3" 
    
    print(f"███ SENTA AUDIO MOTORU (REV 3.1 - AKILLI KESİM & ONSET TABANLI) TEST EDİLİYOR ███")
    if os.path.exists(test_file):
        print("\n[ADIM 1]: Akıllı Kesim Motoru (Smart Crop) Devrede...")
        en_iyi_saniye = find_optimal_audio_segment(test_file)
        
        print("\n[ADIM 2]: Seçilen 10 saniyelik bölüm için işitsel analiz yapılıyor...")
        secilen_sure = 10.0
        bpm_res, timbre_res, hue_res, loud_res = analyze_audio_determinants(test_file, start_time=en_iyi_saniye, duration=secilen_sure)
        print(f"Ritim (BPM): {bpm_res} | Olay Tabanlı Tını: {timbre_res} Hz | Hue: {hue_res}° | Gürlük: {loud_res}px")
    else:
        print(f"[UYARI] '{test_file}' dosyası bulunamadı. Lütfen sistemi Gradio üzerinden test edin.")