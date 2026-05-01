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
    secilen_saniye = 45.0 
    secilen_sure = 10.0
    
    print(f"███ SENTA AUDIO MOTORU (REV 3.0 - ONSET TABANLI) TEST EDİLİYOR ███")
    if os.path.exists(test_file):
        bpm_res, timbre_res, hue_res, loud_res = analyze_audio_determinants(test_file, start_time=secilen_saniye, duration=secilen_sure)
        print(f"Ritim (BPM): {bpm_res} | Olay Tabanlı Tını: {timbre_res} Hz | Hue: {hue_res}° | Gürlük: {loud_res}px")
    else:
        print(f"[UYARI] '{test_file}' dosyası bulunamadı. Lütfen sistemi Gradio üzerinden test edin.")