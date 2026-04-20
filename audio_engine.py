import librosa
import numpy as np
import warnings

# Terminal kirliliğini önlemek için Librosa'nın gereksiz uyarılarını gizliyoruz
warnings.filterwarnings('ignore')

def analyze_audio_determinants(audio_path, start_time=0.0, duration=10.0):
    """
    SENTA İŞİTSEL ANALİZ MOTORU (Akademik Revizyon v2.0):
    İşitsel Algı Belirleyenleri teorisine göre yapılandırılmıştır:
    1. Ritim/Tempo (BPM) -> Form ve Düzlem Yoğunluğu
    2. Tını (Spectral Centroid) -> Sanat Eseri/Doku Referansı seçimi
    3. Perde (Pitch) -> Renk Tayfı (Hue - 0-360 derece)
    4. Gürlük (Loudness) -> Ölçek ve Derinlik (Yakınlık/Uzaklık)
    """
    try:
        # --- GÜVENLİK VE VERİ DOĞRULAMA KATMANI ---
        if not isinstance(start_time, (int, float)) or start_time < 0:
            start_time = 0.0
            
        if not isinstance(duration, (int, float)) or duration <= 0 or duration > 10.0:
            duration = 10.0
            
        y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
        
        # DİKKAT: app.py 4 değişken beklediği için her hata durumunda 4 değer dönmeliyiz!
        if len(y) == 0:
             return None, "SİSTEM HATASI: Seçilen saniyede analiz edilecek yeterli ses verisi bulunamadı.", None, None

        # 1. RİTİM VE TEMPO (BPM)
        # Neden eklendi: İşitsel olayların zaman düzlemindeki organizasyonunu ölçmek için eklendi.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        # 2. TINI (Timbre / Spectral Centroid) - Entropi Yerine Eklendi
        # Neden eklendi: Sesin spektral içeriğini ölçüp farklı ses kaynaklarını ayırt etmek ve en uygun referans sanat eserini eşleştirmek için eklendi.
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        timbre_centroid = float(np.mean(cent))
        
        # 3. PERDE (Pitch -> Hue Mapping)
        # Neden eklendi: Frekansı zihinsel bir ölçeğe oturtmak için eklendi. Çıkan dominant frekans doğrudan 0-360 arası renk çemberine (Hue) dönüştürülerek tek renk hatası çözüldü.
        stft = np.abs(librosa.stft(y))
        power_spectrum = np.sum(stft, axis=1)
        power_spectrum[0] = 0  # Sessizlikte DC offset (0Hz) hatası vermemesi için sıfırlandı.
        frequencies = librosa.fft_frequencies(sr=sr)
        dominant_pitch = float(frequencies[np.argmax(power_spectrum)])
        
        # Logaritmik frekans algısını 0-360 derece Hue çemberine modüle ediyoruz (Dinamik Renk)
        hue_value = int(dominant_pitch % 360)
        
        # 4. GÜRLÜK (Loudness / RMS Energy)
        # Neden eklendi: Sesin şiddetini psikolojik olarak "yakınlık" veya "uzaklık" (derinlik ve boyut) parametresine çevirmek için eklendi.
        rms = librosa.feature.rms(y=y)
        loudness = float(np.mean(rms))
        normalized_loudness = min(max(int((loudness / 0.3) * 150), 20), 150)
        
        # app.py'ye gidecek yeni 4'lü: BPM, Tını (Centroid), Hue (Renk Tayfı), Gürlük
        return round(bpm, 2), round(timbre_centroid, 2), hue_value, normalized_loudness
    
    except Exception as e:
        # Hata anında uygulamanın çökmemesi için 4 adet null değer dönülüyor
        return None, f"KRİTİK İŞİTSEL HATA: {str(e)}", None, None

# --- SİSTEM TEST BLOĞU ---
if __name__ == "__main__":
    import os
    test_file = "test_sarkisi.mp3" 
    secilen_saniye = 45.0 
    secilen_sure = 10.0
    
    print(f"███ SENTA AUDIO MOTORU (REV 2.0) TEST EDİLİYOR ███")
    if os.path.exists(test_file):
        bpm_res, timbre_res, hue_res, loud_res = analyze_audio_determinants(test_file, start_time=secilen_saniye, duration=secilen_sure)
        print(f"Ritim (BPM): {bpm_res} | Tını (Centroid): {timbre_res} Hz | Renk Tayfı (Hue): {hue_res}° | Gürlük (Ölçek): {loud_res}px")
    else:
        print(f"[UYARI] '{test_file}' dosyası bulunamadı. Lütfen sistemi Gradio üzerinden test edin.")