import librosa
import numpy as np
import warnings
import functools

# İşletim sistemi veya donanım limitasyonlarından kaynaklı Librosa
# bağımlılık uyarılarını (FutureWarning vb.) log kirliliğini ve arayüz 
# çökmelerini önlemek adına işletim sistemi seviyesinde bastırır.
warnings.filterwarnings('ignore')

# =====================================================================
# [SİSTEM YAPILANDIRMASI 1]: Bellek Yönetimi ve I/O Optimizasyonu
# =====================================================================
# Neden Eklendi: Kullanıcı arayüzde filtreleri (slider) her değiştirdiğinde 
# aynı ses dosyasının diskten tekrar tekrar okunması darboğaza (bottleneck) yol açar.
# LRU (Least Recently Used) Cache kullanılarak okunan sinyal matrisi RAM'e 
# mühürlenir (Memoization) ve okuma maliyeti O(1) seviyesine indirgenir.
@functools.lru_cache(maxsize=32)
def fetch_audio_buffer(audio_path, start_time, duration):
    """
    Ses dosyasını diskten okuyup NumPy matrisine dönüştüren ve önbelleğe alan yalıtılmış fonksiyon.
    """
    return librosa.load(audio_path, offset=start_time, duration=duration)

# =====================================================================
# [SİNYAL İŞLEME MODÜLÜ 1]: Heuristik Enerji Kestirimi (Smart Crop)
# =====================================================================
def find_optimal_audio_segment(audio_path, window_duration=10.0):
    """
    Neden Eklendi: Müzik dosyasındaki en yoğun enerjili bölümü (kreşendo/drop) 
    otonom olarak bulmak için tasarlanmıştır. Kullanıcı manuel kesim yapmadığında 
    sistemin görsel üretim için en zengin frekans aralığını garantilemesini sağlar.
    """
    try:
        # Neden 8000 Hz: Şarkının tamamını yüksek kalitede taramak RAM'i bloke eder.
        # Sadece enerji yoğunluğunu bulmak için Nyquist teoremine göre düşük örnekleme hızı (Downsampling) yeterlidir.
        sr_fast = 8000
        y, sr = librosa.load(audio_path, sr=sr_fast)
        
        # Sinyalin saniye cinsinden toplam uzunluğunu hesaplar.
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Dosya istenen pencere aralığından zaten kısaysa, tarama yapmadan doğrudan başlangıcı döner.
        if audio_duration <= window_duration:
            return 0.0
            
        # 1. PARAMETRE: RMS Enerjisi (Root Mean Square)
        # Sinyalin genlik değerlerinin karekök ortalamasını alarak fiziksel gürlüğünü (hacmini) çıkarır ve 0-1 arasına normalize eder.
        rms = librosa.feature.rms(y=y)[0]
        rms_norm = librosa.util.normalize(rms)
        
        # 2. PARAMETRE: Onset Zarfi (Saldırı Noktaları)
        # Sinyaldeki ani enerji sıçramalarını (davul vuruşları, gitar atakları) hesaplayıp normalize eder.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_norm = librosa.util.normalize(onset_env)
        
        # 3. NİHAİ SKOR: İki metriğin matrisel toplamı ile genel aktivite yoğunluğunu bulur.
        solenta_score = rms_norm + onset_norm
        
        # 4. KAYAN PENCERE (Rolling Window) ALGORİTMASI
        # Tarama çerçevesinin adım uzunluğunu ve saniye başına düşen kare (frame) sayısını belirler.
        hop_length = 512 
        frames_per_sec = sr / hop_length
        window_frames = int(window_duration * frames_per_sec)
        
        # Sinyalin tamamını belirlediğimiz saniye (window) genişliğindeki bir filtreyle tarayarak
        # her bir olasılık için toplam enerji skorunu konvolüsyon (convolution) işlemi ile çıkarır.
        window = np.ones(window_frames)
        rolling_sum = np.convolve(solenta_score, window, mode='valid')
        
        # Kayan pencerenin en yüksek skoru (maksimum aktivite) bulduğu indeks numarasını çeker.
        best_frame_idx = np.argmax(rolling_sum)
        
        # Matris üzerindeki o indeksi, gerçek zamanlı saniye karşılığına çevirir.
        best_time_sec = librosa.frames_to_time(best_frame_idx, sr=sr, hop_length=hop_length)
        
        # Güvenlik Duvarı (Padding Protection): Seçilen başlama noktasından itibaren 10 saniye eklendiğinde, 
        # dosyanın bitiş sınırını (EOF) aşıp OutOfBounds Error vermesini engeller.
        if best_time_sec + window_duration > audio_duration:
            best_time_sec = max(0.0, audio_duration - window_duration)
            
        print(f"SOLENTA ANALIZ: Otonom segmentasyon tamamlandi. Maksimum enerji {round(best_time_sec, 2)}. saniyede saptandi.")
        return round(best_time_sec, 2)
        
    except Exception as e:
        print(f"SİSTEM HATASI: Heuristik tarama basarisiz. (0.0) atandi. Kod: {e}")
        return 0.0

# =====================================================================
# [SİNYAL İŞLEME MODÜLÜ 2]: Fiziksel Belirleyici (Determinant) Çıkarımı
# =====================================================================
def analyze_audio_determinants(audio_path, start_time=0.0, duration=10.0):
    """
    Neden Eklendi: Görsel üretime yön verecek ana matematiği kurar.
    Sesi körlemesine değil, Olay Tabanlı Bölütleme (Transient/Onset Detection) 
    yöntemiyle sadece vuruş anlarında analiz ederek gürültüyü (noise) eler.
    """
    try:
        # Hatalı parametre girişlerine karşı güvenlik yalıtımı (Fallback)
        if not isinstance(start_time, (int, float)) or start_time < 0:
            start_time = 0.0
        if not isinstance(duration, (int, float)) or duration <= 0 or duration > 10.0:
            duration = 10.0
            
        # Önceden tanımlı bellek optimizasyonlu (LRU) fonksiyon ile ses dosyasını okur.
        y, sr = fetch_audio_buffer(audio_path, start_time, duration)
        
        # Dosyada okunacak herhangi bir frekans verisi yoksa işlemi kilitler.
        if len(y) == 0:
             return None, "SİSTEM HATASI: Sinyal verisi okunamadi.", None, None

        # [HIZLI FOURIER DÖNÜŞÜMÜ (FFT) VE BLOKLAMA YAPILANDIRMASI]
        # Yaklaşık 46 milisaniyelik vuruşları (transient) yakalamak için örnekleme hızına bağlı dinamik bir pencere (window) hesaplanır.
        target_window_ms = 0.046  
        n_fft_ideal = int(sr * target_window_ms)
        
        # Algoritmanın O(n log n) hızında çalışabilmesi için işlemci mimarisine uygun olarak 
        # pencere boyutunu 2'nin en yakın kuvvetine (Power of 2) yuvarlar.
        n_fft = 2 ** int(np.log2(n_fft_ideal)) 
        
        # Bilgi kaybını (spectral leakage) engellemek için %75 oranında örtüşen (overlap) adımlar tanımlar.
        hop_length = n_fft // 4 

        # [TRANSIENT DETECTION: Olay Tabanlı Bölütleme]
        # Tüm sinyali okumak yerine sadece enerjinin patladığı vuruşların kare indekslerini alır.
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        if len(onset_frames) == 0: 
            onset_frames = np.array([0])
            
        # 1. RİTİM VE TEMPO (BPM)
        # Sinyalin tepe noktaları arasındaki periyodik mesafeyi ölçerek dakikadaki vuruş sayısını (BPM) hesaplar.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        # 2. TINI (Spectral Centroid)
        # Sesin parlaklığını (ağırlık merkezini) hesaplar. Frekans spektrumunu çıkarıp vuruş anlarındaki kareleri filtreler.
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        valid_frames = [f for f in onset_frames if f < len(cent)]
        timbre_centroid = float(np.mean(cent[valid_frames])) if valid_frames else float(np.mean(cent))
        
        # 3. PERDE (Dominant Pitch -> Renk Hue Haritalaması)
        # Sinyali zaman uzayından frekans uzayına (STFT) taşır ve mutlak genliklerini alır.
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # DC Offset (Sıfır frekans) hatasını engellemek için spektrumun en alt değerini sıfırlar.
        onset_stft = stft[:, valid_frames] if valid_frames else stft
        power_spectrum = np.sum(onset_stft, axis=1)
        power_spectrum[0] = 0  
        
        # Spektrumdaki enerjinin en yüksek olduğu frekans değerini Hz cinsinden dominant perde olarak belirler.
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        dominant_pitch = float(frequencies[np.argmax(power_spectrum)])
        
        # Bulunan frekansı 0-360 derecelik renk silindirine (Hue) matematiksel olarak sarar (Modulo).
        hue_value = int(dominant_pitch % 360)
        
        # 4. GÜRLÜK (RMS Energy)
        # Vuruş anlarındaki karelerin fiziksel genlik enerjilerini çıkarır ve arayüzdeki sliderlar için 20-150 arasına normalize eder.
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        valid_rms = [f for f in onset_frames if f < len(rms)]
        loudness = float(np.mean(rms[valid_rms])) if valid_rms else float(np.mean(rms))
        normalized_loudness = min(max(int((loudness / 0.3) * 150), 20), 150)
        
        # Yapay zekaya (SOLENTA) yön verecek olan dört fiziksel sabiti çıktı olarak döner.
        return round(bpm, 2), round(timbre_centroid, 2), hue_value, normalized_loudness
    
    except Exception as e:
        return None, f"KRİTİK İŞİTSEL HATA: {str(e)}", None, None

# =====================================================================
# [SİSTEM BÜTÜNLÜK TESTİ (UNIT TEST) BLOĞU]
# Sadece bu dosya tek başına çalıştırıldığında hata ayıklama amaçlı tetiklenir.
# =====================================================================
if __name__ == "__main__":
    import os
    test_file = "test_sarkisi.mp3" 
    
    print(f"SOLENTA AUDIO MOTORU (ONSET TABANLI) KONTROL PROTOKOLÜ")
    if os.path.exists(test_file):
        print("\nAdım 1: Heuristik Enerji Kestirimi (Smart Crop) yürütülüyor...")
        en_iyi_saniye = find_optimal_audio_segment(test_file)
        
        print(f"\nAdım 2: Belirlenen {en_iyi_saniye}. saniyeden itibaren FFT analizi yapılıyor...")
        secilen_sure = 10.0
        bpm_res, timbre_res, hue_res, loud_res = analyze_audio_determinants(test_file, start_time=en_iyi_saniye, duration=secilen_sure)
        print(f"BPM: {bpm_res} | Tını (Centroid): {timbre_res} Hz | Renk Modülü: {hue_res}° | Gürlük Seviyesi: {loud_res}px")
    else:
        print(f"UYARI: '{test_file}' ortamda bulunamadı. Bağımlılıkların testi için arayüzü (app.py) başlatın.")