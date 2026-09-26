# SOLENTA: Synesthetic Generative Laboratory
### *İşitsel-Görsel Sinestezi ve Üretken Sanat Laboratuvarı*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Blocks-orange.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa%20DSP-yellow.svg)](https://librosa.org/)
[![Academic Project](https://img.shields.io/badge/Academic-Graduation%20Project-informational.svg)](docs/PROJE_RAPORU.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Abstract / Proje Özeti

**SOLENTA (Synesthetic Generative Laboratory)**, işitsel sinyalleri (müzik ve ses frekansları) plastik sanat eserleriyle (özellikle Kübist akım) ve yapay görme algoritmalarıyla birleştiren, çok disiplinli (**Multidisciplinary**) bir görüntü işleme ve üretken yapay zeka (**Generative AI**) bitirme projesidir.

Proje, yalnızca hazır bir difüzyon sarmalayıcısı (wrapper) olmak yerine; **dijital sinyal işleme (DSP)**, **geometrik hesaplamalı topoloji (Voronoi şeması)**, **renk kuramı ve Gestalt algısı (Eşzamanlı Kontrast, K-Means++)** ve **çift motorlu nöral sentez (Neural Style Transfer + ControlNet Depth-to-Image with LCM Turbo)** aşamalarını birbirine bağlayan uçtan uca açıklanabilir (**Explainable / White-Box**) bir bilişsel mimari sunar.

> [!NOTE]
> Bu çalışma, bilgisayar teknolojileri ve yapay zekâ bitirme projesi kapsamında geliştirilmiş, yazılım mimarisi ve sistem analizi metodolojileriyle belgelendirilmiş kapsamlı bir araştırma ve geliştirme çalışmasıdır.  
> 📄 Detaylı akademik rapor için: **[Sistem Analizi ve Tasarımı Raporu (docs/PROJE_RAPORU.md)](docs/PROJE_RAPORU.md)**

---

## 🏗️ System Architecture / Mimari İş Akışı

Aşağıdaki şema, ham bir ses dosyasının veya kullanıcı görselinin SOLENTA boru hattından geçerek 1080p nihai sanatsal çıktıya dönüşüm serüvenini göstermektedir:

```mermaid
flowchart TD
    subgraph Audio_Processing ["1. İşitsel Sinyal İşleme (audio_engine)"]
        A["Ham Ses Dosyası (.mp3 / .wav)"] --> B["Nyquist Örnekleme (8 kHz) & Kayan Pencere (Smart Crop)"]
        B --> C["STFT / Hızlı Fourier Dönüşümü (FFT) & Onset Algılama"]
        C --> D["İşitsel Belirleyiciler:\n• Ritim (BPM)\n• Tını (Spectral Centroid Hz)\n• Perde (Dominant Pitch -> Hue)\n• Gürlük (RMS Enerji)"]
    end

    subgraph Topological_Canvas ["2. Topolojik İskelet & Renk Kuramı"]
        D --> E["Akustik Voronoi Şeması\n(BPM Hücre Yoğunluğu & Deterministik Tohum)"]
        D --> F["Renk Eşleme (K-Means++ & Eşzamanlı Kontrast)\nFrekans Rengine Senkronizasyon (Hue Wrapping)"]
        E & F --> G["İşitsel-Görsel Kanvas"]
    end

    subgraph Dual_Engine ["3. Çift Motorlu Nöral Sentez (Dual-Engine)"]
        G --> H{"Sentez Modu Seçimi"}
        H -->|"Light Mod (CPU/GPU)"| I["Arbitrary NST (TensorFlow Hub / VGG-19)\n• Spatial Content & Low-level Style Fusion\n• SOLENTA Signature Enhancement (Luminance & CLAHE)"]
        H -->|"Pro Mod (CUDA)"| J["MiDaS Monocular Depth Estimation\n+ ControlNet Depth Conditioning\n+ SD 1.5 with LCM-LoRA (4-Step Turbo)"]
    end

    subgraph Post_Processing ["4. Canlı Sinyal-Piksel Modifikasyonu & Veri Gölü"]
        I & J --> K["Canlı Filtreler:\n• Tınıya Dayalı Dinamik Konvolüsyon (Laplacian Unsharp)\n• Adaptif Histogram Bükülmesi (Gamma & CLAHE)\n• Vektörel Renk Kuantizasyonu (K-Means++)"]
        K --> L["Lanczos-4 Süper Çözünürlük (1080p Crisp Output)"]
        L --> M["ETL & Veri Ambarı (solenta_logger)\nNoSQL/JSON Parametre Kaydı & Fine-Tuning Veri Gölü"]
    end
```

---

## 🔬 Core Engineering & Mathematical Foundations

### 1. İşitsel Sinyal Analizi (`audio_engine.py`)
- **Heuristik Akıllı Kesim (Smart Crop):** 8000 Hz'e indirgenmiş (downsampled) sinyal üzerinde RMS enerjisi ve Onset zarfı matrisel olarak toplanıp kayan pencere (rolling window) konvolüsyonu ile taranır. Şarkının enerjisinin doruğa ulaştığı (kreşendo/drop) saniye otonom tespit edilir.
- **Spektral Tını & Frekans Haritalaması:** STFT ve FFT analizleri ile sesin spektral ağırlık merkezi (Centroid) Hz cinsinden hesaplanır. En yüksek enerjiye sahip dominant perde frekansı $[0, 360^\circ]$ HSV renk silindirine sarılarak (modulo arithmetic) müziğin baskın renk tonu (Hue) saptanır.

### 2. Akustik Voronoi Kanvası (`app.py`)
- Müziğin fiziksel parametreleri bir araya getirilerek deterministik bir tohum (seed) üretilir ($BPM + Timbre + Hue + Loudness$).
- Voronoi poligonlarının sayısı BPM ve gürlük ile orantılı şekilde dinamik belirlenir. Kenar kalınlıkları ve Gaussian analog grenleri gürlük seviyesine göre boyanır.

### 3. Renk Algısı ve Kübist Akım Eşleme (`color_engine.py` & `data_manager.py`)
- **K-Means++:** 100x100 boyutuna indirgenmiş görsel matrisinde $k=3$ kümeleme yapılarak baskın zemin rengi saptanır.
- **Simultaneous Contrast (Eşzamanlı Kontrast):** Gestalt algı kuramına dayanarak tespit edilen zemin renginin insan gözünde oluşturacağı tamamlayıcı renk illüzyonu (sıcaklık/soğuma etkisi) hesaplanır.
- **2D Fourier Görsel Tınısı:** Eserlerin dokusal frekans genliği $\mathcal{F}_{2D}$ dönüşümü ile hesaplanarak müzikal tını ile eşleştirilir.

### 4. Çift Motorlu Hibrit Yapay Zeka (`nst_engine.py` & `diffusion_engine.py`)
- **Light Mod (Mobil/Hafif Sistemler):** Google Magenta Arbitrary Image Stylization modeli üzerinden ileri besleme (Feed-Forward). NST modellerindeki renk solmasını (Color Shift) önlemek adına LAB renk uzayında Luminance korumalı **SOLENTA Signature Enhancement** rötüşü uygulanır.
- **Pro Mod (Yüksek Başarımlı CUDA):** Intel MiDaS ile derinlik haritası (Z-Axis Depth Map) çıkarılır. Stable Diffusion 1.5, ControlNet Depth koşullandırması ve Latent Consistency Model (LCM) LoRA çekirdeği ile 4-6 adımda fotogerçekçi sentez üretir.
- **Otonom Model İndirme:** Sistem çalıştırıldığında yerel modeller yoksa HuggingFace Hub ve TFHub üzerinden eksik ağırlıkları otomatik olarak çeker ve önbelleğe alır.

---

## 📂 Repository Structure / Dizin Yapısı

```
Solenta/
├── app.py                   # Gradio tabanlı ana orkestrasyon ve arayüz katmanı
├── audio_engine.py          # Librosa tabanlı DSP, STFT, FFT, Smart Crop motoru
├── color_engine.py          # K-Means++ ve Eşzamanlı Kontrast renk analiz motoru
├── data_manager.py          # Veri seti sınıflandırma, 2D-FFT görsel tını eşleme, Safe-IO
├── diffusion_engine.py      # MiDaS Depth + ControlNet + LCM Turbo Pro motoru
├── filter_engine.py         # Laplacian dinamik konvolüsyon & K-Means kuantizasyon
├── nst_engine.py            # Magenta NST (VGG-19) & LAB renk restorasyonu
├── solenta_logger.py        # Deterministik JSON ETL / Data Lake loglama motoru
├── download_dataset.py      # Genişletilmiş veri ambarı (2.300+ eser) indirme aracı
├── requirements.txt         # Standart ve temiz bağımlılık listesi (UTF-8)
├── LICENSE                  # MIT Açık Kaynak Lisansı
├── .gitignore               # Model, venv, build ve 100MB+ dosyaları süzen filtre
├── docs/
│   └── PROJE_RAPORU.md      # Detaylı Sistem Analizi ve Tasarımı Raporu (UML & DFD)
└── dataset/
    ├── analitik/            # Örnek analitik kübist referans eserler
    ├── sentetik/            # Örnek sentetik kübist referans eserler
    ├── proto/               # Örnek proto-kübist referans eserler
    └── audio/               # Hazır test müzikleri (Paganini, Beethoven, Hip-Hop vb.)
```

---

## 🚀 Installation & Quick Start / Kurulum ve Çalıştırma

### 1. Gereksinimler
- **İşletim Sistemi:** Windows 10/11, Linux, macOS
- **Python:** 3.10 veya 3.11 önerilir
- **Donanım:** 
  - *Light Mod:* Herhangi bir modern CPU (veya entegre GPU)
  - *Pro Mod:* En az 4-6 GB VRAM destekli NVIDIA GPU (CUDA 12+)

### 2. Klonlama ve Sanal Ortam Kurulumu
```bash
# Depoyu klonlayın
git clone https://github.com/Warkrone34/Solenta.git
cd Solenta

# Sanal ortam oluşturup aktive edin
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Bağımlılıkların Yüklenmesi
```bash
# Temel kütüphaneleri yükleyin
pip install -r requirements.txt

# (Opsiyonel) Pro Mod için CUDA destekli PyTorch yüklemesi:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Uygulamayı Başlatma
```bash
python app.py
```
> Sunucu çalıştığında varsayılan tarayıcınızda otomatik olarak `http://127.0.0.1:7860` adresinde açılacaktır. İlk çalıştırmada Pro Mod seçilirse gerekli model ağırlıkları HuggingFace üzerinden otomatik indirilecektir.

---

## 🎨 Eser Havuzu (Dataset) Hakkında

Sistem iki farklı veri boyutuyla tam uyumlu çalışmaktadır:

1. **Hazır Çekirdek Paket (Out-of-the-Box Demo):**
   - Repoyu klonladığınızda `dataset/` dizininde gelen seçkin referans Kübist eserler ve test müzikleri sayesinde **ek hiçbir dosya indirmeden** sistemi doğrudan başlatabilir (`python app.py`), tüm üretim ve stil aktarım aşamalarını test edebilirsiniz.

2. **Genişletilmiş 2.300+ Eserlik Tam Veri Ambarı (Full Research Dataset):**
   - Tez ve araştırma kapsamında derlenen tam Kübist sanat veri ambarı (Analitik, Sentetik ve Proto-Kübizm olmak üzere 2.300'den fazla eser), GitHub dosya boyutu limitleri nedeniyle ana deponun dışında tutulmaktadır.
   - **Releases Üzerinden İndirme:** Bu arşivi doğrudan reponun **[Releases](https://github.com/Warkrone34/Solenta/releases)** sekmesinden (`solenta_dataset.zip`) tek tıkla indirebilir ve arşiv içeriğini projenin `dataset/` klasörüne çıkartabilirsiniz.
   - **Otomatik Kurulum Aracı:** Alternatif olarak proje ana dizininde terminalden şu komutu çalıştırmanız yeterlidir:
     ```bash
     python download_dataset.py
     ```

---

## 📊 Academic Context & Contributors / Akademik Bağlam ve Katkılar

- **Proje Türü:** Bilgisayar Programcılığı / Bitirme Tezi
- **Geliştirici:** Ömer Faruk SAĞLAM
- **Akademik Danışman:** Refik Tanju SİRMEN
- **Katkıda Bulunan:** Deniz Tan SİRMEN
- **Detaylı Sistem Analizi Raporu:** [docs/PROJE_RAPORU.md](docs/PROJE_RAPORU.md)

### 🙏 Teşekkürler / Acknowledgements
Bu çalışmanın araştırma, sistem analizi ve algoritmik geliştirme süreçlerindeki paha biçilmez rehberlikleri, akademik vizyonları ve destekleri için değerli danışmanım **Sayın Refik Tanju SİRMEN**'e ve projeye sunduğu katkılardan ötürü **Sayın Deniz Tan SİRMEN**'e en içten teşekkürlerimi ve saygılarımı sunarım.

---

## 🔭 Future Work & Roadmap / Gelecek Çalışmalar

- [ ] **Multimodal Latent Alignment:** Kural tabanlı (heuristic) BPM-akım eşleşmesinin, **CLAP (Contrastive Language-Audio Pretraining)** ve **CLIP** gömme uzaylarında kosinüs benzerliği ile uçtan uca öğrenilmesi.
- [ ] **Real-time Live Audio Reactive Stream:** Canlı mikrofon akışından anlık FFT alarak video kareleri arası pürüzsüz interpolasyon sağlayan dinamik görselleştirici.
- [ ] **Docker & Cloud Deployment:** HuggingFace Spaces ve Docker imajları ile tek tıkla bulut dağıtımı.

---

## 📄 License / Lisans

Bu proje [MIT Lisansı](LICENSE) altında açık kaynak olarak sunulmaktadır. Eğitim, araştırma ve geliştirme amaçlı kullanıma tamamen açıktır.
