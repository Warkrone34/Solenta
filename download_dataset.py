"""
SOLENTA - Eser Havuzu (Dataset) İndirme ve Kurulum Aracı
-------------------------------------------------------
Bu araç, SOLENTA için derlenen 2.300+ eserlik genişletilmiş
Kübist sanat veri ambarını indirip yerel 'dataset' klasörüne açar.
"""

import os
import sys
import zipfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# GitHub Releases veya HuggingFace üzerindeki arşiv bağlantısı
DATASET_RELEASE_URL = "https://github.com/Warkrone34/Solenta/releases/download/v1.0.0/solenta_dataset.zip"

def download_and_extract_dataset(url=DATASET_RELEASE_URL):
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "solenta_dataset.zip")
    
    print("=" * 60)
    print("SOLENTA: Genişletilmiş Eser Havuzu İndirme Aracı")
    print("=" * 60)
    
    if os.path.exists(zip_path):
        print(f"[*] Yerel arşiv bulundu: {zip_path}")
    else:
        print(f"[*] Arşiv indiriliyor: {url}")
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, int((downloaded / total_size) * 100))
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\rİlerleme: %{percent} [{mb_down:.1f} MB / {mb_total:.1f} MB]")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
            print("\n[*] İndirme tamamlandı.")
        except Exception as e:
            print(f"\n[i] Genişletilmiş 2.300+ eser arşivi bulut bağlantısı henüz aktif değil veya çevrimdışı ({e}).")
            print("[✓] SOLENTA, repo ile birlikte gelen çekirdek referans eser havuzuyla (dataset/) tam fonksiyonel olarak çalışmaya hazırdır.")
            print("[✓] 'python app.py' komutuyla projeyi hemen başlatabilirsiniz.")
            print("[i] Yerel arşiviniz varsa: 'solenta_dataset.zip' dosyasını proje ana dizinine koyup bu betiği tekrar çalıştırabilirsiniz.")
            return False

    print(f"[*] Arşiv 'dataset' klasörüne ayıklanıyor...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("[*] Eser havuzu başarıyla kuruldu!")
        return True
    except Exception as e:
        print(f"[!] Arşiv açılırken hata: {e}")
        return False

if __name__ == "__main__":
    download_and_extract_dataset()
