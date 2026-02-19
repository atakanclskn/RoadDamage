# 🛣️ RoadDamage - Yol Hasar Tespiti

Yol yüzeyindeki hasarları (çatlak, çukur, kapak) tespit eden derin öğrenme modelleri.

## 📁 Proje Yapısı

```
RoadDamage/
├── .env                    # API anahtarları (git'e yüklenmez)
├── .env.example            # Örnek ortam değişkenleri
├── .gitignore              # Git'e yüklenmeyecek dosyalar
├── config.py               # Merkezi konfigürasyon
├── download_dataset.py     # Roboflow'dan dataset indirme
├── train.py                # Merkezi eğitim başlatıcı (CLI)
│
├── models/                 # Model eğiticileri
│   ├── __init__.py         # Model registry
│   ├── base.py             # BaseTrainer (temel sınıf)
│   ├── yolo26.py           # YOLO26 eğitici
│   ├── rfdetr.py           # RF-DETR detection eğitici
│   ├── rfdetr_seg.py       # RF-DETR segmentation eğitici
│   └── rtdetr.py           # RT-DETR eğitici (placeholder)
│
├── datasets/               # İndirilen datasetler (git'e yüklenmez)
├── weights/                # Pretrained ağırlıklar (git'e yüklenmez)
└── runs/                   # Eğitim çıktıları (git'e yüklenmez)
```

## 🚀 Kurulum

### 1. Gerekli Paketler
```bash
pip install ultralytics rfdetr roboflow python-dotenv
```

### 2. API Anahtarı
`.env.example` dosyasını `.env` olarak kopyala ve API anahtarını ekle:
```bash
cp .env.example .env
# .env dosyasını düzenle ve ROBOFLOW_API_KEY değerini gir
```

### 3. Dataset İndirme
```bash
python download_dataset.py                          # Tüm datasetleri indir
python download_dataset.py --project seg-test-1     # Tek proje indir
python download_dataset.py --project box-test-1 --format yolo26
```

## 🏋️ Eğitim

### CLI ile (Önerilen)
```bash
# Mevcut modelleri listele
python train.py --list

# YOLO26
python train.py --model yolo26 \
    --weight yolo26s.pt \
    --dataset-yaml datasets/box-test-1-v3(yolo26)/data.yaml \
    --epochs 100 --batch-size 48

# RF-DETR Detection
python train.py --model rfdetr \
    --size nano \
    --dataset-dir datasets/box-test-1-v3(coco) \
    --epochs 100 --batch-size 16 --amp --multi-scale

# RF-DETR Segmentation
python train.py --model rfdetr-seg \
    --size small \
    --dataset-dir datasets/seg-test-1-v1(coco) \
    --epochs 100 --batch-size 12
```

### Doğrudan Modül Çalıştırma
```bash
python -m models.yolo26
python -m models.rfdetr
python -m models.rfdetr_seg
```

## ➕ Yeni Model Ekleme

1. `models/` altına yeni bir dosya oluştur (ör: `models/yeni_model.py`)
2. `BaseTrainer`'dan türet:
```python
from models.base import BaseTrainer

class YeniModelTrainer(BaseTrainer):
    MODEL_NAME = "yeni-model"
    DESCRIPTION = "Yeni model açıklaması"

    def setup_model(self, **kwargs):
        # Modeli yükle
        pass

    def run_training(self, **kwargs):
        # Eğitimi başlat
        pass
```
3. `models/__init__.py` dosyasındaki `AVAILABLE_MODELS` sözlüğüne ekle

## 📊 Sınıflar
| ID | Sınıf | Açıklama |
|----|-------|----------|
| 0  | cover-kapak | Rögar/baca kapağı |
| 1  | crack-catlak | Yol çatlağı |
| 2  | pothole-cukur | Yol çukuru |
