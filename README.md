# RoadDamage - Automatic Road Damage Detection

Yol yuzeyindeki hasarlari (catlak, cukur, kapak) tespit eden derin ogrenme modelleri.

## Proje Yapisi

```
RoadDamage/
├── .env                    # API anahtarlari (git'e yuklenmez)
├── .env.example            # Ornek ortam degiskenleri
├── .gitignore
├── config.py               # Merkezi konfigurasyon
├── download_dataset.py     # Roboflow'dan dataset indirme
├── train.py                # Merkezi egitim baslatici (CLI + interaktif)
├── requirements.txt
│
├── models/                 # Model egiticileri
│   ├── __init__.py         # Model registry
│   ├── base.py             # BaseTrainer (temel sinif)
│   ├── yolo26.py           # YOLO26 egitici
│   ├── rfdetr.py           # RF-DETR detection egitici
│   ├── rfdetr_seg.py       # RF-DETR segmentation egitici
│   └── rtdetr.py           # RT-DETR egitici (placeholder)
│
├── datasets/               # Indirilen datasetler (git'e yuklenmez)
├── weights/                # Pretrained agirliklar (git'e yuklenmez)
└── runs/                   # Egitim ciktilari (git'e yuklenmez)
```

## Kurulum

### 1. Gerekli Paketler

```bash
pip install -r requirements.txt
```

### 2. API Anahtari

`.env.example` dosyasini `.env` olarak kopyala ve API anahtarini ekle:

```bash
cp .env.example .env
```

`.env` dosyasini duzenle ve `ROBOFLOW_API_KEY` degerini gir.

### 3. Dataset Indirme

```bash
python download_dataset.py                          # Tum datasetleri indir
python download_dataset.py --project seg-test-1     # Tek proje indir
python download_dataset.py --project box-test-1 --format yolo26
```

## Egitim

### Interaktif Menu (Onerilen)

Hicbir arguman vermeden calistirirsan adim adim menu gelir:

```bash
python train.py
```

Menu sirasiyla model, dataset, boyut ve parametreleri sorar, ozeti gosterir, onaylarsan baslatir.

### CLI Argumanlariyla

```bash
# Mevcut modelleri, datasetleri ve agirliklari listele
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

## Desteklenen Modeller

| Model       | Gorev                  | Boyutlar                    | Durum   |
|-------------|------------------------|-----------------------------|---------|
| yolo26      | Detection/Segmentation | n, s, m, l, x              | Hazir   |
| rfdetr      | Object Detection       | nano, small, base, large    | Hazir   |
| rfdetr-seg  | Instance Segmentation  | small                       | Hazir   |
| rtdetr      | Object Detection       | -                           | Yakin   |

## Yeni Model Ekleme

1. `models/` altina yeni bir dosya olustur (orn: `models/yeni_model.py`)
2. `BaseTrainer`'dan turet:

```python
from models.base import BaseTrainer

class YeniModelTrainer(BaseTrainer):
    MODEL_NAME = "yeni-model"
    DESCRIPTION = "Yeni model aciklamasi"

    def setup_model(self, **kwargs):
        # Modeli yukle
        pass

    def run_training(self, **kwargs):
        # Egitimi baslat
        pass
```

3. `models/__init__.py` dosyasindaki `AVAILABLE_MODELS` sozlugune ekle

## Siniflar

| ID | Sinif         | Aciklama         |
|----|---------------|------------------|
| 0  | cover-kapak   | Rogar/baca kapagi|
| 1  | crack-catlak  | Yol catlagi      |
| 2  | pothole-cukur | Yol cukuru       |

## Lisans

CC BY 4.0
