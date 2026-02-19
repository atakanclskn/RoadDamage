"""
Merkezi Eğitim Başlatıcı
=========================
Tüm modeller tek bir komutla çalıştırılabilir.

Kullanım:
    python train.py --model yolo26 --config configs/yolo26_box.yaml
    python train.py --model rfdetr --size nano --dataset BOX-TEST-1-3
    python train.py --model rfdetr-seg --size small --dataset SEG-TEST-1-1
    python train.py --list   # Mevcut modelleri listele
"""

import argparse
import sys
from models import get_trainer, list_models


def main():
    parser = argparse.ArgumentParser(
        description="🛣️ RoadDamage - Model Eğitim Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python train.py --model yolo26 --weight yolo26s.pt --dataset-yaml data.yaml --epochs 100
  python train.py --model rfdetr --size nano --dataset-dir BOX-TEST-1-3 --epochs 100
  python train.py --model rfdetr-seg --size small --dataset-dir SEG-TEST-1-1 --epochs 100
  python train.py --list
        """,
    )

    parser.add_argument("--model", type=str, help="Kullanılacak model (yolo26, rfdetr, rfdetr-seg, rtdetr)")
    parser.add_argument("--list", action="store_true", help="Mevcut modelleri listele")

    # Ortak parametreler
    parser.add_argument("--epochs", type=int, default=100, help="Epoch sayısı (varsayılan: 100)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch boyutu")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="Veri yükleme worker sayısı")
    parser.add_argument("--experiment", type=str, default=None, help="Deney adı")

    # YOLO-spesifik
    parser.add_argument("--weight", type=str, default=None, help="Pretrained ağırlık dosyası")
    parser.add_argument("--dataset-yaml", type=str, default=None, help="YOLO data.yaml yolu")
    parser.add_argument("--imgsz", type=int, default=640, help="Görüntü boyutu")
    parser.add_argument("--optimizer", type=str, default="MuSGD", help="Optimizer")

    # RF-DETR-spesifik
    parser.add_argument("--size", type=str, default=None, help="Model boyutu (nano, small, base, large)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="COCO formatında dataset dizini")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation adımları")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epoch sayısı")
    parser.add_argument("--resolution", type=int, default=None, help="RF-DETR çözünürlük")
    parser.add_argument("--multi-scale", action="store_true", help="Multi-scale eğitim")
    parser.add_argument("--amp", action="store_true", help="Automatic Mixed Precision")

    args = parser.parse_args()

    # Model listesi
    if args.list:
        list_models()
        sys.exit(0)

    if args.model is None:
        parser.print_help()
        print("\n❌ --model parametresi gerekli. Mevcut modeller için: python train.py --list")
        sys.exit(1)

    # Trainer oluştur
    TrainerClass = get_trainer(args.model)
    trainer = TrainerClass()

    # Kwargs oluştur
    kwargs = {
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
    }

    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.experiment:
        kwargs["experiment_name"] = args.experiment

    # Model-spesifik kwargs
    if args.model == "yolo26":
        if args.weight:
            kwargs["weight"] = args.weight
        if args.dataset_yaml:
            kwargs["dataset_yaml"] = args.dataset_yaml
        kwargs["imgsz"] = args.imgsz
        kwargs["optimizer"] = args.optimizer

    elif args.model in ("rfdetr", "rfdetr-seg"):
        if args.size:
            kwargs["size"] = args.size
        if args.dataset_dir:
            kwargs["dataset_dir"] = args.dataset_dir
        kwargs["grad_accum_steps"] = args.grad_accum
        if args.lr:
            kwargs["lr"] = args.lr
        if args.warmup_epochs:
            kwargs["warmup_epochs"] = args.warmup_epochs
        if args.resolution:
            kwargs["resolution"] = args.resolution
        if args.multi_scale:
            kwargs["multi_scale"] = True
        if args.amp:
            kwargs["amp"] = True

    # Eğitimi başlat
    trainer.train(**kwargs)


if __name__ == "__main__":
    main()
