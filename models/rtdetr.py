"""
RT-DETR Model Eğitici (Placeholder)
=====================================
RT-DETR modeli ile object detection eğitimi.
Bu dosya gelecekte kullanım için hazırlanmıştır.
"""

from models.base import BaseTrainer


class RTDETRTrainer(BaseTrainer):
    MODEL_NAME = "rtdetr"
    DESCRIPTION = "RT-DETR Object Detection (Henüz eklenmedi)"
    DEFAULT_BATCH_SIZE = 16

    def setup_model(self, **kwargs):
        """RT-DETR modelini yükler."""
        # TODO: RT-DETR model kurulumu eklenecek
        raise NotImplementedError(
            "RT-DETR eğitimi henüz implement edilmedi. "
            "Bu dosyayı kendi ihtiyaçlarına göre doldur."
        )

    def run_training(self, **kwargs):
        """RT-DETR eğitimini başlatır."""
        # TODO: RT-DETR eğitim döngüsü eklenecek
        raise NotImplementedError(
            "RT-DETR eğitimi henüz implement edilmedi."
        )


# =============================================================
# Doğrudan çalıştırma: python -m models.rtdetr
# =============================================================
if __name__ == "__main__":
    trainer = RTDETRTrainer()
    trainer.train()
