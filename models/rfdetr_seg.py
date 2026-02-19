"""
RF-DETR Segmentation Model Eğitici
====================================
RF-DETR modeli ile instance segmentation eğitimi.
Desteklenen boyutlar: Small
"""

from models.base import BaseTrainer


# RF-DETR-SEG boyut sınıfları mapping
RFDETR_SEG_SIZES = {
    "small": "RFDETRSegSmall",
}


def _get_rfdetr_seg_class(size: str):
    """Dinamik olarak rfdetr segmentation sınıfını import eder."""
    import rfdetr
    class_name = RFDETR_SEG_SIZES.get(size.lower())
    if class_name is None:
        raise ValueError(
            f"Bilinmeyen RF-DETR-SEG boyutu: '{size}'. "
            f"Geçerli boyutlar: {list(RFDETR_SEG_SIZES.keys())}"
        )
    return getattr(rfdetr, class_name)


class RFDETRSegTrainer(BaseTrainer):
    MODEL_NAME = "rfdetr-seg"
    DESCRIPTION = "RF-DETR Instance Segmentation (Small)"
    DEFAULT_BATCH_SIZE = 12

    def setup_model(self, **kwargs):
        """RF-DETR Segmentation modelini oluşturur."""
        size = kwargs.get("size", "small")
        model_cls = _get_rfdetr_seg_class(size)
        self.model = model_cls()
        print(f"[+] RF-DETR-SEG {size.upper()} modeli yuklendi.")

    def run_training(self, **kwargs):
        """RF-DETR Segmentation eğitimini başlatır."""
        dataset_dir = kwargs.get("dataset_dir")
        if not dataset_dir:
            dataset_name = kwargs.get("dataset_name", "SEG-TEST-1-1")
            dataset_dir = self.get_dataset_path(dataset_name)

        experiment_name = kwargs.get("experiment_name", "rfdetr_seg_experiment")

        train_kwargs = {
            "dataset_dir": dataset_dir,
            "epochs": kwargs.get("epochs", self.DEFAULT_EPOCHS),
            "batch_size": kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE),
            "grad_accum_steps": kwargs.get("grad_accum_steps", 4),
            "num_workers": kwargs.get("workers", 4),
            "checkpoint_interval": kwargs.get("checkpoint_interval", 5),
            "tensorboard": kwargs.get("tensorboard", True),
            "early_stopping_patience": kwargs.get("patience", 25),
            "output_dir": self.get_output_dir(experiment_name),
        }

        self.model.train(**train_kwargs)


# =============================================================
# Doğrudan çalıştırma desteği: python -m models.rfdetr_seg
# =============================================================
if __name__ == "__main__":
    trainer = RFDETRSegTrainer()
    trainer.train(
        size="small",
        dataset_dir=r"C:\projects\RoadDamage\datasets\SEG-TEST-1-1(COCO)",
        epochs=100,
        batch_size=12,
        grad_accum_steps=4,
        workers=4,
        patience=25,
        checkpoint_interval=5,
        experiment_name="rfdetr_SEGTESTl",
    )
