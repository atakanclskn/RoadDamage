"""
RF-DETR Model Eğitici (Object Detection)
=========================================
RF-DETR modeli ile object detection eğitimi.
Desteklenen boyutlar: Nano, Small, Base, Large
"""

from models.base import BaseTrainer


# RF-DETR boyut sınıfları mapping
RFDETR_SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def _get_rfdetr_class(size: str):
    """Dinamik olarak rfdetr sınıfını import eder."""
    import rfdetr
    class_name = RFDETR_SIZES.get(size.lower())
    if class_name is None:
        raise ValueError(
            f"Bilinmeyen RF-DETR boyutu: '{size}'. "
            f"Geçerli boyutlar: {list(RFDETR_SIZES.keys())}"
        )
    return getattr(rfdetr, class_name)


class RFDETRTrainer(BaseTrainer):
    MODEL_NAME = "rfdetr"
    DESCRIPTION = "RF-DETR Object Detection (Nano/Small/Base/Large)"
    DEFAULT_BATCH_SIZE = 16

    def setup_model(self, **kwargs):
        """RF-DETR modelini seçilen boyutta oluşturur."""
        size = kwargs.get("size", "nano")
        model_cls = _get_rfdetr_class(size)
        self.model = model_cls()
        print(f"📦 RF-DETR {size.upper()} modeli yüklendi.")

    def run_training(self, **kwargs):
        """RF-DETR eğitimini başlatır."""
        dataset_dir = kwargs.get("dataset_dir")
        if not dataset_dir:
            dataset_name = kwargs.get("dataset_name", "BOX-TEST-1-3")
            dataset_dir = self.get_dataset_path(dataset_name)

        experiment_name = kwargs.get("experiment_name", "rfdetr_experiment")

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

        # Opsiyonel parametreler
        if "resolution" in kwargs:
            train_kwargs["resolution"] = kwargs["resolution"]
        if "multi_scale" in kwargs:
            train_kwargs["multi_scale"] = kwargs["multi_scale"]
        if "amp" in kwargs:
            train_kwargs["amp"] = kwargs["amp"]
        if "lr" in kwargs:
            train_kwargs["lr"] = kwargs["lr"]
        if "warmup_epochs" in kwargs:
            train_kwargs["warmup_epochs"] = kwargs["warmup_epochs"]

        self.model.train(**train_kwargs)


# =============================================================
# Doğrudan çalıştırma desteği: python -m models.rfdetr
# =============================================================
if __name__ == "__main__":
    trainer = RFDETRTrainer()
    trainer.train(
        size="nano",
        dataset_dir=r"C:\projects\RoadDamage\BOX-TEST-1-3",
        epochs=100,
        batch_size=16,
        grad_accum_steps=4,
        workers=4,
        resolution=640,
        multi_scale=True,
        amp=True,
        lr=0.0001,
        warmup_epochs=3.0,
        patience=25,
        checkpoint_interval=5,
        experiment_name="rfdetr_l_BOXTEST",
    )
