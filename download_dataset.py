"""
Dataset İndirme Aracı
======================
Roboflow'dan dataset indirir. API anahtarı .env dosyasından okunur.

Kullanım:
    python download_dataset.py                     # Tüm projeleri indir
    python download_dataset.py --project seg-test-1 --format coco
"""

import argparse
from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECTS,
    DATASETS_DIR,
)


def download_dataset(project_key: str, export_format: str = "coco"):
    """Tek bir projeyi indirir."""
    if ROBOFLOW_API_KEY is None:
        raise ValueError(
            "❌ ROBOFLOW_API_KEY bulunamadı!\n"
            "   .env dosyasına API anahtarını ekle veya .env.example dosyasını kopyala."
        )

    project_info = ROBOFLOW_PROJECTS.get(project_key)
    if project_info is None:
        available = ", ".join(ROBOFLOW_PROJECTS.keys())
        raise ValueError(
            f"Bilinmeyen proje: '{project_key}'. Mevcut projeler: {available}"
        )

    print(f"\n📥 İndiriliyor: {project_key} ({export_format} formatında)")
    print(f"   Açıklama: {project_info['description']}")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(project_info["project_name"])
    version = project.version(project_info["version"])

    # datasets/ klasörüne indir
    dataset = version.download(export_format, location=str(DATASETS_DIR / f"{project_key}-v{project_info['version']}({export_format})"))
    
    print(f"✅ Tamamlandı: {project_key}")
    return dataset


def download_all():
    """Tüm projeleri varsayılan formatlarda indirir."""
    for key, info in ROBOFLOW_PROJECTS.items():
        for fmt in info["formats"]:
            try:
                download_dataset(key, fmt)
            except Exception as e:
                print(f"⚠️  {key} ({fmt}) indirilemedi: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow'dan dataset indir")
    parser.add_argument("--project", type=str, help="Proje adı (ör: seg-test-1)")
    parser.add_argument("--format", type=str, default="coco", help="Format (coco, yolo26, vb.)")
    parser.add_argument("--all", action="store_true", help="Tüm projeleri indir")

    args = parser.parse_args()

    if args.all or args.project is None:
        download_all()
    else:
        download_dataset(args.project, args.format)
