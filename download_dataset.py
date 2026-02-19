"""
Dataset Indirme Araci
======================
Roboflow'dan dataset indirir. API anahtari .env dosyasindan okunur.
Argumansin calistirirsan interaktif menu gelir.

Kullanim:
    python download_dataset.py                          # Interaktif menu
    python download_dataset.py --project seg-test-1     # Direkt indir
    python download_dataset.py --all                    # Tumunu indir
"""

import argparse
import sys
from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECTS,
    DATASETS_DIR,
)


def check_api_key():
    """API key kontrolu."""
    if ROBOFLOW_API_KEY is None:
        print("[X] ROBOFLOW_API_KEY bulunamadi!")
        print("    .env dosyasina API anahtarini ekle.")
        print("    Ornek: .env.example dosyasini kopyala.")
        sys.exit(1)


def download_dataset(project_key: str, export_format: str = "coco"):
    """Tek bir projeyi indirir."""
    check_api_key()

    project_info = ROBOFLOW_PROJECTS.get(project_key)
    if project_info is None:
        available = ", ".join(ROBOFLOW_PROJECTS.keys())
        raise ValueError(
            f"Bilinmeyen proje: '{project_key}'. Mevcut projeler: {available}"
        )

    save_name = f"{project_key}-v{project_info['version']}({export_format})"
    save_path = DATASETS_DIR / save_name

    print(f"\n  Proje   : {project_key}")
    print(f"  Format  : {export_format}")
    print(f"  Aciklama: {project_info['description']}")
    print(f"  Hedef   : datasets/{save_name}")
    print(f"  Indiriliyor...")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(project_info["project_name"])
    version = project.version(project_info["version"])
    dataset = version.download(export_format, location=str(save_path))

    print(f"  [OK] Tamamlandi.\n")
    return dataset


def download_all():
    """Tum projeleri varsayilan formatlarda indirir."""
    check_api_key()
    for key, info in ROBOFLOW_PROJECTS.items():
        for fmt in info["formats"]:
            try:
                download_dataset(key, fmt)
            except Exception as e:
                print(f"  [!] {key} ({fmt}) indirilemedi: {e}")


def interactive_menu():
    """Interaktif indirme menusu."""
    check_api_key()

    print("\n" + "=" * 50)
    print("  DATASET INDIRME")
    print("=" * 50)

    # Proje sec
    print("\n  Mevcut projeler:\n")
    projects = list(ROBOFLOW_PROJECTS.items())
    for i, (key, info) in enumerate(projects, 1):
        print(f"  [{i}] {key:<20} {info['description']}")
    print(f"  [{len(projects)+1}] Tumunu indir")

    while True:
        try:
            choice = int(input(f"\n  Secimin (1-{len(projects)+1}): "))
            if 1 <= choice <= len(projects) + 1:
                break
            print(f"  [X] 1-{len(projects)+1} arasi bir sayi gir.")
        except ValueError:
            print("  [X] Gecerli bir sayi gir.")

    # Tumunu indir
    if choice == len(projects) + 1:
        print("\n  Tum projeler indiriliyor...\n")
        download_all()
        return

    # Tek proje secildi
    selected_key = projects[choice - 1][0]
    selected_info = projects[choice - 1][1]

    # Format sec
    formats = selected_info["formats"]
    print(f"\n  Format sec:\n")
    for i, fmt in enumerate(formats, 1):
        print(f"  [{i}] {fmt}")

    while True:
        try:
            fmt_choice = int(input(f"\n  Secimin (1-{len(formats)}): "))
            if 1 <= fmt_choice <= len(formats):
                break
            print(f"  [X] 1-{len(formats)} arasi bir sayi gir.")
        except ValueError:
            print("  [X] Gecerli bir sayi gir.")

    selected_format = formats[fmt_choice - 1]

    # Indir
    download_dataset(selected_key, selected_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow'dan dataset indir")
    parser.add_argument("--project", type=str, help="Proje adi (seg-test-1, box-test-1)")
    parser.add_argument("--format", type=str, default="coco", help="Format (coco, yolo26)")
    parser.add_argument("--all", action="store_true", help="Tum projeleri indir")

    args = parser.parse_args()

    if args.all:
        download_all()
    elif args.project:
        download_dataset(args.project, args.format)
    else:
        interactive_menu()
