import subprocess
import sys
from pathlib import Path

from config import set_site

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"

print("=" * 60)
print(" BUILDING DAMAGE ASSESSMENT SYSTEM ")
print("=" * 60)

print("\nChoisissez une zone :\n")
print("1 - Amizmiz")
print("2 - Asni")
print("3 - Ighil")
print("4 - Tafeghaghte")
print("5 - TalatNYaaqoub")
print("6 - SerghinePleiades")
print("7 - AmizmizPleiades")
print("8 - AitYahya")

choice = input("\nVotre choix : ")

sites = {
    "1": "Amizmiz",
    "2": "Asni",
    "3": "Ighil",
    "4": "Tafeghaghte",
    "5": "TalatNYaaqoub",
    "6": "SerghinePleiades",
    "7": "AmizmizPleiades",
    "8": "AitYahya",
}

if choice not in sites:
    print("Choix invalide.")
    sys.exit()

SITE = sites[choice]
set_site(SITE)

print("\n" + "=" * 60)
print("Zone selectionnee :", SITE)
print("=" * 60)

HIGH_RES_SITES = ["SerghinePleiades", "AmizmizPleiades", "AitYahya"]

if SITE in HIGH_RES_SITES:
    pipeline = [
        "02_tile_images.py",
        "03_prepare_tensor.py",
        "04_run_model.py",
        "05_merge_logits.py",
        "07_filter_buildings.py",
        "07_visualize.py",
        "08_diagnostic_classes.py",
        "09_evaluation_metrics.py",
        "report_generator.py",
    ]
else:
    pipeline = [
        "sentinel_to_rgb.py",
        "01_get_bounds.py",
        "00_crop_to_village.py",
        "prepare_buildings.py",
        "02_tile_images.py",
        "03_prepare_tensor.py",
        "04_run_model.py",
        "05_merge_logits.py",
        "07_filter_buildings.py",
        "07_visualize.py",
        "08_diagnostic_classes.py",
        "09_evaluation_metrics.py",
        "report_generator.py",
    ]

for script in pipeline:
    script_path = SCRIPTS / script
    if not script_path.exists():
        print(f"\nATTENTION : {script} introuvable dans {SCRIPTS} — ignore.")
        continue
    print("\n" + "=" * 60)
    print("Execution :", script)
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        print(f"\nErreur dans {script}")
        sys.exit(result.returncode)

print("\n" + "=" * 60)
print("PIPELINE TERMINEE AVEC SUCCES")
print("=" * 60)
