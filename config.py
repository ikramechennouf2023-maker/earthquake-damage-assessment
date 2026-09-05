"""
==========================================================
config.py
Configuration globale du projet
==========================================================
"""

from pathlib import Path
import json

# ==========================================================
# RACINE DU PROJET
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# DOSSIERS
# ==========================================================

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIR / "model.onnx"

# ==========================================================
# CONFIGURATION
# ==========================================================

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / "current_site.json"

# ==========================================================
# ZONE
# ==========================================================

def set_site(site: str):
    """Enregistre la zone choisie."""

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"site": site}, f, indent=4)


def get_site():
    """Retourne la zone sélectionnée."""

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILE} n'existe pas. Lancez d'abord main.py."
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    site = data.get("site")

    if not site:
        raise ValueError(
            "Aucune zone sélectionnée."
        )

    return site

# ==========================================================
# DOSSIERS DE LA ZONE
# ==========================================================

def site_dir():
    return INPUT_DIR / get_site()


def pre_dir():
    return site_dir() / "pre"


def post_dir():
    return site_dir() / "post"


def building_file():
    return site_dir() / "building_footprints.gpkg"


# ==========================================================
# DOSSIERS DE SORTIE
# ==========================================================

def output_dir():

    folder = OUTPUTS_DIR / get_site()
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def report_dir():

    folder = REPORTS_DIR / get_site()
    folder.mkdir(parents=True, exist_ok=True)

    return folder

# ==========================================================
# FICHIERS DE SORTIE
# ==========================================================

def pre_rgb():
    return output_dir() / "pre_rgb.png"


def post_rgb():
    return output_dir() / "post_rgb.png"


def prediction_png():
    return output_dir() / "prediction.png"


def damage_png():
    return output_dir() / "damage_map.png"


def overlay_png():
    return output_dir() / "overlay.png"


def damage_geojson():
    return output_dir() / "damage.geojson"


def damage_gpkg():
    return output_dir() / "damage.gpkg"


def metrics_txt():
    return output_dir() / "metrics.txt"


def report_pdf():
    return report_dir() / "report.pdf"