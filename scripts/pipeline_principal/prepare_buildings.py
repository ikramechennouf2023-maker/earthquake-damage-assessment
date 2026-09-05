# ==========================================================
# 02_prepare_buildings.py
# Extrait uniquement les bâtiments présents dans l'emprise
# Sentinel et crée outputs/<site>/building_footprints.gpkg
# ==========================================================

import json
import sys
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

# ==========================================================
# AJOUT DU DOSSIER PROJET AU PYTHONPATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

# ==========================================================
# CONFIGURATION
# ==========================================================

SITE = get_site()

INPUT_GPKG = PROJECT_ROOT / "input" / "building_footprints.gpkg"

OUTPUT_FOLDER = Path(output_dir())
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BOUNDS_FILE = OUTPUT_FOLDER / "bounds.json"

OUTPUT_GPKG = OUTPUT_FOLDER / "building_footprints.gpkg"

# ==========================================================
# VERIFICATIONS
# ==========================================================

if not INPUT_GPKG.exists():
    raise FileNotFoundError(INPUT_GPKG)

if not BOUNDS_FILE.exists():
    raise FileNotFoundError(BOUNDS_FILE)

# ==========================================================
# LECTURE DES BOUNDS
# ==========================================================

with open(BOUNDS_FILE, "r") as f:
    bounds = json.load(f)

xmin = bounds["left"]
ymin = bounds["bottom"]
xmax = bounds["right"]
ymax = bounds["top"]

print("=" * 60)
print("SITE :", SITE)
print("=" * 60)

print("Lecture uniquement des bâtiments de la zone...")

# ==========================================================
# LECTURE SPATIALE (PAS LES 3.5 Go)
# ==========================================================

# Lire uniquement le CRS du GPKG
tmp = gpd.read_file(INPUT_GPKG, rows=1)
gpkg_crs = tmp.crs

# Transformer le bbox du raster (EPSG:32629) vers le CRS du GPKG
transformer = Transformer.from_crs(
    "EPSG:32629",
    gpkg_crs,
    always_xy=True
)

xmin2, ymin2 = transformer.transform(xmin, ymin)
xmax2, ymax2 = transformer.transform(xmax, ymax)

bbox_geom = box(xmin2, ymin2, xmax2, ymax2)

gdf = gpd.read_file(
    INPUT_GPKG,
    layer="with_geom",
    bbox=bbox_geom,
    engine="pyogrio"
)

print(f"Bâtiments extraits : {len(gdf)}")

if len(gdf) == 0:
    raise RuntimeError("Aucun bâtiment trouvé dans cette zone.")

# ==========================================================
# SAUVEGARDE
# ==========================================================

gdf.to_file(
    OUTPUT_GPKG,
    driver="GPKG"
)

print()
print("Fichier créé :")
print(OUTPUT_GPKG)

print("Terminé.")

print("CRS GPKG :", gpkg_crs)
print("BBox reprojeté :", bbox_geom.bounds)
print("Bâtiments extraits :", len(gdf))