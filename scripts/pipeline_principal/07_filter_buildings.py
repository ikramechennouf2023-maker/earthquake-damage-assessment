
# ==========================================================
# 07_filter_buildings.py
# VERSION CORRIGEE — reprojection des batiments vers le CRS de
# l'image AVANT le calcul des fenetres locales.
#
# BUG CORRIGE : le fichier building_footprints.gpkg est en
# EPSG:4326 (degres), alors que bounds.json / la carte de
# dommages sont en EPSG:32629 (metres, UTM). Sans reprojection,
# geometry.bounds renvoie des coordonnees en degres compareés a
# une transform en metres -> aucune fenetre ne correspond ->
# 100% des batiments classes "Unknown" (-1).
# ==========================================================
 
import sys
import json
from pathlib import Path
 
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
 
# ==========================================================
# AJOUT DU DOSSIER RACINE
# ==========================================================
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
from config import output_dir
 
# ==========================================================
# DOSSIERS
# ==========================================================
 
OUTPUT = Path(output_dir())
 
BUILDINGS_FILE = OUTPUT / "building_footprints.gpkg"
BOUNDS_FILE = OUTPUT / "bounds.json"
DAMAGE_MAP_FILE = OUTPUT / "damage_map.npy"
 
OUTPUT_FILE = OUTPUT / "damage_buildings.gpkg"
 
 
# ==========================================================
# CHARGEMENT DES DONNEES
# ==========================================================
 
def load_buildings():
    print("Chargement des batiments...")
    gdf = gpd.read_file(BUILDINGS_FILE)
    print(f"{len(gdf)} batiments charges.")
    print(f"CRS des batiments : {gdf.crs}")
    return gdf
 
 
def load_damage_map():
    print("Chargement de la carte des degats...")
    damage_map = np.load(DAMAGE_MAP_FILE)
    print(damage_map.shape)
    return damage_map
 
 
def load_transform():
    with open(BOUNDS_FILE, "r") as f:
        bounds = json.load(f)
 
    width = bounds["width"]
    height = bounds["height"]
    crs = bounds["crs"]
 
    transform = from_bounds(
        bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
        width, height,
    )
 
    print(f"CRS de l'image (bounds.json) : {crs}")
 
    return transform, width, height, crs
 
 
# ==========================================================
# FENETRE LOCALE (le fix de performance)
# ==========================================================
 
def get_local_window(geometry, transform, width, height):
    """Calcule la petite fenetre (en pixels) qui contient le batiment,
    au lieu de considerer toute l'image."""
 
    minx, miny, maxx, maxy = geometry.bounds
 
    window = window_from_bounds(minx, miny, maxx, maxy, transform=transform)
 
    row0 = max(0, int(np.floor(window.row_off)))
    col0 = max(0, int(np.floor(window.col_off)))
    row1 = min(height, int(np.ceil(window.row_off + window.height)))
    col1 = min(width, int(np.ceil(window.col_off + window.width)))
 
    if row1 <= row0 or col1 <= col0:
        return None
 
    return row0, row1, col0, col1
 
 
def build_local_transform(transform, row0, col0):
    x = transform.c + col0 * transform.a
    y = transform.f + row0 * transform.e
    return rasterio.Affine(transform.a, transform.b, x, transform.d, transform.e, y)
 
 
def classify_one_building(geometry, damage_map, transform, width, height):
 
    if geometry is None or geometry.is_empty:
        return -1
 
    window = get_local_window(geometry, transform, width, height)
    if window is None:
        return -1
 
    row0, row1, col0, col1 = window
    local_transform = build_local_transform(transform, row0, col0)
    out_shape = (row1 - row0, col1 - col0)
 
    # Rasterisation UNIQUEMENT sur la petite fenetre locale, pas toute l'image
    mask = rasterize(
        [(geometry, 1)], out_shape=out_shape, transform=local_transform,
        fill=0, all_touched=True, dtype=np.uint8,
    )
 
    local_damage = damage_map[row0:row1, col0:col1]
    pixels = local_damage[mask == 1]
 
    if pixels.size == 0:
        return -1
 
    values, counts = np.unique(pixels, return_counts=True)
    return int(values[np.argmax(counts)])
 
 
# ==========================================================
# ATTRIBUTION DES CLASSES DE DOMMAGE
# ==========================================================
 
def classify_buildings():
 
    gdf = load_buildings()
    damage_map = load_damage_map()
    transform, width, height, image_crs = load_transform()
 
    # ------------------------------------------------------
    # LE FIX : reprojection des batiments vers le CRS de l'image
    # AVANT tout calcul de fenetre. Sans ca, geometry.bounds
    # renvoie des degres compareés a une transform en metres.
    # ------------------------------------------------------
    if str(gdf.crs) != str(image_crs):
        print(f"Reprojection des batiments : {gdf.crs} -> {image_crs}")
        gdf = gdf.to_crs(image_crs)
    else:
        print("CRS deja coherent, pas de reprojection necessaire.")
 
    damage_classes = []
    total = len(gdf)
 
    print()
    print("Classification des batiments (fenetre locale)...")
    print()
 
    for i, geom in enumerate(gdf.geometry, start=1):
 
        cls = classify_one_building(geom, damage_map, transform, width, height)
        damage_classes.append(cls)
 
        if i % 500 == 0 or i == total:
            print(f"{i}/{total} batiments traites")
 
    gdf["damage_class"] = damage_classes
 
    unclassified = sum(1 for c in damage_classes if c == -1)
    print()
    print(f"Batiments non classes (Unknown) : {unclassified}/{total}")
 
    return gdf
 
 
# ==========================================================
# SAUVEGARDE
# ==========================================================
 
def save_results(gdf):
    gdf.to_file(OUTPUT_FILE, driver="GPKG")
    print()
    print("Resultat sauvegarde :")
    print(OUTPUT_FILE)
 
 
# ==========================================================
# MAIN
# ==========================================================
 
def main():
 
    print("=" * 60)
    print("07_filter_buildings.py — version corrigee (reprojection CRS)")
    print("=" * 60)
 
    gdf = classify_buildings()
    save_results(gdf)
 
    print()
    print("=" * 60)
    print("07_filter_buildings.py termine avec succes")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 
