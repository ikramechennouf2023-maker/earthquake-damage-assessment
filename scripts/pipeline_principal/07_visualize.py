# ==========================================================
# 08_visualize.py
# VERSION RAPIDE — rasterisation vectorisee (au lieu de
# dessiner chaque batiment individuellement avec geopandas.plot,
# qui devient tres lent au-dela de quelques milliers de polygones)
# ==========================================================

import sys
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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

IMAGE_FILE = OUTPUT / "post_rgb.png"
BUILDINGS_FILE = OUTPUT / "damage_buildings.gpkg"
BOUNDS_FILE = OUTPUT / "bounds.json"

OUTPUT_IMAGE = OUTPUT / "visualization.png"

OVERLAY_ALPHA = 0.6

# ==========================================================
# COULEURS DES CLASSES (RGB, pour la rasterisation numpy)
# ==========================================================

CLASS_RGB = {
    -1: (149, 165, 166),   # Unknown - gris
    0:  (46, 204, 113),    # No Damage - vert
    1:  (241, 196, 15),    # Minor Damage - jaune
    2:  (230, 126, 34),    # Major Damage - orange
    3:  (231, 76, 60),     # Destroyed - rouge
}

LABELS = {
    0: "No Damage", 1: "Minor Damage", 2: "Major Damage",
    3: "Destroyed", -1: "Unknown",
}


# ==========================================================
# CHARGEMENT
# ==========================================================

def load_data():

    print("Chargement de l'image...")
    image = np.array(Image.open(IMAGE_FILE).convert("RGB"))
    print("  Taille :", image.shape)

    print("Chargement des bounds...")
    with open(BOUNDS_FILE, "r") as f:
        bounds = json.load(f)

    transform = rasterio.transform.from_bounds(
        bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
        bounds["width"], bounds["height"],
    )

    print("Chargement des batiments...")
    gdf = gpd.read_file(BUILDINGS_FILE)
    print(f"  {len(gdf)} batiments charges.")

    return image, gdf, transform


# ==========================================================
# RASTERISATION VECTORISEE (rapide, un seul appel)
# ==========================================================

def rasterize_damage(gdf, transform, out_shape):

    # decalage +1 : -1 (Unknown) -> 0, 0..3 -> 1..4, fill=0 = "pas de batiment"
    shapes = [
        (geom, int(cls) + 1)
        for geom, cls in zip(gdf.geometry, gdf["damage_class"])
        if geom is not None and not geom.is_empty
    ]

    print(f"Rasterisation de {len(shapes)} batiments...")

    raster = rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=0, all_touched=True, dtype=np.uint8,
    )

    return raster


def raster_to_rgb(raster):

    rgb = np.zeros((*raster.shape, 3), dtype=np.uint8)

    mapping = {
        0: CLASS_RGB[-1], 1: CLASS_RGB[0], 2: CLASS_RGB[1],
        3: CLASS_RGB[2], 4: CLASS_RGB[3],
    }
    for value, color in mapping.items():
        rgb[raster == value] = color

    return rgb


# ==========================================================
# OVERLAY (blend numpy vectorise, pas de boucle)
# ==========================================================

def build_overlay(background, damage_raster):

    damage_rgb = raster_to_rgb(damage_raster)

    # "pas de batiment" (valeur brute 0, avant decalage) reste transparent
    mask = damage_raster > 0

    overlay = background.copy().astype(np.float32)
    overlay[mask] = (
        background[mask].astype(np.float32) * (1 - OVERLAY_ALPHA)
        + damage_rgb[mask].astype(np.float32) * OVERLAY_ALPHA
    )

    return overlay.astype(np.uint8)


# ==========================================================
# AFFICHAGE FINAL (un seul imshow, rapide quel que soit N)
# ==========================================================

def create_visualization(overlay):

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(overlay)

    legend = [
        mpatches.Patch(color=np.array(CLASS_RGB[key]) / 255, label=LABELS[key])
        for key in [0, 1, 2, 3, -1]
    ]

    ax.legend(handles=legend, loc="upper right")
    ax.set_title("Building Damage Assessment")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    plt.close()

    print()
    print("Visualisation enregistree :")
    print(OUTPUT_IMAGE)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("08_visualize.py — version rapide (rasterisation vectorisee)")
    print("=" * 60)

    image, gdf, transform = load_data()

    damage_raster = rasterize_damage(gdf, transform, image.shape[:2])

    overlay = build_overlay(image, damage_raster)

    create_visualization(overlay)

    print()
    print("=" * 60)
    print("08_visualize.py termine avec succes")
    print("=" * 60)


if __name__ == "__main__":
    main()