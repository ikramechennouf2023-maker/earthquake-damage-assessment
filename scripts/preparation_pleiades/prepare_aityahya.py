"""
==========================================================================
01_prepare_aityahya.py
--------------------------------------------------------------------------
Prepare l'image aityahya.tif (deja petite, 817x663 px, pas besoin de
fenetre de lecture partielle) pour le pipeline du modele.

Genere : post_rgb.png, pre_rgb.png (zero), bounds.json,
building_footprints.gpkg — dans outputs/AitYahya/.

Lance depuis n'importe ou :
    python3 01_prepare_aityahya.py
==========================================================================
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
import geopandas as gpd

TIF_FILE = "/Users/ikramechennouf/Documents/aityahya.tif"

BUILDINGS_FILE = "/Users/ikramechennouf/Documents/model_dinov3/input/building_footprints.gpkg"

OUTPUT_DIR = Path("/Users/ikramechennouf/Documents/model_dinov3/outputs/AitYahya")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2 + 1e-8)
    return (band * 255).astype(np.uint8)


def main():

    print("=" * 70)
    print("PREPARATION DE L'IMAGE — Ait Yahya")
    print("=" * 70)

    with rasterio.open(TIF_FILE) as src:

        print(f"\nTaille : {src.width} x {src.height} pixels")
        print(f"CRS : {src.crs}")
        print(f"Bandes : {src.count}")

        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

        bounds = src.bounds
        crs = str(src.crs)
        width, height = src.width, src.height

    print("\nNormalisation et conversion en RGB...")

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])

    post_png_path = OUTPUT_DIR / "post_rgb.png"
    Image.fromarray(rgb).save(post_png_path)
    print(f"post_rgb.png sauvegarde : {post_png_path}")

    print("\nGeneration de pre_rgb.png (zero, pas d'image PRE disponible)...")
    zero_rgb = np.zeros_like(rgb)
    Image.fromarray(zero_rgb).save(OUTPUT_DIR / "pre_rgb.png")

    bounds_data = {
        "width": width,
        "height": height,
        "crs": crs,
        "left": bounds.left,
        "bottom": bounds.bottom,
        "right": bounds.right,
        "top": bounds.top,
    }

    with open(OUTPUT_DIR / "bounds.json", "w") as f:
        json.dump(bounds_data, f, indent=4)

    print(f"\nbounds.json sauvegarde : {bounds_data}")

    print("\n" + "=" * 70)
    print("VERIFICATION DE LA COUVERTURE DES BATIMENTS")
    print("=" * 70)

    gdf = gpd.read_file(
        BUILDINGS_FILE,
        bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top)
    )

    print(f"\nBatiments trouves dans cette zone : {len(gdf)}")

    if len(gdf) == 0:
        print("\nATTENTION : aucun batiment trouve pour cette zone precise.")
        print("La zone est peut-etre trop petite ou hors couverture de la")
        print("base de donnees mondiale de batiments.")
    else:
        gdf.to_file(OUTPUT_DIR / "building_footprints.gpkg", driver="GPKG")
        print(f"Batiments sauvegardes : {OUTPUT_DIR / 'building_footprints.gpkg'}")

    n_tiles = (width // 512 + 1) * (height // 512 + 1)
    print(f"\nNombre de tuiles 512x512 estime : ~{n_tiles}")
    print(f"Temps d'inference estime : ~{n_tiles * 10 / 60:.1f} minutes")

    print("\n" + "=" * 70)
    print("TERMINE")
    print("=" * 70)
    print("\nProchaine etape : ajoute 'AitYahya' au menu de main.py,")
    print("puis lance python3 main.py -> choix correspondant")


if __name__ == "__main__":
    main()