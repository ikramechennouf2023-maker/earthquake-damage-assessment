"""
==========================================================================
01_crop_pleiades_to_zone.py
--------------------------------------------------------------------------
Script UNIQUE et reutilisable pour recadrer n'importe quelle image
Pleiades enregistree ci-dessous, sans avoir a recreer un fichier a
chaque nouvelle image.

Pour ajouter une nouvelle image Pleiades plus tard : ajoute simplement
une ligne dans le dictionnaire PLEIADES_IMAGES en bas de ce fichier.

Genere : post_rgb.png, pre_rgb.png (zero), bounds.json,
building_footprints.gpkg — dans outputs/<site>/.

Lance depuis n'importe ou :
    python3 01_crop_pleiades_to_zone.py
==========================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import geopandas as gpd

PLEIADES_IMAGES = {
    "1": {
        "site": "SerghinePleiades",
        "file": (
            "/Users/ikramechennouf/Documents/"
            "001 - IMG_PXS_MN351a-202309101103360-L3_-G001-177068132.JP2"
        ),
        "note": "Kadiri / Serghine, province d'Azilal (~200 km epicentre)",
    },
    "2": {
        "site": "AmizmizPleiades",
        "file": (
            "/Users/ikramechennouf/Documents/"
            "003 - IMG_PXS_MN351a-202309101103590-L3_-G001-177067686.JP2"
        ),
        "note": "Couvre Amizmiz — comparable directement avec les resultats Sentinel-2",
    },
}

BUILDINGS_FILE = "/Users/ikramechennouf/Documents/model_dinov3/input/building_footprints.gpkg"

PROJECT_ROOT = Path("/Users/ikramechennouf/Documents/model_dinov3")

CROP_SIZE_PX = 6000


def normalize(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2 + 1e-8)
    return (band * 255).astype(np.uint8)


def choose_image():

    print("=" * 70)
    print("IMAGES PLEIADES DISPONIBLES")
    print("=" * 70)

    for key, info in PLEIADES_IMAGES.items():
        print(f"\n{key} - {info['site']}")
        print(f"    {info['note']}")

    choice = input("\nTon choix : ").strip()

    if choice not in PLEIADES_IMAGES:
        print("Choix invalide.")
        sys.exit(1)

    return PLEIADES_IMAGES[choice]


def main():

    selected = choose_image()

    site = selected["site"]
    pleiades_file = selected["file"]

    output_dir = PROJECT_ROOT / "outputs" / site
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"RECADRAGE — {site}")
    print("=" * 70)

    with rasterio.open(pleiades_file) as src:

        print(f"\nImage complete : {src.width} x {src.height} pixels")
        print(f"CRS : {src.crs}")

        center_col = src.width // 2
        center_row = src.height // 2
        half = CROP_SIZE_PX // 2

        col_off = max(0, center_col - half)
        row_off = max(0, center_row - half)
        width = min(CROP_SIZE_PX, src.width - col_off)
        height = min(CROP_SIZE_PX, src.height - row_off)

        window = Window(col_off, row_off, width, height)

        print(f"\nLecture de la fenetre centrale : "
              f"{width}x{height} px (colonne {col_off}, ligne {row_off})")

        red = src.read(1, window=window)
        green = src.read(2, window=window)
        blue = src.read(3, window=window)

        print("Lecture terminee.")

        left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
        crs = str(src.crs)

    print("\nNormalisation et conversion en RGB...")

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])

    post_png_path = output_dir / "post_rgb.png"
    Image.fromarray(rgb).save(post_png_path)
    print(f"post_rgb.png sauvegarde : {post_png_path}")
    print(f"Taille finale : {rgb.shape[1]} x {rgb.shape[0]} pixels")

    print("\nGeneration de pre_rgb.png (zero, pas d'image PRE disponible)...")
    zero_rgb = np.zeros_like(rgb)
    pre_png_path = output_dir / "pre_rgb.png"
    Image.fromarray(zero_rgb).save(pre_png_path)
    print(f"pre_rgb.png sauvegarde : {pre_png_path}")

    bounds_data = {
        "width": width,
        "height": height,
        "crs": crs,
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
    }

    bounds_path = output_dir / "bounds.json"
    with open(bounds_path, "w") as f:
        json.dump(bounds_data, f, indent=4)

    print(f"\nbounds.json sauvegarde : {bounds_path}")
    print(f"  {bounds_data}")

    print("\n" + "=" * 70)
    print("VERIFICATION DE LA COUVERTURE DES BATIMENTS")
    print("=" * 70)

    gdf = gpd.read_file(BUILDINGS_FILE, bbox=(left, bottom, right, top))

    print(f"\nBatiments trouves dans cette zone : {len(gdf)}")

    if len(gdf) == 0:
        print("\nATTENTION : aucun batiment trouve pour cette zone precise.")
    else:
        output_buildings = output_dir / "building_footprints.gpkg"
        gdf.to_file(output_buildings, driver="GPKG")
        print(f"Batiments sauvegardes : {output_buildings}")

    print("\n" + "=" * 70)
    print(f"TERMINE — zone : {site}")
    print("=" * 70)
    print(f"\nPour continuer : ajoute '{site}' a la liste des zones dans")
    print("main.py, puis lance python3 main.py en choisissant cette zone.")


if __name__ == "__main__":
    main()