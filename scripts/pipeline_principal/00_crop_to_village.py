"""
==========================================================================
00_crop_to_village.py
--------------------------------------------------------------------------
Recadre les images RGB (deja generees par sentinel_to_rgb.py) sur une
petite zone autour du village de la zone selectionnee, au lieu de garder
toute la tuile Sentinel-2 (110km x 110km).
==========================================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

Image.MAX_IMAGE_PIXELS = None

VILLAGE_COORDS = {
    "Amizmiz": (31.2147, -8.2469),
    "Asni": (31.2500, -7.9833),
    "Ighil": (31.05, -8.40),
    "Tafeghaghte": (31.1967, -8.2239),
    "TalatNYaaqoub": (30.9920, -8.1840),
}

HALF_SIZE_KM = 3.0


def main():

    site = get_site()

    print("=" * 60)
    print("00_crop_to_village.py")
    print("=" * 60)
    print(f"Zone : {site}")

    if site not in VILLAGE_COORDS:
        raise ValueError(
            f"Coordonnees inconnues pour '{site}'. "
            "Ajoute-les dans VILLAGE_COORDS en haut de ce script."
        )

    lat, lon = VILLAGE_COORDS[site]
    print(f"Coordonnees village : lat={lat}, lon={lon}")

    out = output_dir()
    bounds_file = out / "bounds.json"

    with open(bounds_file, "r", encoding="utf-8") as f:
        bounds = json.load(f)

    crs = bounds["crs"]

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    print(f"Centre en coordonnees image ({crs}) : x={cx:.1f}, y={cy:.1f}")

    half_m = HALF_SIZE_KM * 1000

    new_left = cx - half_m
    new_right = cx + half_m
    new_bottom = cy - half_m
    new_top = cy + half_m

    pixel_size = (bounds["right"] - bounds["left"]) / bounds["width"]

    col0 = int((new_left - bounds["left"]) / pixel_size)
    col1 = int((new_right - bounds["left"]) / pixel_size)
    row0 = int((bounds["top"] - new_top) / pixel_size)
    row1 = int((bounds["top"] - new_bottom) / pixel_size)

    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(bounds["width"], col1)
    row1 = min(bounds["height"], row1)

    if row1 <= row0 or col1 <= col0:
        raise ValueError(
            f"Le village {site} semble tomber hors de l'emprise de "
            f"l'image telechargee. Verifie les coordonnees ou l'image source."
        )

    print(f"Fenetre de recadrage : lignes {row0}-{row1}, colonnes {col0}-{col1}")
    print(f"Taille finale : {col1 - col0} x {row1 - row0} pixels "
          f"(~{(col1-col0)*pixel_size/1000:.1f} x {(row1-row0)*pixel_size/1000:.1f} km)")

    for name in ["pre_rgb.png", "post_rgb.png"]:

        path = out / name

        if not path.exists():
            print(f"  ATTENTION : {name} introuvable, ignore.")
            continue

        img = np.array(Image.open(path))
        cropped = img[row0:row1, col0:col1]
        Image.fromarray(cropped).save(path)

        print(f"  {name} : {img.shape} -> {cropped.shape}")

    new_bounds = {
        "width": col1 - col0,
        "height": row1 - row0,
        "crs": crs,
        "left": bounds["left"] + col0 * pixel_size,
        "right": bounds["left"] + col1 * pixel_size,
        "top": bounds["top"] - row0 * pixel_size,
        "bottom": bounds["top"] - row1 * pixel_size,
    }

    with open(bounds_file, "w", encoding="utf-8") as f:
        json.dump(new_bounds, f, indent=4)

    print()
    print("bounds.json mis a jour avec la nouvelle emprise recadree :")
    print(f"  {new_bounds}")

    print()
    print("=" * 60)
    print("00_crop_to_village.py termine avec succes")
    print("=" * 60)


if __name__ == "__main__":
    main()
