"""
==========================================================
sentinel_to_rgb.py
PARTIE 1
Imports + Configuration + Lecture des bandes Sentinel-2
==========================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from PIL import Image

from config import (
    pre_dir,
    post_dir,
    output_dir
)

RGB_BANDS = {
    "R": "B04",
    "G": "B03",
    "B": "B02",
}

def find_band(folder, band):
    """Cherche specifiquement la bande a 10m dans IMG_DATA/R10m/,
    avec le motif exact '*_B0X_10m.jp2'. Ignore volontairement les
    fichiers QI_DATA (masques de qualite) qui contiennent aussi le
    nom de la bande dans leur nom mais ne sont PAS les vraies
    donnees satellite (ex: MSK_QUALIT_B04.jp2, MSK_DETFOO_B04.jp2)."""

    folder = Path(folder)

    matches = list(folder.rglob(f"IMG_DATA/R10m/*_{band}_10m.jp2"))

    if not matches:
        matches = [
            m for m in folder.rglob(f"*_{band}_10m.jp2")
            if "QI_DATA" not in m.parts
        ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Bande {band} (10m) introuvable dans {folder} "
            "(recherche limitee a IMG_DATA, QI_DATA exclu)"
        )

    if len(matches) > 1:
        print(f"  ATTENTION : {len(matches)} fichiers trouves pour {band}, "
              f"utilisation du premier : {matches[0]}")

    return matches[0]

def read_band(path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
    return band

def normalize(img):
    img = np.clip(img, 0, 3000)
    img = img / 3000.0
    img = (img * 255).astype(np.uint8)
    return img

def create_rgb(folder):

    red_path = find_band(folder, RGB_BANDS["R"])
    green_path = find_band(folder, RGB_BANDS["G"])
    blue_path = find_band(folder, RGB_BANDS["B"])

    print(f"  R (B04) : {red_path}")
    print(f"  G (B03) : {green_path}")
    print(f"  B (B02) : {blue_path}")

    red = read_band(red_path)
    green = read_band(green_path)
    blue = read_band(blue_path)

    print(f"  Verification - R min/max : {red.min():.0f}/{red.max():.0f}")
    if red.max() == 0:
        raise ValueError(
            f"La bande R lue est entierement a 0 ({red_path}). "
            "Verifier le fichier source."
        )

    rgb = np.dstack([
        normalize(red),
        normalize(green),
        normalize(blue)
    ])

    return Image.fromarray(rgb)

def save_rgb(image, filename):
    save_path = output_dir() / filename
    image.save(save_path)
    print(f"Saved : {save_path}")

def main():

    print("=" * 60)
    print("CONVERSION SENTINEL-2 -> RGB")
    print("=" * 60)

    print("\nCreating PRE RGB...")
    pre = create_rgb(pre_dir())
    save_rgb(pre, "pre_rgb.png")

    print("\nCreating POST RGB...")
    post = create_rgb(post_dir())
    save_rgb(post, "post_rgb.png")

    print("\nConversion terminée.")

if __name__ == "__main__":
    main()
