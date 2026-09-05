"""
==========================================================================
01_crop_pleiades_amizmiz_centered.py
--------------------------------------------------------------------------
Recadre l'image Pleiades 003 sur les VRAIES coordonnees du centre
d'Amizmiz (pas le centre geometrique de l'image), avec une zone
elargie pour capturer l'ensemble du village et ses environs directs.

Contrairement au recadrage precedent (centre arbitraire de l'image,
6x6 km), celui-ci cible directement la zone habitee reelle, augmentant
les chances de capturer d'eventuels batiments endommages sans avoir
besoin de traiter l'image entiere (36572x43800 px, ~18-20h de calcul).

Lance depuis n'importe ou :
    python3 01_crop_pleiades_amizmiz_centered.py
==========================================================================
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from PIL import Image
import geopandas as gpd

PLEIADES_FILE = (
    "/Users/ikramechennouf/Documents/"
    "003 - IMG_PXS_MN351a-202309101103590-L3_-G001-177067686.JP2"
)

BUILDINGS_FILE = "/Users/ikramechennouf/Documents/model_dinov3/input/building_footprints.gpkg"

OUTPUT_DIR = Path("/Users/ikramechennouf/Documents/model_dinov3/outputs/AmizmizPleiades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vraies coordonnees du centre d'Amizmiz (memes que sur Sentinel-2)
AMIZMIZ_LAT = 31.2147
AMIZMIZ_LON = -8.2469

# Zone elargie : 8 km de cote (au lieu de 6), pour capturer largement
# le village ET ses environs directs, sans traiter toute l'image
HALF_SIZE_KM = 4.0


def normalize(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2 + 1e-8)
    return (band * 255).astype(np.uint8)


def main():

    print("=" * 70)
    print("RECADRAGE CENTRE SUR AMIZMIZ (vraies coordonnees)")
    print("=" * 70)

    with rasterio.open(PLEIADES_FILE) as src:

        print(f"\nImage complete : {src.width} x {src.height} pixels")
        print(f"CRS : {src.crs}")

        # Conversion km -> degres (approximation a cette latitude)
        half_deg_lat = HALF_SIZE_KM / 111.0
        half_deg_lon = HALF_SIZE_KM / (111.0 * 0.855)  # cos(31.2 deg)

        left = AMIZMIZ_LON - half_deg_lon
        right = AMIZMIZ_LON + half_deg_lon
        bottom = AMIZMIZ_LAT - half_deg_lat
        top = AMIZMIZ_LAT + half_deg_lat

        print(f"\nZone ciblee (vraies coord. Amizmiz) :")
        print(f"  lon : {left:.5f} -> {right:.5f}")
        print(f"  lat : {bottom:.5f} -> {top:.5f}")
        print(f"  taille : ~{HALF_SIZE_KM*2:.1f} x {HALF_SIZE_KM*2:.1f} km")

        window = from_bounds(left, bottom, right, top, transform=src.transform)

        col_off = max(0, int(window.col_off))
        row_off = max(0, int(window.row_off))
        width = min(int(window.width), src.width - col_off)
        height = min(int(window.height), src.height - row_off)

        if width <= 0 or height <= 0:
            raise ValueError(
                "La zone ciblee tombe hors de l'emprise de l'image. "
                "Verifie que cette image (003) couvre bien Amizmiz."
            )

        from rasterio.windows import Window
        real_window = Window(col_off, row_off, width, height)

        print(f"\nFenetre reelle lue : {width}x{height} px")
        print("Cette lecture peut prendre un moment selon la taille...")

        red = src.read(1, window=real_window)
        green = src.read(2, window=real_window)
        blue = src.read(3, window=real_window)

        print("Lecture terminee.")

        real_left, real_bottom, real_right, real_top = rasterio.windows.bounds(
            real_window, src.transform
        )
        crs = str(src.crs)

    print("\nNormalisation et conversion en RGB...")

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])

    post_png_path = OUTPUT_DIR / "post_rgb.png"
    Image.fromarray(rgb).save(post_png_path)
    print(f"post_rgb.png sauvegarde : {post_png_path}")
    print(f"Taille finale : {rgb.shape[1]} x {rgb.shape[0]} pixels")

    print("\nGeneration de pre_rgb.png (zero)...")
    zero_rgb = np.zeros_like(rgb)
    Image.fromarray(zero_rgb).save(OUTPUT_DIR / "pre_rgb.png")

    bounds_data = {
        "width": width,
        "height": height,
        "crs": crs,
        "left": real_left,
        "bottom": real_bottom,
        "right": real_right,
        "top": real_top,
    }

    with open(OUTPUT_DIR / "bounds.json", "w") as f:
        json.dump(bounds_data, f, indent=4)

    print(f"\nbounds.json sauvegarde : {bounds_data}")

    print("\n" + "=" * 70)
    print("VERIFICATION DE LA COUVERTURE DES BATIMENTS")
    print("=" * 70)

    gdf = gpd.read_file(
        BUILDINGS_FILE,
        bbox=(real_left, real_bottom, real_right, real_top)
    )

    print(f"\nBatiments trouves dans cette zone : {len(gdf)}")

    if len(gdf) == 0:
        print("\nATTENTION : aucun batiment trouve pour cette zone precise.")
    else:
        gdf.to_file(OUTPUT_DIR / "building_footprints.gpkg", driver="GPKG")
        print(f"Batiments sauvegardes : {OUTPUT_DIR / 'building_footprints.gpkg'}")

    n_tiles_estimate = (width // 512 + 1) * (height // 512 + 1)
    print(f"\nNombre de tuiles 512x512 estime : ~{n_tiles_estimate}")
    print(f"Temps d'inference estime : ~{n_tiles_estimate * 10 / 60:.0f} minutes "
          f"(a ~10s/tuile)")

    print("\n" + "=" * 70)
    print("TERMINE")
    print("=" * 70)
    print("\nProchaine etape : python3 main.py -> choix 7 (AmizmizPleiades)")


if __name__ == "__main__":
    main()