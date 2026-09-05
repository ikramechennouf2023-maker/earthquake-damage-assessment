"""
==========================================================================
01_crop_pleiades_amizmiz_small.py
--------------------------------------------------------------------------
Version REDUITE de 01_crop_pleiades_amizmiz_centered.py : zone de
4x4 km (au lieu de 8x8 km), pour un temps de calcul raisonnable
(~40-60 minutes au lieu de plusieurs heures), tout en couvrant
ENTIEREMENT la zone choisie (pas d'echantillonnage, pas de trous).

Ce fichier est independant de 01_crop_pleiades_amizmiz_centered.py,
qui reste inchange.

Lance depuis n'importe ou :
    python3 01_crop_pleiades_amizmiz_small.py
==========================================================================
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from PIL import Image
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PLEIADES_FILE = (
    "/Users/ikramechennouf/Documents/"
    "003 - IMG_PXS_MN351a-202309101103590-L3_-G001-177067686.JP2"
)

BUILDINGS_FILE = "/Users/ikramechennouf/Documents/model_dinov3/input/building_footprints.gpkg"

OUTPUT_DIR = Path("/Users/ikramechennouf/Documents/model_dinov3/outputs/AmizmizPleiades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMIZMIZ_LAT = 31.2147
AMIZMIZ_LON = -8.2469

# Zone reduite : 2 km de rayon = 4x4 km au total, couverte ENTIEREMENT
HALF_SIZE_KM = 2.0


def normalize(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2 + 1e-8)
    return (band * 255).astype(np.uint8)


def main():

    print("=" * 70)
    print("RECADRAGE REDUIT — AMIZMIZ (4x4 km, couverture complete)")
    print("=" * 70)

    with rasterio.open(PLEIADES_FILE) as src:

        print(f"\nImage complete : {src.width} x {src.height} pixels")
        print(f"CRS : {src.crs}")

        half_deg_lat = HALF_SIZE_KM / 111.0
        half_deg_lon = HALF_SIZE_KM / (111.0 * 0.855)

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
                "La zone ciblee tombe hors de l'emprise de l'image."
            )

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
    create_overview(
    src,
    col_off,
    row_off,
    width,
    height,
    OUTPUT_DIR / "overview_crop.png"
)

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
# ==========================================================
# APERÇU DE L'IMAGE COMPLETE
# ==========================================================

def create_overview(src,
                    col_off,
                    row_off,
                    width,
                    height,
                    output_file):

    print("\nCréation de l'aperçu de la zone...")

    factor = 32

    preview = src.read(
        [1, 2, 3],
        out_shape=(
            3,
            src.height // factor,
            src.width // factor
        )
    )

    preview = np.moveaxis(preview, 0, -1)

    for i in range(3):
        preview[:, :, i] = normalize(preview[:, :, i])

    scale_x = preview.shape[1] / src.width
    scale_y = preview.shape[0] / src.height

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(preview)

    rect = Rectangle(
        (
            col_off * scale_x,
            row_off * scale_y
        ),
        width * scale_x,
        height * scale_y,
        linewidth=3,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)

    # centre
    cx = (col_off + width/2) * scale_x
    cy = (row_off + height/2) * scale_y

    ax.plot(
        cx,
        cy,
        "ro",
        markersize=6
    )

    ax.text(
        cx + 10,
        cy,
        "Centre d'Amizmiz",
        color="red",
        fontsize=10,
        weight="bold"
    )

    ax.set_title(
        "Localisation de la zone extraite dans l'image Pléiades",
        fontsize=14,
        weight="bold"
    )

    ax.axis("off")

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Aperçu sauvegardé :", output_file)

if __name__ == "__main__":
    main()