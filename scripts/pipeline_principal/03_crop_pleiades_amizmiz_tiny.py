"""
03_crop_pleiades_amizmiz_tiny.py — zone TRES reduite (1.5x1.5 km)
pour un calcul rapide et complet (~10-15 minutes), sans interruption.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from PIL import Image
import geopandas as gpd

PLEIADES_FILE = (
    "/Users/ikramechennouf/Documents/"
    "003 - IMG_PXS_MN351a-202309101103590-L3_-G001-177067686.JP2"
)

BUILDINGS_FILE = "/Users/ikramechennouf/Documents/model_dinov3/input/building_footprints.gpkg"
OUTPUT_DIR = Path("/Users/ikramechennouf/Documents/model_dinov3/outputs/AmizmizPleiades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMIZMIZ_LAT = 31.2147
AMIZMIZ_LON = -8.2469
HALF_SIZE_KM = 0.75  # zone de 1.5x1.5 km -> ~64 tuiles, ~10-15 min


def normalize(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2 + 1e-8)
    return (band * 255).astype(np.uint8)


def main():
    print("=" * 70)
    print("RECADRAGE TRES REDUIT — AMIZMIZ (1.5x1.5 km)")
    print("=" * 70)

    with rasterio.open(PLEIADES_FILE) as src:
        half_deg_lat = HALF_SIZE_KM / 111.0
        half_deg_lon = HALF_SIZE_KM / (111.0 * 0.855)
        left = AMIZMIZ_LON - half_deg_lon
        right = AMIZMIZ_LON + half_deg_lon
        bottom = AMIZMIZ_LAT - half_deg_lat
        top = AMIZMIZ_LAT + half_deg_lat

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        col_off = max(0, int(window.col_off))
        row_off = max(0, int(window.row_off))
        width = min(int(window.width), src.width - col_off)
        height = min(int(window.height), src.height - row_off)

        real_window = Window(col_off, row_off, width, height)
        print(f"Fenetre : {width}x{height} px")

        red = src.read(1, window=real_window)
        green = src.read(2, window=real_window)
        blue = src.read(3, window=real_window)

        real_left, real_bottom, real_right, real_top = rasterio.windows.bounds(
            real_window, src.transform
        )
        crs = str(src.crs)

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
    Image.fromarray(rgb).save(OUTPUT_DIR / "post_rgb.png")
    Image.fromarray(np.zeros_like(rgb)).save(OUTPUT_DIR / "pre_rgb.png")

    bounds_data = {
        "width": width, "height": height, "crs": crs,
        "left": real_left, "bottom": real_bottom,
        "right": real_right, "top": real_top,
    }
    with open(OUTPUT_DIR / "bounds.json", "w") as f:
        json.dump(bounds_data, f, indent=4)

    gdf = gpd.read_file(BUILDINGS_FILE, bbox=(real_left, real_bottom, real_right, real_top))
    print(f"Batiments trouves : {len(gdf)}")
    gdf.to_file(OUTPUT_DIR / "building_footprints.gpkg", driver="GPKG")

    n_tiles = (width // 512 + 1) * (height // 512 + 1)
    print(f"Tuiles estimees : ~{n_tiles} (~{n_tiles*10/60:.0f} min)")
    print("TERMINE")


if __name__ == "__main__":
    main()
