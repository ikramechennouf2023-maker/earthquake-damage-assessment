"""
==========================================================================
zoomed_damage_map.py
--------------------------------------------------------------------------
Genere une carte de classification NETTE (batiments colores bien visibles),
style xBD, en tracant les vraies formes VECTORIELLES des batiments
(pas une rasterisation sur l'image Sentinel entiere, qui les ecrase en
points invisibles).

Principe : on trouve la zone la plus dense en batiments, on zoome dessus,
et on trace directement les polygones avec matplotlib (geopandas.plot).
Les formes restent nettes quel que soit le zoom, contrairement a un
overlay raster sur l'image satellite.

Lance depuis la racine du projet, apres avoir choisi une zone via main.py :
    python3 zoomed_damage_map.py
==========================================================================
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


def find_project_root(start):
    current = Path(start).resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "config.py").is_file():
            return candidate
    raise FileNotFoundError(f"Impossible de trouver config.py depuis {start}.")


PROJECT_ROOT = find_project_root(Path(__file__).parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

SITE = get_site()
OUTPUT = Path(output_dir())

# Couleurs, meme style que xBD (fond noir, batiments colores nets)
CLASS_STYLE = {
    -1: {"label": "Unknown",       "color": "#808080"},
    0:  {"label": "No Damage",     "color": "#00E000"},
    1:  {"label": "Minor Damage",  "color": "#FFD700"},
    2:  {"label": "Major Damage",  "color": "#FF8C00"},
    3:  {"label": "Destroyed",     "color": "#FF0000"},
}

# Taille de la fenetre de zoom, en metres (300 = zone de 300x300m)
ZOOM_SIZE_M = 300


def find_densest_area(gdf, window_size):
    """Trouve la zone window_size x window_size la plus dense en batiments,
    en utilisant les centroides et un histogramme par cellule."""

    centroids = gdf.geometry.centroid
    xs = centroids.x.values
    ys = centroids.y.values

    bin_size = window_size / 2
    x_bins = (xs // bin_size).astype(int)
    y_bins = (ys // bin_size).astype(int)

    counts = Counter(zip(x_bins, y_bins))
    best_bin, n = counts.most_common(1)[0]

    center_x = best_bin[0] * bin_size + bin_size
    center_y = best_bin[1] * bin_size + bin_size

    print(f"Zone la plus dense trouvee : ~{n} batiments autour de "
          f"({center_x:.0f}, {center_y:.0f})")

    return center_x, center_y


def main():

    print("=" * 70)
    print(f"CARTE DE CLASSIFICATION ZOOMEE — {SITE}")
    print("=" * 70)

    gpkg_path = OUTPUT / "damage_buildings.gpkg"
    gdf = gpd.read_file(gpkg_path)
    print(f"\n{len(gdf)} batiments charges.")

    if "damage_class" not in gdf.columns:
        raise ValueError("Colonne 'damage_class' absente du fichier.")

    # Trouve la zone la plus dense pour un zoom pertinent
    center_x, center_y = find_densest_area(gdf, ZOOM_SIZE_M)

    half = ZOOM_SIZE_M / 2
    xlim = (center_x - half, center_x + half)
    ylim = (center_y - half, center_y + half)

    # ------------------------------------------------------
    # Trace les batiments, colores par classe, style xBD
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for cls, style in CLASS_STYLE.items():
        subset = gdf[gdf["damage_class"] == cls]
        if len(subset) == 0:
            continue
        subset.plot(ax=ax, facecolor=style["color"], edgecolor="none")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"Prediction — {SITE}\n(zone {ZOOM_SIZE_M}x{ZOOM_SIZE_M}m, "
        f"la plus dense en batiments)",
        color="white", fontsize=13, pad=15,
    )

    legend_handles = [
        mpatches.Patch(color=style["color"], label=style["label"])
        for cls, style in CLASS_STYLE.items()
        if len(gdf[gdf["damage_class"] == cls]) > 0
    ]
    legend = ax.legend(
        handles=legend_handles, loc="upper right",
        facecolor="black", edgecolor="white", fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("white")

    output_file = OUTPUT / "zoomed_damage_map.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"\nCarte sauvegardee : {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()