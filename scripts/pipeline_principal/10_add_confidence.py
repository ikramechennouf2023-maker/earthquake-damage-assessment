"""
==========================================================================
10_add_confidence.py
--------------------------------------------------------------------------
Ajoute la CONFIANCE CALIBREE (softmax(logits/temperature)) et un
indicateur "needs_review" (seuil officiel du modele) a une COPIE du
fichier de resultats existant. Ne modifie PAS damage_buildings.gpkg
original, pour ne rien casser dans le pipeline principal.

Utilise :
  - calibration.json (temperature = 1.1719)
  - config.yaml (confidence_threshold = 0.5)
  - merged_logits.npy (les VRAIS logits, pas juste la classe finale)

Lance depuis la racine du projet, apres avoir choisi une zone via main.py :
    python3 10_add_confidence.py
==========================================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

SITE = get_site()
OUTPUT = Path(output_dir())

BUILDINGS_FILE = OUTPUT / "damage_buildings.gpkg"
LOGITS_FILE = OUTPUT / "merged_logits.npy"
BOUNDS_FILE = OUTPUT / "bounds.json"

OUTPUT_FILE = OUTPUT / "damage_buildings_with_confidence.gpkg"

# ==========================================================
# PARAMETRES OFFICIELS DU MODELE
# ==========================================================

with open(PROJECT_ROOT / "calibration.json") as f:
    CALIBRATION = json.load(f)

TEMPERATURE = CALIBRATION["temperature"]  # 1.1719

CONFIDENCE_THRESHOLD = 0.25  # config.yaml : confidence_threshold

LABELS = {0: "No Damage", 1: "Minor Damage", 2: "Major Damage", 3: "Destroyed"}


def softmax(logits, temperature):
    scaled = logits / temperature
    scaled = scaled - np.max(scaled, axis=0, keepdims=True)  # stabilite numerique
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=0, keepdims=True)


def load_logits():
    """Charge merged_logits.npy et le remet au format (4, H, W)."""

    logits = np.load(LOGITS_FILE)
    print(f"Shape logits chargee : {logits.shape}")

    if logits.shape[0] == 4:
        return logits  # deja (4, H, W)
    elif logits.shape[-1] == 4:
        return np.transpose(logits, (2, 0, 1))  # (H, W, 4) -> (4, H, W)
    else:
        raise ValueError(f"Shape de logits inattendue : {logits.shape}")


def get_local_window(geometry, transform, width, height):

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


def compute_building_confidence(geometry, probs, transform, width, height):
    """Calcule la classe predite ET sa confiance calibree pour un batiment,
    en regroupant tous les pixels du batiment (moyenne des probabilites,
    comme le modele le fait au niveau objet)."""

    if geometry is None or geometry.is_empty:
        return -1, 0.0

    window = get_local_window(geometry, transform, width, height)
    if window is None:
        return -1, 0.0

    row0, row1, col0, col1 = window
    local_transform = build_local_transform(transform, row0, col0)
    out_shape = (row1 - row0, col1 - col0)

    mask = rasterize(
        [(geometry, 1)], out_shape=out_shape, transform=local_transform,
        fill=0, all_touched=True, dtype=np.uint8,
    )

    if mask.sum() == 0:
        return -1, 0.0

    local_probs = probs[:, row0:row1, col0:col1]  # (4, h, w)

    # Moyenne des probabilites sur tous les pixels du batiment
    building_probs = local_probs[:, mask == 1].mean(axis=1)  # (4,)

    predicted_class = int(np.argmax(building_probs))
    confidence = float(building_probs[predicted_class])

    return predicted_class, confidence


def main():

    print("=" * 70)
    print(f"10_add_confidence.py — {SITE}")
    print("=" * 70)

    print(f"\nTemperature de calibration : {TEMPERATURE}")
    print(f"Seuil de confiance officiel : {CONFIDENCE_THRESHOLD}")

    if not LOGITS_FILE.exists():
        raise FileNotFoundError(
            f"merged_logits.npy introuvable : {LOGITS_FILE}\n"
            "Ce fichier doit etre genere par 05_merge_logits.py avant de "
            "lancer ce script."
        )

    print("\nChargement des logits fusionnes...")
    logits = load_logits()

    print("Calcul du softmax calibre (temperature = calibration.json)...")
    probs = softmax(logits, TEMPERATURE)

    with open(BOUNDS_FILE) as f:
        bounds = json.load(f)

    transform = from_bounds(
        bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
        bounds["width"], bounds["height"],
    )

    print("\nChargement des batiments...")
    gdf = gpd.read_file(BUILDINGS_FILE)
    print(f"{len(gdf)} batiments charges.")

    image_crs = bounds["crs"]
    if str(gdf.crs) != str(image_crs):
        print(f"Reprojection : {gdf.crs} -> {image_crs}")
        gdf = gdf.to_crs(image_crs)

    print("\nCalcul de la confiance calibree par batiment...")

    predicted_classes = []
    confidences = []

    total = len(gdf)
    for i, geom in enumerate(gdf.geometry, start=1):

        cls, conf = compute_building_confidence(
            geom, probs, transform, probs.shape[2], probs.shape[1]
        )
        predicted_classes.append(cls)
        confidences.append(conf)

        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} batiments traites")

    gdf["damage_class_v2"] = predicted_classes
    gdf["confidence"] = confidences
    gdf["needs_review"] = gdf["confidence"] < CONFIDENCE_THRESHOLD

    gdf.to_file(OUTPUT_FILE, driver="GPKG")

    print(f"\nResultat sauvegarde : {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)

    print(f"\nConfiance moyenne : {gdf['confidence'].mean():.3f}")
    print(f"Confiance mediane : {gdf['confidence'].median():.3f}")

    n_review = gdf["needs_review"].sum()
    pct_review = 100 * n_review / total if total else 0
    print(f"\nBatiments sous le seuil ({CONFIDENCE_THRESHOLD}) — a reviser : "
          f"{n_review}/{total} ({pct_review:.1f}%)")

    print("\nConfiance moyenne par classe predite :")
    for cls, label in LABELS.items():
        subset = gdf[gdf["damage_class_v2"] == cls]
        if len(subset) > 0:
            print(f"  {label:15s} : confiance moyenne = {subset['confidence'].mean():.3f} "
                  f"({len(subset)} batiments)")

    # Verifie la coherence avec l'ancienne classification (argmax brut)
    if "damage_class" in gdf.columns:
        agreement = (gdf["damage_class"] == gdf["damage_class_v2"]).sum()
        print(f"\nAccord avec l'ancienne classification (argmax brut) : "
              f"{agreement}/{total} ({100*agreement/total:.1f}%)")

    print("\n" + "=" * 70)
    print("TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()