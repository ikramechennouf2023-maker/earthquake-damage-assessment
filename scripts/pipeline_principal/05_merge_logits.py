# ==========================================================
# 06_merge_logits.py
# PARTIE 1
# IMPORTS + CONFIGURATION
# ==========================================================

import sys
import json
import csv
from pathlib import Path

import numpy as np

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

OUTPUT = output_dir()

PRED_DIR = OUTPUT / "predictions"

METADATA_FILE = OUTPUT / "tiles_metadata.csv"

BOUNDS_FILE = OUTPUT / "bounds.json"

MERGED_LOGITS = OUTPUT / "merged_logits.npy"

LABEL_MAP = OUTPUT / "damage_map.npy"

# ==========================================================
# CHARGEMENT DES DIMENSIONS
# ==========================================================

def load_bounds():

    with open(BOUNDS_FILE, "r") as f:
        bounds = json.load(f)

    width = bounds["width"]
    height = bounds["height"]

    print(f"Largeur  : {width}")
    print(f"Hauteur  : {height}")

    return width, height

# ==========================================================
# CHARGEMENT DES METADONNEES
# ==========================================================

def load_metadata():

    tiles = []

    with open(METADATA_FILE, newline="") as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:

            tiles.append({

                "tile_id": int(row["tile_id"]),

                "x": int(row["x"]),

                "y": int(row["y"]),

                "width": int(row["width"]),

                "height": int(row["height"])

            })

    print(f"{len(tiles)} métadonnées chargées.")

    return tiles

# ==========================================================
# CREATION DE LA CARTE COMPLETE
# ==========================================================

def create_empty_logits(width, height):

    merged = np.zeros(

        (4, height, width),

        dtype=np.float32

    )

    print()

    print("Carte vide créée :")

    print(merged.shape)

    return merged
# ==========================================================
# FUSION DES LOGITS
# ==========================================================

def merge_predictions():

    width, height = load_bounds()

    tiles = load_metadata()

    merged = create_empty_logits(width, height)

    print()
    print("Fusion des prédictions...")
    print()

    for i, tile in enumerate(tiles, start=1):

        prediction_file = PRED_DIR / f"prediction_{tile['tile_id']:06d}.npy"

        if not prediction_file.exists():

            print(f"Prédiction absente : {prediction_file.name}")

            continue

        logits = np.load(prediction_file)

        # (1,4,H,W) -> (4,H,W)
        if logits.ndim == 4:
            logits = logits[0]

        x = tile["x"]
        y = tile["y"]

# Taille réellement disponible dans l'image
        h = min(tile["height"], height - y)
        w = min(tile["width"], width - x)

    merged[:, y:y+h, x:x+w] = logits[:, :h, :w]

    if i % 50 == 0 or i == len(tiles):

            print(f"{i}/{len(tiles)} tuiles fusionnées")

    return merged


# ==========================================================
# CREATION DE LA CARTE DES CLASSES
# ==========================================================

def create_damage_map(merged_logits):

    print()
    print("Calcul de la carte des dégâts...")

    damage_map = np.argmax(
        merged_logits,
        axis=0
    ).astype(np.uint8)

    return damage_map


# ==========================================================
# SAUVEGARDE
# ==========================================================

def save_results(merged_logits, damage_map):

    np.save(MERGED_LOGITS, merged_logits)

    np.save(LABEL_MAP, damage_map)

    print()
    print("Résultats sauvegardés :")
    print(MERGED_LOGITS)
    print(LABEL_MAP)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("06_merge_logits.py")
    print("=" * 60)

    merged_logits = merge_predictions()

    damage_map = create_damage_map(merged_logits)

    save_results(
        merged_logits,
        damage_map
    )

    print()
    print("=" * 60)
    print("06_merge_logits.py terminé avec succès")
    print("=" * 60)


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    main()