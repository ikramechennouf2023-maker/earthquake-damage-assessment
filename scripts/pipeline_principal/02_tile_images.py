"""
==========================================================
03_tile_images.py
PARTIE 1
Découpage des images RGB Sentinel en tuiles 512x512
==========================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from config import output_dir

# ==========================================================
# PARAMÈTRES
# ==========================================================

TILE_SIZE = 512
STRIDE = 512


# ==========================================================
# OUVERTURE DES IMAGES RGB
# ==========================================================

def open_images():

    pre_path = output_dir() / "pre_rgb.png"
    post_path = output_dir() / "post_rgb.png"

    if not pre_path.exists():
        raise FileNotFoundError(pre_path)

    if not post_path.exists():
        raise FileNotFoundError(post_path)

    pre = np.array(Image.open(pre_path))
    post = np.array(Image.open(post_path))

    print(f"PRE  : {pre.shape}")
    print(f"POST : {post.shape}")

    return pre, post


# ==========================================================
# CRÉATION DES DOSSIERS
# ==========================================================

def create_output_folders():

    root = output_dir() / "tiles"

    pre_folder = root / "pre"
    post_folder = root / "post"

    pre_folder.mkdir(parents=True, exist_ok=True)
    post_folder.mkdir(parents=True, exist_ok=True)

    return pre_folder, post_folder


# ==========================================================
# PADDING D'UNE TUILE
# ==========================================================

def pad_tile(tile):

    h, w = tile.shape[:2]

    if h == TILE_SIZE and w == TILE_SIZE:
        return tile

    padded = np.zeros(
        (TILE_SIZE, TILE_SIZE, 3),
        dtype=np.uint8
    )

    padded[:h, :w] = tile

    return padded


# ==========================================================
# SAUVEGARDE D'UNE TUILE
# ==========================================================

def save_tile(tile, output_file):

    Image.fromarray(tile).save(output_file)


# ==========================================================
# DÉCOUPAGE EN TUILES
# ==========================================================

def create_tiles(
    pre_img,
    post_img,
    pre_folder,
    post_folder
):

    height, width, _ = pre_img.shape

    print(f"Largeur : {width}")
    print(f"Hauteur : {height}")

    metadata = []

    tile_id = 0

    for y in range(0, height, STRIDE):

        for x in range(0, width, STRIDE):

            pre_tile = pre_img[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            post_tile = post_img[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            pre_tile = pad_tile(pre_tile)
            post_tile = pad_tile(post_tile)

            # Ignorer les tuiles complètement noires
            if np.max(pre_tile) == 0 and np.max(post_tile) == 0:
                continue

            pre_name = f"pre_{tile_id:06d}.png"
            post_name = f"post_{tile_id:06d}.png"

            save_tile(
                pre_tile,
                pre_folder / pre_name
            )

            save_tile(
                post_tile,
                post_folder / post_name
            )

            metadata.append([
                tile_id,
                x,
                y,
                TILE_SIZE,
                TILE_SIZE
            ])

            tile_id += 1

            if tile_id % 100 == 0:

                print(f"{tile_id} tuiles créées")

    return metadata
# ==========================================================
# SAUVEGARDE DES MÉTADONNÉES
# ==========================================================

def save_metadata(metadata):

    csv_file = output_dir() / "tiles_metadata.csv"

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "tile_id",
            "x",
            "y",
            "width",
            "height"
        ])

        writer.writerows(metadata)

    print(f"\nMétadonnées sauvegardées : {csv_file}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("03_tile_images.py")
    print("=" * 60)

    # ------------------------------------------------------
    # Ouverture des images RGB
    # ------------------------------------------------------

    pre_img, post_img = open_images()

    # Vérification des dimensions
    if pre_img.shape != post_img.shape:
        raise ValueError(
            "Les images PRE et POST n'ont pas les mêmes dimensions."
        )

    # ------------------------------------------------------
    # Création des dossiers
    # ------------------------------------------------------

    pre_folder, post_folder = create_output_folders()

    # ------------------------------------------------------
    # Découpage en tuiles
    # ------------------------------------------------------

    metadata = create_tiles(
        pre_img,
        post_img,
        pre_folder,
        post_folder
    )

    # ------------------------------------------------------
    # Sauvegarde des métadonnées
    # ------------------------------------------------------

    save_metadata(metadata)

    print(f"\nNombre total de tuiles : {len(metadata)}")

    print("\nStructure créée :")
    print(output_dir() / "tiles")

    print("\n" + "=" * 60)
    print("03_tile_images.py terminé avec succès")
    print("=" * 60)


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":
    main()