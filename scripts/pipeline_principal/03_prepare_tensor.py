"""
==========================================================
04_prepare_tensor.py
Préparation des tenseurs pour le modèle ONNX
==========================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from config import output_dir

# ==========================================================
# DOSSIERS
# ==========================================================

TILES_DIR = output_dir() / "tiles"

PRE_INPUT = TILES_DIR / "pre"
POST_INPUT = TILES_DIR / "post"

TENSORS_DIR = output_dir() / "tensors"

PRE_OUTPUT = TENSORS_DIR / "pre"
POST_OUTPUT = TENSORS_DIR / "post"

# ==========================================================
# DOSSIERS DE SORTIE
# ==========================================================

def create_output_folders():

    PRE_OUTPUT.mkdir(parents=True, exist_ok=True)
    POST_OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LISTE DES TUILES
# ==========================================================

def get_tiles():

    files = sorted(PRE_INPUT.glob("pre_*.png"))

    print(f"Nombre de tuiles : {len(files)}")

    return files

# ==========================================================
# LECTURE IMAGE
# ==========================================================

def read_image(path):

    img = Image.open(path).convert("RGB")

    return np.array(img)

# ==========================================================
# NORMALISATION
# ==========================================================

def normalize(img):

    img = img.astype(np.float32)

    img /= 255.0

    return img

# ==========================================================
# CONVERSION EN TENSEUR
# ==========================================================

def prepare_tensor(path):

    img = read_image(path)

    img = normalize(img)

    # (H,W,C) -> (C,H,W)
    img = np.transpose(img, (2, 0, 1))

    # (C,H,W) -> (1,C,H,W)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

# ==========================================================
# SAUVEGARDE
# ==========================================================

def save_tensor(tensor, output_file):

    np.save(output_file, tensor)

# ==========================================================
# TRAITEMENT D'UNE TUILE
# ==========================================================

def process_tile(pre_file):

    tile_id = pre_file.stem.replace("pre_", "")

    post_file = POST_INPUT / f"post_{tile_id}.png"

    if not post_file.exists():

        print(f"POST absent : {post_file.name}")

        return

    pre_tensor = prepare_tensor(pre_file)
    post_tensor = prepare_tensor(post_file)

    save_tensor(
        pre_tensor,
        PRE_OUTPUT / f"pre_{tile_id}.npy"
    )

    save_tensor(
        post_tensor,
        POST_OUTPUT / f"post_{tile_id}.npy"
    )

# ==========================================================
# TRAITEMENT GLOBAL
# ==========================================================

def process_all():

    files = get_tiles()

    total = len(files)

    for i, pre_file in enumerate(files, start=1):

        process_tile(pre_file)

        if i % 100 == 0 or i == total:

            print(f"{i}/{total} tuiles traitées")

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("04_prepare_tensor.py")
    print("=" * 60)

    create_output_folders()

    process_all()

    print("\nPréparation terminée.")

    print("=" * 60)

# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    main()