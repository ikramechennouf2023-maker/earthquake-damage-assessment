"""
02b_tile_images_post_only.py — tuilage avec PRE simule a zero
Noms de fichiers : pre_{id}.png / post_{id}.png (sans zero-padding),
pour matcher exactement ce qu'attend 03_prepare_tensor.py.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

SITE = get_site()
OUTPUT = Path(output_dir())

TILE_SIZE = 512


def tile_image(image, tile_size=TILE_SIZE):
    h, w = image.shape[:2]
    tiles, positions = [], []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded
            tiles.append(tile)
            positions.append((x, y))
    return tiles, positions


def main():

    print("=" * 60)
    print(f"02b_tile_images_post_only.py — {SITE}")
    print("=" * 60)

    post_path = OUTPUT / "post_rgb.png"
    if not post_path.exists():
        raise FileNotFoundError(f"post_rgb.png introuvable : {post_path}")

    post_img = np.array(Image.open(post_path).convert("RGB"))
    print(f"\npost_rgb.png charge : {post_img.shape}")

    post_tiles, positions = tile_image(post_img)
    print(f"Tuiles POST : {len(post_tiles)}")

    tile_dir = OUTPUT / "tiles"
    pre_dir = tile_dir / "pre"
    post_dir = tile_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    zero_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

    for i, post_tile in enumerate(post_tiles):
        Image.fromarray(post_tile).save(post_dir / f"post_{i}.png")
        Image.fromarray(zero_tile).save(pre_dir / f"pre_{i}.png")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(post_tiles)} tuiles sauvegardees")

    index = [[i, x, y] for i, (x, y) in enumerate(positions)]
    df = pd.DataFrame(index, columns=["tile", "x", "y"])
    df.to_csv(tile_dir / "tiles_index.csv", index=False)

    print(f"\n{len(post_tiles)} tuiles POST + {len(post_tiles)} tuiles PRE (zero) creees.")
    print(f"Noms : pre_0.png, pre_1.png, ... / post_0.png, post_1.png, ...")

    print("\n" + "=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()
