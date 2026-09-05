import sys
from pathlib import Path
import json
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import pre_dir, output_dir


def find_band(folder):
    folder = Path(folder)
    matches = list(folder.rglob("*B04*.jp2"))

    if not matches:
        raise FileNotFoundError(f"Aucune bande B04 dans {folder}")

    return matches[0]


def main():

    image = find_band(pre_dir())

    with rasterio.open(image) as src:

        data = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "left": src.bounds.left,
            "bottom": src.bounds.bottom,
            "right": src.bounds.right,
            "top": src.bounds.top
        }

    out = Path(output_dir())
    out.mkdir(parents=True, exist_ok=True)

    outfile = out / "bounds.json"

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("=" * 60)
    print("bounds.json créé")
    print(outfile)
    print("=" * 60)


if __name__ == "__main__":
    main()