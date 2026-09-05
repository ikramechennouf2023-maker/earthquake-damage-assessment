# ==========================================================
# 08_diagnostic_classes.py
# Analyse des classes de dommages
# ==========================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

OUTPUT = Path(output_dir())

DAMAGE_MAP = OUTPUT / "damage_map.npy"

CSV_OUTPUT = OUTPUT / "class_statistics.csv"

FIGURE_OUTPUT = OUTPUT / "class_distribution.png"

# ==========================================================
# CLASSES
# ==========================================================

CLASS_NAMES = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed"
}

# ==========================================================
# CHARGEMENT
# ==========================================================

def load_damage_map():

    if not DAMAGE_MAP.exists():
        raise FileNotFoundError(DAMAGE_MAP)

    return np.load(DAMAGE_MAP)

# ==========================================================
# STATISTIQUES
# ==========================================================

def compute_statistics(damage_map):

    total = damage_map.size

    values, counts = np.unique(
        damage_map,
        return_counts=True
    )

    rows = []

    print("\n==============================")

    for value, count in zip(values, counts):

        percent = 100 * count / total

        name = CLASS_NAMES.get(
            int(value),
            f"Class {value}"
        )

        print(
            f"{name:<15} : "
            f"{count:>10} pixels "
            f"({percent:5.2f}%)"
        )

        rows.append({
            "Class": int(value),
            "Label": name,
            "Pixels": int(count),
            "Percentage": percent
        })

    print("==============================\n")

    return pd.DataFrame(rows)

# ==========================================================
# HISTOGRAMME
# ==========================================================

def plot_distribution(df):

    plt.figure(figsize=(8,5))

    plt.bar(
        df["Label"],
        df["Pixels"]
    )

    plt.title("Distribution des classes")

    plt.xlabel("Classe")

    plt.ylabel("Nombre de pixels")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURE_OUTPUT,
        dpi=300
    )

    plt.close()

# ==========================================================
# SAUVEGARDE
# ==========================================================

def save_csv(df):

    df.to_csv(
        CSV_OUTPUT,
        index=False
    )

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*60)
    print("08_diagnostic_classes.py")
    print("="*60)

    damage_map = load_damage_map()

    df = compute_statistics(
        damage_map
    )

    save_csv(df)

    plot_distribution(df)

    print("CSV enregistré :")
    print(CSV_OUTPUT)

    print()

    print("Histogramme enregistré :")
    print(FIGURE_OUTPUT)

    print()

    print("="*60)
    print("Diagnostic terminé")
    print("="*60)

# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":
    main()