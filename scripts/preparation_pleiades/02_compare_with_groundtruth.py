"""
==========================================================================
02_compare_with_groundtruth.py
--------------------------------------------------------------------------
Compare les predictions du modele avec le VRAI ground truth officiel
Copernicus EMS (EMSR695, zone Asni), pour savoir si le modele detecte
correctement les batiments reellement endommages.
==========================================================================
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

GROUNDTRUTH_FILE = (
    "/Users/ikramechennouf/Downloads/EMSR695_AOI03_GRA_PRODUCT_v4 2/"
    "EMSR695_AOI03_GRA_PRODUCT_builtUpP_v1.shp"
)

MODEL_FILE = "/Users/ikramechennouf/Documents/model_dinov3/outputs/Asni/damage_buildings.gpkg"

MAX_DISTANCE_M = 30

LABELS = {-1: "Unknown", 0: "No Damage", 1: "Minor Damage", 2: "Major Damage", 3: "Destroyed"}


def main():

    print("=" * 70)
    print("COMPARAISON AVEC LE GROUND TRUTH OFFICIEL (Copernicus EMS)")
    print("=" * 70)

    print("\nChargement du ground truth Copernicus...")
    gt = gpd.read_file(GROUNDTRUTH_FILE)
    print(f"  {len(gt)} batiments confirmes endommages (experts Copernicus)")
    print(f"  Repartition : {gt['damage_gra'].value_counts().to_dict()}")

    print("\nChargement des predictions du modele...")
    model = gpd.read_file(MODEL_FILE)
    print(f"  {len(model)} batiments analyses par le modele")

    target_crs = model.crs

    if gt.crs != target_crs:
        print(f"\nReprojection du ground truth : {gt.crs} -> {target_crs}")
        gt = gt.to_crs(target_crs)

    if model.crs != target_crs:
        model = model.to_crs(target_crs)

    model = model.copy()
    model["geometry"] = model.geometry.centroid

    print(f"\nRecherche du batiment le plus proche pour chacun des {len(gt)} points...")

    matches = gpd.sjoin_nearest(
        gt, model, how="left", distance_col="distance_m",
        max_distance=MAX_DISTANCE_M,
    )

    n_matched = matches["damage_class"].notna().sum()
    n_unmatched = len(matches) - n_matched

    print(f"  {n_matched} batiments trouves a moins de {MAX_DISTANCE_M}m")
    print(f"  {n_unmatched} batiments SANS correspondance proche (ignores)")

    matches = matches[matches["damage_class"].notna()].copy()

    print("\n" + "=" * 70)
    print("RESULTAT : sur les batiments VRAIMENT endommages (confirmes)")
    print("=" * 70)

    matches["model_detecte_dommage"] = matches["damage_class"] >= 1

    n_total = len(matches)
    n_detected = matches["model_detecte_dommage"].sum()
    n_missed = n_total - n_detected

    print(f"\nTotal de batiments compares : {n_total}")
    print(f"  Modele a dit 'dommage' (classe 1/2/3)  : {n_detected}  "
          f"({100*n_detected/n_total:.1f} %)")
    print(f"  Modele a dit 'No Damage' (RATE)         : {n_missed}  "
          f"({100*n_missed/n_total:.1f} %)")

    print("\nDetail — ce que le modele a predit pour ces batiments abimes :")
    detail = matches["damage_class"].value_counts().sort_index()
    for cls, n in detail.items():
        label = LABELS.get(int(cls), f"Classe {cls}")
        print(f"  {label:15s} : {n:4d}  ({100*n/n_total:5.1f} %)")

    print("\nDetail — par gravite officielle Copernicus :")
    for grade in matches["damage_gra"].unique():
        subset = matches[matches["damage_gra"] == grade]
        detected = subset["model_detecte_dommage"].sum()
        total = len(subset)
        print(f"  {grade:20s} : {detected}/{total} detectes par le modele "
              f"({100*detected/total:.1f} %)")

    output_dir = Path("/Users/ikramechennouf/Documents/model_dinov3/outputs/Asni")
    output_file = output_dir / "comparison_with_groundtruth.csv"

    matches[["damage_gra", "damage_class", "distance_m", "model_detecte_dommage"]].to_csv(
        output_file, index=False
    )

    print(f"\nDetail complet sauvegarde : {output_file}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nSur {n_total} batiments dont on est SUR qu'ils sont endommages")
    print(f"(confirme par des experts Copernicus), le modele n'en a")
    print(f"correctement signale que {n_detected} ({100*n_detected/n_total:.1f} %).")
    print(f"\nLe taux de detection reel (Recall) sur les vrais dommages est")
    print(f"de {100*n_detected/n_total:.1f} % — c'est un chiffre officiel,")
    print("calcule a partir d'un vrai ground truth, pas d'une estimation.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
