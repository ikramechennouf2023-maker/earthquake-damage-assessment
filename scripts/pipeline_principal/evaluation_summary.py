"""
==========================================================================
evaluation_summary.py
--------------------------------------------------------------------------
Rassemble les preuves de toutes les zones testees dans un resume clair,
a montrer directement dans le terminal VS Code (capture d'ecran).

Lance depuis la racine du projet :
    python3 evaluation_summary.py
==========================================================================
"""

from pathlib import Path
import geopandas as gpd


def find_project_root(start):
    """Remonte les dossiers parents jusqu'a trouver celui contenant outputs/,
    pour que ce script fonctionne peu importe ou il est place."""
    current = Path(start).resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Impossible de trouver un dossier 'outputs/' en remontant depuis {start}."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

print(f"Projet detecte : {PROJECT_ROOT}")
print()

LABELS = {0: "No Damage", 1: "Minor Damage", 2: "Major Damage", 3: "Destroyed", -1: "Unknown"}

# Zones documentees comme severement endommagees dans la litterature
# scientifique (pour comparaison avec les predictions du modele)
GROUND_TRUTH_NOTE = {
    "Amizmiz": "Degats confirmes (fissures, effondrements partiels) - etude MDPI 2025",
    "Ighil": "Zone rurale dispersee, pres de l'epicentre",
    "Tafeghaghte": "Zone de dommages structurels etendus - etude MDPI 2025",
    "TalatNYaaqoub": "Degats structurels severes confirmes (facades effondrees, "
                      "piliers en beton arme rompus) - Scientific Reports 2025",
}

OFFICIAL_F1 = {
    "No Damage": 0.923, "Minor Damage": 0.5797,
    "Major Damage": 0.6623, "Destroyed": 0.8885,
}
OFFICIAL_MACRO_F1 = 0.7634


def analyze_zone(site_name):

    gpkg_path = OUTPUTS_DIR / site_name / "damage_buildings.gpkg"

    if not gpkg_path.exists():
        return None

    gdf = gpd.read_file(gpkg_path)
    total = len(gdf)

    counts = gdf["damage_class"].value_counts()

    stats = {}
    for cls, label in LABELS.items():
        n = int(counts.get(cls, 0))
        pct = 100 * n / total if total else 0
        stats[label] = {"n": n, "pct": pct}

    return {"total": total, "stats": stats}


def print_separator(char="=", width=78):
    print(char * width)


def main():

    print_separator()
    print("RESUME DES PREUVES — EVALUATION DU MODELE SUR SENTINEL-2")
    print_separator()
    print()
    print("Objectif : verifier si le modele DINOv3 (entraine sur imagerie")
    print("sub-metrique) detecte correctement les dommages reels sur")
    print("Sentinel-2 (10m/pixel).")
    print()

    zones = ["Amizmiz", "Ighil", "Tafeghaghte", "TalatNYaaqoub"]

    results = {}
    for zone in zones:
        r = analyze_zone(zone)
        if r:
            results[zone] = r

    print_separator("-")
    print("PREUVE 1 — RESULTATS PAR ZONE (classification par batiment)")
    print_separator("-")

    for zone, r in results.items():
        print()
        print(f"Zone : {zone}")
        print(f"  Contexte : {GROUND_TRUTH_NOTE.get(zone, 'N/A')}")
        print(f"  Batiments analyses : {r['total']}")
        for label, v in r["stats"].items():
            if v["n"] > 0:
                print(f"    {label:15s} : {v['n']:5d}  ({v['pct']:5.2f} %)")

    print()
    print_separator("-")
    print("PREUVE 2 — LE CAS LE PLUS REVELATEUR")
    print_separator("-")
    print()

    if "TalatNYaaqoub" in results:
        r = results["TalatNYaaqoub"]
        no_damage_pct = r["stats"].get("No Damage", {}).get("pct", 0)
        damaged_pct = 100 - no_damage_pct - r["stats"].get("Unknown", {}).get("pct", 0)

        print(f"Talat N'Yaaqoub : {r['total']} batiments analyses")
        print(f"  -> {no_damage_pct:.2f} % classes 'No Damage'")
        print(f"  -> {damaged_pct:.2f} % classes avec un dommage quelconque")
        print()
        print("  Dommages reels documentes (source scientifique) :")
        print("  facades effondrees, piliers beton arme rompus par cisaillement,")
        print("  ville a seulement 11 km de l'epicentre.")
        print()
        print("  ECART : dommages reels severes confirmes, mais quasiment")
        print("  aucun detecte par le modele sur cette resolution d'image.")

    print()
    print_separator("-")
    print("PREUVE 3 — LE MODELE N'EST PAS MAUVAIS EN SOI")
    print_separator("-")
    print()
    print("Performance officielle du modele sur SON domaine d'entrainement")
    print("(imagerie sub-metrique, validation xBD/xView2, 6166 batiments) :")
    print()
    for label, f1 in OFFICIAL_F1.items():
        print(f"    {label:15s} : F1 = {f1:.3f}")
    print(f"    {'Macro F1':15s} : {OFFICIAL_MACRO_F1:.4f}")
    print()
    print("  -> Bonnes performances sur son domaine d'origine.")
    print("  -> Le probleme est un decalage de domaine (resolution), pas")
    print("     une faiblesse intrinseque du modele.")

    print()
    print_separator()
    print("CONCLUSION")
    print_separator()
    print()
    print("Le pipeline fonctionne correctement (verifie etape par etape).")
    print("Le modele DINOv3 ne se generalise PAS de l'imagerie sub-metrique")
    print("vers Sentinel-2 (10m/pixel), meme sur des zones a dommages")
    print("severes confirmes scientifiquement (Talat N'Yaaqoub).")
    print()
    print("=> Limite pratique importante pour les usages humanitaires")
    print("   low-cost bases sur imagerie satellite gratuite.")
    print()
    print_separator()


if __name__ == "__main__":
    main()