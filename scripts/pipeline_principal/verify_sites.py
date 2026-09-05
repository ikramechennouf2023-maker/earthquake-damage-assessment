"""
==========================================================================
verify_sites.py
--------------------------------------------------------------------------
Verifie que chaque zone (input/<site>/pre et post) correspond bien a des
donnees Sentinel-2 distinctes, en comparant les VRAIES metadonnees
(tuile MGRS, date d'acquisition, emprise GPS exacte) plutot que l'aspect
visuel des vignettes (qui peut se ressembler entre villages voisins).

Lance simplement :
    python3 verify_sites.py
==========================================================================
"""

import re
from pathlib import Path
import rasterio


def find_project_root(start):
    """Remonte les dossiers parents jusqu'a trouver celui contenant input/,
    pour que ce script fonctionne peu importe ou il est place."""
    current = Path(start).resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "input").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Impossible de trouver un dossier 'input/' en remontant depuis {start}. "
        "Verifie que ce script est bien dans le projet model_dinov3/ (racine ou scripts/)."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)
INPUT_DIR = PROJECT_ROOT / "input"

print(f"Projet detecte : {PROJECT_ROOT}")


def find_safe(folder):
    """Trouve le vrai dossier .SAFE officiel (celui qui contient un
    sous-dossier GRANULE), pas un eventuel dossier .SAFE "emballage"
    externe qui pourrait exister au-dessus (ex: nommage personnalise
    du type MonSite_PRE_.SAFE)."""

    folder = Path(folder)
    candidates = [d for d in folder.rglob("*.SAFE") if d.is_dir()]

    # Ne garde que ceux qui contiennent vraiment GRANULE (le vrai produit)
    valid = [d for d in candidates if (d / "GRANULE").is_dir()]

    if valid:
        # S'il y en a plusieurs, prend le plus profond (le plus specifique)
        return max(valid, key=lambda d: len(d.parts))

    return None


def parse_safe_name(safe_folder):
    """Extrait tuile MGRS et date d'acquisition depuis le nom du .SAFE."""
    name = safe_folder.name
    # Ex: S2A_MSIL2A_20230908T110621_N0510_R137_T29RNQ_20241023T144814.SAFE
    m = re.search(r"_(\d{8})T\d{6}_N\d{4}_R\d{3}_(T\d{2}[A-Z]{3})_", name)
    if m:
        date, tile = m.group(1), m.group(2)
        return date, tile
    return "INCONNU", "INCONNU"


def find_band(safe_folder, band="B04"):
    matches = list(safe_folder.rglob(f"IMG_DATA/R10m/*_{band}_10m.jp2"))
    if not matches:
        matches = [m for m in safe_folder.rglob(f"*_{band}_10m.jp2")
                   if "QI_DATA" not in m.parts]
    return matches[0] if matches else None


def get_bounds(band_path):
    with rasterio.open(band_path) as src:
        b = src.bounds
        return (round(b.left, 4), round(b.bottom, 4), round(b.right, 4), round(b.top, 4))


def analyze_site(site_folder):

    result = {"site": site_folder.name}

    for period in ["pre", "post"]:
        folder = site_folder / period
        safe = find_safe(folder)

        if safe is None:
            result[period] = {"error": "Aucun .SAFE trouve"}
            continue

        date, tile = parse_safe_name(safe)
        band_path = find_band(safe)

        if band_path is None:
            result[period] = {"date": date, "tile": tile, "error": "Bande B04 introuvable"}
            continue

        bounds = get_bounds(band_path)
        result[period] = {"date": date, "tile": tile, "bounds": bounds, "safe_name": safe.name}

    return result


def main():

    print("=" * 90)
    print("VERIFICATION DES ZONES — comparaison des vraies metadonnees Sentinel-2")
    print("=" * 90)

    sites = [f for f in sorted(INPUT_DIR.iterdir())
             if f.is_dir() and (f / "pre").exists() and (f / "post").exists()]

    if not sites:
        print(f"Aucune zone trouvee dans {INPUT_DIR}")
        return

    all_results = []

    for site_folder in sites:
        print(f"\nAnalyse : {site_folder.name}...")
        all_results.append(analyze_site(site_folder))

    print("\n" + "=" * 90)
    print("RESUME")
    print("=" * 90)

    for r in all_results:
        print(f"\nZone : {r['site']}")
        for period in ["pre", "post"]:
            info = r.get(period, {})
            if "error" in info:
                print(f"  {period.upper():5s} : ERREUR - {info['error']}")
            else:
                print(f"  {period.upper():5s} : date={info['date']}  tuile={info['tile']}  bounds={info['bounds']}")

    # ------------------------------------------------------
    # Comparaison croisee : bounds identiques entre sites ?
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("VERIFICATION CROISEE")
    print("=" * 90)

    post_bounds = {}
    for r in all_results:
        info = r.get("post", {})
        if "bounds" in info:
            post_bounds.setdefault(info["bounds"], []).append(r["site"])

    identical_found = False
    for bounds, site_list in post_bounds.items():
        if len(site_list) > 1:
            identical_found = True
            print(f"\n  ATTENTION : ces zones ont EXACTEMENT les memes coordonnees POST :")
            print(f"    {site_list}")
            print(f"    Bounds : {bounds}")
            print(f"    -> Elles traitent la meme emprise geographique, pas des zones distinctes.")

    if not identical_found:
        print("\n  OK : chaque zone a des coordonnees POST distinctes.")

    # Verifier aussi les dates PRE/POST identiques par erreur (meme date pour les 2)
    print()
    for r in all_results:
        pre_date = r.get("pre", {}).get("date")
        post_date = r.get("post", {}).get("date")
        if pre_date and post_date and pre_date == post_date:
            print(f"  ATTENTION : {r['site']} — PRE et POST ont la MEME date ({pre_date}). "
                  f"Ce n'est probablement pas voulu (pas de comparaison avant/apres possible).")

    print("\n" + "=" * 90)
    print("TERMINE")
    print("=" * 90)


if __name__ == "__main__":
    main()