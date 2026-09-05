"""
00_check_pleiades_coverage.py
Verifie l'emprise geographique reelle de l'image Pleiades et la compare
aux 4 zones d'etude deja utilisees avec Sentinel-2.
"""

import rasterio
from pyproj import Transformer

PLEIADES_FILE = (
    "/Users/ikramechennouf/Documents/"
    "001 - IMG_PXS_MN351a-202309101103360-L3_-G001-177068132.JP2"
)

ZONES = {
    "Amizmiz": (31.2147, -8.2469),
    "Asni": (31.2500, -7.9833),
    "Ighil": (31.05, -8.40),
    "TalatNYaaqoub": (30.9920, -8.1840),
}


def main():

    print("=" * 70)
    print("VERIFICATION DE LA COUVERTURE — IMAGE PLEIADES")
    print("=" * 70)

    print(f"\nOuverture : {PLEIADES_FILE}")

    with rasterio.open(PLEIADES_FILE) as src:

        print(f"\nCRS de l'image        : {src.crs}")
        print(f"Taille                 : {src.width} x {src.height} pixels")
        print(f"Nombre de bandes        : {src.count}")

        bounds = src.bounds
        print(f"\nEmprise (dans le CRS de l'image) :")
        print(f"  left   = {bounds.left}")
        print(f"  bottom = {bounds.bottom}")
        print(f"  right  = {bounds.right}")
        print(f"  top    = {bounds.top}")

        if src.crs is not None and str(src.crs) != "EPSG:4326":
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
        else:
            lon_min, lat_min = bounds.left, bounds.bottom
            lon_max, lat_max = bounds.right, bounds.top

        print(f"\nEmprise en lat/lon (EPSG:4326) :")
        print(f"  Longitude : {lon_min:.5f} -> {lon_max:.5f}")
        print(f"  Latitude  : {lat_min:.5f} -> {lat_max:.5f}")

        largeur_km = abs(lon_max - lon_min) * 111 * 0.85
        hauteur_km = abs(lat_max - lat_min) * 111
        print(f"\nTaille approximative de la zone couverte : "
              f"{largeur_km:.1f} km x {hauteur_km:.1f} km")

    print("\n" + "=" * 70)
    print("COMPARAISON AVEC LES ZONES D'ETUDE")
    print("=" * 70)

    for name, (lat, lon) in ZONES.items():

        inside = (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)

        status = "DANS L'EMPRISE" if inside else "hors emprise"
        print(f"\n{name} (lat={lat}, lon={lon}) : {status}")

        if not inside:
            dist_lon = min(abs(lon - lon_min), abs(lon - lon_max))
            dist_lat = min(abs(lat - lat_min), abs(lat - lat_max))
            print(f"  Ecart approximatif : {dist_lon*111*0.85:.1f} km (longitude), "
                  f"{dist_lat*111:.1f} km (latitude)")

    print("\n" + "=" * 70)
    print("TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
