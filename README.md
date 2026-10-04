# Détection et Classification Automatique des Bâtiments Endommagés par Deep Learning

Détection et classification automatique des dommages aux bâtiments à partir d'images satellitaires, en utilisant une architecture siamoise DINOv3-UperNet. Projet de stage réalisé au sein du **FSEC** (Fonds de Solidarité contre les Événements Catastrophiques, Maroc), avec évaluation sur le dataset **xBD** et application réelle au **séisme d'Al Haouz** (8 septembre 2023).

## 📋 Contexte

Après une catastrophe naturelle, l'évaluation rapide des dommages aux bâtiments est essentielle pour organiser les secours et l'indemnisation des victimes. Les méthodes traditionnelles (inspections de terrain) sont lentes et coûteuses en ressources humaines. Ce projet explore l'utilisation d'un modèle de Deep Learning pré-entraîné pour automatiser cette évaluation à partir d'images satellitaires pré/post-catastrophe.

## 🎯 Ce que fait ce projet

- Évalue un modèle DINOv3-UperNet pré-entraîné sur le dataset de référence **xBD**
- Applique ce modèle à un cas réel : le séisme d'Al Haouz (Maroc, 2023)
- Compare les performances entre imagerie **Sentinel-2** (gratuite, 10 m/pixel) et **Pléiades** (commerciale, ~0,5 m/pixel)
- Valide les résultats face à un référentiel officiel de dommages (**Copernicus EMS**)
- Analyse la calibration des probabilités et la robustesse du seuil de confiance
- Propose une application de démonstration interactive (**Streamlit**)

## 🏗️ Architecture et pipeline

```
Images pré/post-catastrophe
        │
        ▼
Découpage en tuiles 512×512
        │
        ▼
Normalisation (statistiques ImageNet)
        │
        ▼
Backbone DINOv3 (siamois)
        │
        ▼
Décodeur UperNet
        │
        ▼
Logits (1, 4, H, W)
        │
        ▼
Softmax + calibration (T = 1,1719)
        │
        ▼
Carte de dommages (par pixel)
        │
        ▼
Agrégation par empreinte de bâtiment
        │
        ▼
Classe de dommage par bâtiment
   (No Damage / Minor / Major / Destroyed)
```

> **Note** : le modèle DINOv3-UperNet utilisé était déjà pré-entraîné et fourni au format ONNX. Le travail de ce projet porte sur la reconstruction du pipeline d'inférence, le prétraitement, l'évaluation, l'analyse de généralisation et l'application opérationnelle — pas sur l'entraînement du modèle.

## 📊 Résultats clés

### Évaluation sur le dataset xBD

| Métrique | Valeur |
|---|---|
| Accuracy globale | 95,55 % |
| F1-Score pondéré | 94,47 % |
| Accuracy — scènes Mexico | 99,46 % |
| Accuracy — scènes Palu | 91,64 % |

### Application au séisme d'Al Haouz

| Indicateur | Sentinel-2 (10 m) | Pléiades (~0,5 m) |
|---|---|---|
| Recall sur 93 bâtiments confirmés (Copernicus EMS) | **0 %** | — |
| Confiance calibrée moyenne | 0,250 | 0,46 – 0,61 |

**Constat principal** : le modèle, excellent sur son domaine d'entraînement (xBD), voit ses performances s'effondrer sur imagerie Sentinel-2 (résolution 10 m), mais retrouve des niveaux de confiance satisfaisants sur imagerie Pléiades (résolution sub-métrique, proche du domaine d'entraînement). Ce résultat suggère que la résolution spatiale constitue un facteur majeur expliquant les performances observées, avec une implication opérationnelle directe pour le FSEC : l'évaluation fiable des dommages avec ce type de modèle nécessite une imagerie à très haute résolution.

## 📁 Structure du dépôt

```
.
├── main.py                        # Point d'entrée principal du pipeline
├── config.py / config.yaml        # Configuration du projet
├── calibration.json               # Paramètres de calibration du modèle
├── requirements.txt               # Dépendances Python
├── scripts/
│   ├── pipeline_principal/        # Pipeline complet : tuilage, inférence, 
│   │                               # évaluation, visualisation
│   └── preparation_pleiades/      # Scripts spécifiques au traitement 
│                                   # des images Pléiades (Al Haouz)
├── model_dinov3_3/                # Notebooks et exports de synthèse
└── structure.txt                  # Arborescence détaillée du projet
```

> Les dossiers de données volumineux (`input/`, `model/`, `outputs/`, `tensors/`, `tiles/`) ne sont pas versionnés sur ce dépôt en raison de leur taille (plusieurs Go) — voir `.gitignore`.

## ⚙️ Installation

```bash
git clone https://github.com/ikramechennouf2023-maker/earthquake-damage-assessment.git
cd earthquake-damage-assessment
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt --break-system-packages
```

## 🛠️ Technologies utilisées

- **Modèle** : DINOv3 (backbone, Meta AI) + UperNet (décodeur de segmentation)
- **Inférence** : ONNX Runtime (avec accélération CoreML sur macOS)
- **Traitement géospatial** : Rasterio, GeoPandas, Shapely, PyProj
- **Traitement d'image** : Pillow, OpenCV, NumPy
- **Évaluation** : Scikit-learn
- **Application de démonstration** : Streamlit
- **Langage** : Python 3

## 📚 Données

- **[xBD](https://xview2.org/)** — dataset de référence pour l'évaluation des dommages post-catastrophe (Gupta et al., 2019)
- **Sentinel-2** — imagerie optique gratuite, programme Copernicus
- **Pléiades** — imagerie commerciale à très haute résolution
- **[Copernicus EMS](https://rapidmapping.emergency.copernicus.eu/EMSR695)** — référentiel officiel de dommages (activation EMSR695, séisme d'Al Haouz)

## 🎓 Contexte académique

Projet réalisé dans le cadre d'un stage de fin d'études à l'**Université Euro-Méditerranéenne de Fès (UEMF)**, filière Ingénierie Digitale et Intelligence Artificielle, au sein du **Fonds de Solidarité contre les Événements Catastrophiques (FSEC)**, Maroc.

## 👥 Auteurs

- **Ikrame Chennouf**
- **Firdawss El Hayouni**

**Encadrante** : Mme Lamya Amghar

## 📄 Licence

Ce projet est distribué sous licence MIT — voir le fichier `LICENSE` pour plus de détails.
