# ==========================================================
# report_generator.py
# PARTIE 1
# IMPORTS + CONFIGURATION + PAGE DE GARDE
# ==========================================================

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

# ==========================================================
# AJOUT DU DOSSIER RACINE
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_site, output_dir

# ==========================================================
# CONFIGURATION
# ==========================================================

SITE = get_site()

OUTPUT = Path(output_dir())

PDF_FILE = OUTPUT / "Final_Report.pdf"

# ==========================================================
# IMAGES
# ==========================================================

PRE_IMAGE = OUTPUT / "pre_rgb.png"
POST_IMAGE = OUTPUT / "post_rgb.png"

PREDICTION_IMAGE = OUTPUT / "damage_map.png"

VISUALIZATION = OUTPUT / "visualization.png"

CLASS_DIST = OUTPUT / "class_distribution.png"

CONFUSION = OUTPUT / "confusion_matrix.png"

# ==========================================================
# CSV
# ==========================================================

CLASS_STAT = OUTPUT / "class_statistics.csv"

METRICS = OUTPUT / "evaluation_metrics.csv"

# ==========================================================
# DOCUMENT PDF
# ==========================================================

doc = SimpleDocTemplate(
    str(PDF_FILE),
    pagesize=(21 * cm, 29.7 * cm),
    rightMargin=1.5 * cm,
    leftMargin=1.5 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.5 * cm
)

story = []

styles = getSampleStyleSheet()

# ==========================================================
# STYLES
# ==========================================================

title_style = styles["Heading1"]
title_style.alignment = TA_CENTER
title_style.textColor = HexColor("#003366")
title_style.spaceAfter = 15

heading = styles["Heading2"]
heading.textColor = HexColor("#1F618D")
heading.spaceBefore = 10
heading.spaceAfter = 8

normal = styles["BodyText"]
normal.leading = 20

# ==========================================================
# PAGE DE GARDE
# ==========================================================

story.append(
    Paragraph(
        "BUILDING DAMAGE ASSESSMENT REPORT",
        title_style
    )
)

story.append(Spacer(1, 1 * cm))

story.append(
    Paragraph(
        "<b>Deep Learning Building Damage Detection using Sentinel-2 Images</b>",
        normal
    )
)

story.append(Spacer(1, 0.8 * cm))

story.append(
    Paragraph(
        "<b>University :</b> Euro-Mediterranean University of Fez (UEMF)",
        normal
    )
)

story.append(
    Paragraph(
        "<b>Project :</b> End-of-Year Internship (PFA)",
        normal
    )
)

story.append(
    Paragraph(
        f"<b>Site :</b> {SITE}",
        normal
    )
)

story.append(
    Paragraph(
        "<b>Model :</b> DINOv3 (ONNX)",
        normal
    )
)

story.append(
    Paragraph(
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}",
        normal
    )
)

story.append(Spacer(1, 2 * cm))

story.append(
    Paragraph(
        "<b>Pipeline Overview</b>",
        heading
    )
)

story.append(
    Paragraph(
        """
        sentinel_to_rgb.py<br/>
        01_get_bounds.py<br/>
        02_crop_building.py<br/>
        03_tile_images.py<br/>
        04_prepare_tensor.py<br/>
        05_run_model.py<br/>
        06_merge_logits.py<br/>
        07_filtre_building.py<br/>
        08_building_damage.py<br/>
        09_visualize.py<br/>
        10_diagnostic_classes.py<br/>
        11_evaluation_metrics.py<br/>
        report_generator.py
        """,
        normal
    )
)

story.append(PageBreak())


# ==========================================================
# PARTIE 2
# IMAGES PRE / POST
# ==========================================================

def image_exists(path):

    return Path(path).exists()


# ==========================================================
# SECTION PRE / POST
# ==========================================================

story.append(

    Paragraph(

        "1. Input Sentinel Images",

        heading

    )

)

story.append(

    Paragraph(

        """
The following figures show the Sentinel-2 images
used as input for the Deep Learning model.
The PRE image corresponds to the satellite image
before the disaster, while the POST image
corresponds to the satellite image acquired
after the disaster.
""",

        normal

    )

)

story.append(Spacer(1,0.5*cm))

# ==========================================================
# PRE IMAGE
# ==========================================================

if image_exists(PRE_IMAGE):

    pre = Image(
        str(PRE_IMAGE),
        width=8*cm,
        height=8*cm
    )

else:

    pre = Paragraph(
        "<b>PRE image not found.</b>",
        normal
    )

# ==========================================================
# POST IMAGE
# ==========================================================

if image_exists(POST_IMAGE):

    post = Image(
        str(POST_IMAGE),
        width=8*cm,
        height=8*cm
    )

else:

    post = Paragraph(
        "<b>POST image not found.</b>",
        normal
    )

# ==========================================================
# TABLEAU
# ==========================================================

table = Table(

    [

        [

            pre,

            post

        ],

        [

            Paragraph(

                "<b>PRE Image</b>",

                normal

            ),

            Paragraph(

                "<b>POST Image</b>",

                normal

            )

        ]

    ],

    colWidths=[9*cm,9*cm]

)

table.setStyle(

    TableStyle(

        [

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

            ("GRID",(0,0),(-1,-1),0.25,colors.grey),

            ("BOTTOMPADDING",(0,0),(-1,-1),10),

            ("TOPPADDING",(0,0),(-1,-1),10)

        ]

    )

)

story.append(table)

story.append(Spacer(1,0.8*cm))

# ==========================================================
# INFORMATIONS
# ==========================================================

story.append(

    Paragraph(

        "Image Information",

        heading

    )

)

rows = [

    ["Parameter","Value"],

    ["Satellite","Sentinel-2"],

    ["Bands used","RGB"],

    ["Tile Size","512 × 512"],

    ["Model Input","(1,3,512,512)"],

    ["Site",SITE]

]

info = Table(

    rows,

    colWidths=[7*cm,9*cm]

)

info.setStyle(

    TableStyle(

        [

            ("BACKGROUND",(0,0),(-1,0),HexColor("#D6EAF8")),

            ("GRID",(0,0),(-1,-1),0.5,colors.grey),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ]

    )

)

story.append(info)

story.append(PageBreak())


# ==========================================================
# PARTIE 3
# PREDICTION + BUILDING DAMAGE DETECTION
# ==========================================================

# ==========================================================
# DAMAGE MAP
# ==========================================================

story.append(
    Paragraph(
        "2. Damage Prediction Map",
        heading
    )
)

story.append(
    Paragraph(
        """
The following figure represents the pixel-wise damage
prediction generated by the DINOv3 segmentation model.
Each pixel is assigned to one of the four damage classes.
""",
        normal
    )
)

story.append(Spacer(1,0.5*cm))

if image_exists(PREDICTION_IMAGE):

    pred = Image(
        str(PREDICTION_IMAGE),
        width=16*cm,
        height=16*cm
    )

    story.append(pred)

else:

    story.append(
        Paragraph(
            "<b>Prediction image not found.</b>",
            normal
        )
    )

story.append(Spacer(1,0.8*cm))

# ==========================================================
# LEGEND
# ==========================================================

story.append(
    Paragraph(
        "Damage Classes",
        heading
    )
)

legend = Table(

    [

        ["Class","Description"],

        ["0","🟩 No Damage"],

        ["1","🟨 Minor Damage"],

        ["2","🟧 Major Damage"],

        ["3","🟥 Destroyed"]

    ],

    colWidths=[3*cm,12*cm]

)

legend.setStyle(

    TableStyle(

        [

            ("BACKGROUND",(0,0),(-1,0),HexColor("#D6EAF8")),

            ("GRID",(0,0),(-1,-1),0.5,colors.grey),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ]

    )

)

story.append(legend)

story.append(PageBreak())

# ==========================================================
# BUILDING DAMAGE DETECTION
# ==========================================================

story.append(
    Paragraph(
        "3. Building Damage Detection",
        heading
    )
)

story.append(
    Paragraph(
        """
Detected building footprints are overlaid on the
post-disaster Sentinel-2 image.
Each building is coloured according to the dominant
predicted damage class.
""",
        normal
    )
)

story.append(Spacer(1,0.5*cm))

if image_exists(VISUALIZATION):

    vis = Image(
        str(VISUALIZATION),
        width=17*cm,
        height=17*cm
    )

    story.append(vis)

else:

    story.append(
        Paragraph(
            "<b>Visualization image not found.</b>",
            normal
        )
    )

story.append(Spacer(1,0.7*cm))

# ==========================================================
# COLOR DESCRIPTION
# ==========================================================

colors_table = Table(

    [

        ["Color","Meaning"],

        ["Green","No Damage"],

        ["Yellow","Minor Damage"],

        ["Orange","Major Damage"],

        ["Red","Destroyed"],

        ["Gray","Unknown / No Building"]

    ],

    colWidths=[4*cm,11*cm]

)

colors_table.setStyle(

    TableStyle(

        [

            ("BACKGROUND",(0,0),(-1,0),HexColor("#AED6F1")),

            ("GRID",(0,0),(-1,-1),0.5,colors.black),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ]

    )

)

story.append(colors_table)

story.append(PageBreak())



# ==========================================================
# PARTIE 4
# STATISTICS + METRICS
# ==========================================================

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

story.append(
    Paragraph(
        "4. Damage Class Distribution",
        heading
    )
)

story.append(
    Paragraph(
        """
This chart illustrates the percentage of pixels
predicted for each damage category.
""",
        normal
    )
)

story.append(Spacer(1,0.4*cm))

if image_exists(CLASS_DIST):

    story.append(

        Image(
            str(CLASS_DIST),
            width=15*cm,
            height=9*cm
        )

    )

else:

    story.append(
        Paragraph(
            "<b>class_distribution.png not found.</b>",
            normal
        )
    )

story.append(Spacer(1,0.8*cm))

# ==========================================================
# CLASS STATISTICS TABLE
# ==========================================================

story.append(
    Paragraph(
        "Class Statistics",
        heading
    )
)

if image_exists(CLASS_STAT):

    df = pd.read_csv(CLASS_STAT)

    data = [list(df.columns)]

    data += df.values.tolist()

    table = Table(data)

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),HexColor("#AED6F1")),

            ("TEXTCOLOR",(0,0),(-1,0),colors.black),

            ("GRID",(0,0),(-1,-1),0.4,colors.grey),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("BOTTOMPADDING",(0,0),(-1,-1),7)

        ])

    )

    story.append(table)

else:

    story.append(
        Paragraph(
            "Statistics file not found.",
            normal
        )
    )

story.append(PageBreak())

# ==========================================================
# METRIQUES OFFICIELLES DU MODELE (remplace confusion matrix
# et evaluation metrics, qui necessitent un ground truth absent
# pour cette zone Sentinel-2)
# ==========================================================

story.append(
    Paragraph(
        "5. Metriques officielles du modele",
        heading
    )
)

story.append(
    Paragraph(
        """
Aucun ground truth n'est disponible pour cette zone (imagerie Sentinel-2,
hors domaine d'entrainement du modele). Les metriques ci-dessous sont
celles publiees officiellement par HOTOSM sur le jeu de validation
xBD/xView2 (imagerie sub-metrique, domaine d'entrainement du modele).
Elles mesurent la performance du modele sur son domaine d'origine,
pas sur ce site precis.
""",
        normal
    )
)

story.append(Spacer(1, 0.4*cm))

official_metrics = [

    ["Classe", "F1-score"],
    ["No Damage", "0.923"],
    ["Minor Damage", "0.580"],
    ["Major Damage", "0.662"],
    ["Destroyed", "0.889"],
    ["Macro F1", "0.7634"],
    ["F1 harmonique", "0.7348"],

]

metrics_table = Table(
    official_metrics,
    colWidths=[8*cm, 8*cm]
)

metrics_table.setStyle(

    TableStyle([

        ("BACKGROUND",(0,0),(-1,0),HexColor("#D6EAF8")),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("FONTNAME",(0,-2),(-1,-1),"Helvetica-Bold"),

        ("BACKGROUND",(0,-2),(-1,-1),HexColor("#EEF1F4")),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ])

)

story.append(metrics_table)

story.append(Spacer(1, 0.4*cm))

story.append(
    Paragraph(
        "<i>Source : fiche Hugging Face du modele "
        "hotosm/earthquake-damage-assessment-model, "
        "validation sur 6166 batiments.</i>",
        normal
    )
)

story.append(PageBreak())

# ==========================================================
# PARTIE 5

# ==========================================================
# PARTIE 5
# CONCLUSION + ANNEXES + GENERATION DU PDF
# ==========================================================

story.append(
    Paragraph(
        "6. Conclusion",
        heading
    )
)

story.append(
    Paragraph(
        """
This report presents the results obtained using a Deep Learning
pipeline for building damage assessment from Sentinel-2 satellite
imagery.

The workflow includes image preparation, tile generation,
neural network inference using a DINOv3 ONNX model,
prediction merging, damage classification and visualization.

The obtained maps provide an overview of the spatial
distribution of building damages after the disaster.
Evaluation metrics indicate the overall quality of the model
when a ground truth dataset is available.
""",
        normal
    )
)

story.append(Spacer(1,0.7*cm))

# ==========================================================
# TECHNICAL INFORMATION
# ==========================================================

story.append(
    Paragraph(
        "7. Technical Information",
        heading
    )
)

technical = [

    ["Parameter","Value"],

    ["Model","DINOv3 ONNX"],

    ["Satellite","Sentinel-2"],

    ["Input Size","512 × 512"],

    ["Framework","ONNX Runtime"],

    ["Programming Language","Python"],

    ["Generated Report",datetime.now().strftime("%d/%m/%Y %H:%M")],

    ["Output Directory",str(OUTPUT)]

]

tech_table = Table(
    technical,
    colWidths=[7*cm,9*cm]
)

tech_table.setStyle(

    TableStyle([

        ("BACKGROUND",(0,0),(-1,0),HexColor("#AED6F1")),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ])

)

story.append(tech_table)

story.append(Spacer(1,1*cm))

# ==========================================================
# GENERATED FILES
# ==========================================================

story.append(
    Paragraph(
        "Generated Outputs",
        heading
    )
)

files = [

    "✓ pre_rgb.png",

    "✓ post_rgb.png",

    "✓ damage_map.png",

    "✓ visualization.png",

    "✓ class_distribution.png",

    "✓ confusion_matrix.png",

    "✓ class_statistics.csv",

    "✓ evaluation_metrics.csv",

    "✓ damage_buildings.gpkg"

]

for f in files:

    story.append(
        Paragraph(f,normal)
    )

story.append(Spacer(1,1*cm))

# ==========================================================
# FINAL NOTE
# ==========================================================

story.append(
    Paragraph(
        """
<b>End of Report</b>
""",
        title_style
    )
)

story.append(
    Paragraph(
        """
This report was automatically generated by the Building Damage
Assessment Pipeline.
""",
        normal
    )
)

# ==========================================================
# GENERATE PDF
# ==========================================================

doc.build(story)

print("="*60)
print("REPORT GENERATED SUCCESSFULLY")
print(PDF_FILE)
print("="*60)


