# ==========================================================
# 04_run_model.py
# VERSION AVEC REPRISE — saute automatiquement les tuiles deja
# traitees, pour reprendre exactement ou le pipeline s'etait
# arrete en cas d'interruption. Garde CoreML/threads.
# ==========================================================

import os
import glob
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_site,
    output_dir,
    MODEL_PATH
)

SITE = get_site()

TENSOR_DIR = output_dir() / "tensors"
PRED_DIR = output_dir() / "predictions"

PRED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Chargement du modèle ONNX...")
print("=" * 60)

n_cores = os.cpu_count() or 4
n_threads = min(3, max(1, n_cores // 2))

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
options.intra_op_num_threads = n_threads
options.inter_op_num_threads = 1

print(f"Coeurs CPU disponibles : {n_cores} (utilisation de {n_threads} threads)")

available = ort.get_available_providers()
print("Providers disponibles :", available)

session = None

if "CoreMLExecutionProvider" in available:
    try:
        print("-> Tentative d'utilisation de CoreML...")
        session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        print("-> CoreML initialise avec succes")
    except Exception as e:
        print(f"-> Echec de l'initialisation CoreML : {e}")
        print("-> Repli automatique sur CPU")
        session = None

if session is None:
    print("-> Utilisation du CPU")
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=options,
        providers=["CPUExecutionProvider"],
    )

print("Modèle chargé avec succès.")

print("\nEntrées du modèle :")
for inp in session.get_inputs():
    print(f" - {inp.name} : {inp.shape}")

print("\nSorties du modèle :")
for out in session.get_outputs():
    print(f" - {out.name} : {out.shape}")

INPUT_PRE = "pre"
INPUT_POST = "post"
OUTPUT_NAME = "damage_logits"


def list_tensors():

    pre_files = sorted(
        glob.glob(str(TENSOR_DIR / "pre" / "pre_*.npy"))
    )

    post_files = sorted(
        glob.glob(str(TENSOR_DIR / "post" / "post_*.npy"))
    )

    if len(pre_files) == 0:
        raise FileNotFoundError(
            f"Aucun tenseur trouvé dans {TENSOR_DIR}"
        )

    if len(pre_files) != len(post_files):
        raise ValueError(
            "Le nombre de tenseurs PRE et POST est différent."
        )

    return pre_files, post_files


def load_pair(pre_file, post_file):

    pre = np.load(pre_file).astype(np.float32)
    post = np.load(post_file).astype(np.float32)

    return pre, post


def predict(pre_tensor, post_tensor):

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_PRE: pre_tensor,
            INPUT_POST: post_tensor
        }
    )

    return outputs[0]


def save_prediction(logits, index):

    filename = f"prediction_{index:06d}.npy"

    np.save(
        PRED_DIR / filename,
        logits.astype(np.float32)
    )


def prediction_exists(index):
    filename = f"prediction_{index:06d}.npy"
    return (PRED_DIR / filename).exists()


def run_all_predictions():

    pre_files, post_files = list_tensors()

    total = len(pre_files)

    print(f"\n{total} tuiles à traiter.\n")

    already_done = sum(1 for i in range(total) if prediction_exists(i))
    if already_done > 0:
        print(f"{already_done}/{total} tuiles deja traitees lors d'un "
              f"lancement precedent — elles seront sautees.\n")

    start = time.time()
    processed_this_run = 0

    for i, (pre_file, post_file) in enumerate(zip(pre_files, post_files), start=1):

        if prediction_exists(i - 1):
            continue

        pre_tensor, post_tensor = load_pair(pre_file, post_file)

        logits = predict(pre_tensor, post_tensor)

        save_prediction(logits, i - 1)
        processed_this_run += 1

        elapsed = time.time() - start
        avg = elapsed / processed_this_run
        remaining_tiles = total - already_done - processed_this_run
        remaining = avg * remaining_tiles

        if processed_this_run % 5 == 0 or i == total:
            print(
                f"[{i}/{total}] "
                f"{os.path.basename(pre_file)} | "
                f"Écoulé : {elapsed:.1f}s | "
                f"Restant : {remaining:.1f}s"
            )

    print("\n" + "=" * 60)
    print("Toutes les prédictions ont été générées.")
    print("=" * 60)


def main():

    print("=" * 60)
    print("04_run_model.py")
    print("=" * 60)

    print(f"Site : {SITE}")
    print(f"Modèle : {MODEL_PATH}\n")

    run_all_predictions()

    print("\n" + "=" * 60)
    print("04_run_model.py terminé avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
