---
base_model: kshitijrajsharma/dinov3
datasets:
  - kshitijrajsharma/xview2-xbd
license: cc-by-nc-sa-4.0
tags:
  - disaster-response
  - damage-assessment
  - remote-sensing
  - dinov3
  - building-damage
---

# kshitijrajsharma/dinov3-damage-assessment

Building-level disaster damage assessment. A frozen DINOv3 ViT-L/16 satellite backbone with a
trainable UperNet decoder and two heads (localization + ordinal 4-class damage). Given building
footprints and post-disaster imagery (optionally pre-disaster imagery), it assigns each building a
damage level on the xBD Joint Damage Scale with a calibrated confidence.

## Damage classes

`no-damage`, `minor-damage`, `major-damage`, `destroyed`.

## Object-level metrics (val split, 6166 buildings)

| class | F1 |
|---|---|
| no-damage | 0.923 |
| minor-damage | 0.5797 |
| major-damage | 0.6623 |
| destroyed | 0.8885 |

Macro F1 0.7634 | harmonic damage F1 0.7348.

The damage F1 is computed on building pixels only (background excluded). Numbers are
in-distribution (the xView2 benchmark splits by tile, so the same events appear in train and
test); cross-event generalisation to a fully unseen disaster is harder for the subtle
minor/major classes.

## Inputs and outputs

- Input: building footprints (GeoJSON) + post-disaster RGB GeoTIFF, optionally a pre-disaster
  GeoTIFF aligned to the post grid. Footprints must overlay the post image correctly.
- Output: the footprints annotated with `damage_class`, `damage`, `confidence`, `review`, and
  per-class probabilities.

## Files

- `model.onnx`: self-contained inference graph (post, pre -> damage logits).
- `model.ckpt`: Lightning checkpoint for evaluation or further training.
- `config.yaml`: training configuration.
- `calibration.json`: confidence temperature (1.1719).

## Confidence

Apply `softmax(logits / 1.1719)` for calibrated probabilities. Buildings below the
confidence threshold are flagged for human review.

## Backbone

DINOv3 ViT-L/16 (`sat493m`), frozen. Decoder, fusion, and heads are the only trained parameters
(~24M). Trained on xBD (xView2), license CC BY-NC-SA 4.0; this model inherits the non-commercial
share-alike terms.
