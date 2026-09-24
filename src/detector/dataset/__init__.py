"""Dataset preparation for the ProjektCiezary weightlifting-plates export (Roboflow v10).

Three stages, each usable on its own:

- ``manifest`` — read a Roboflow COCO export, fingerprint every image, apply the exclusion rules.
- ``split`` — assign kept images to train/val/test by capture date and session (pure, no images).
- ``prepare`` — download the export, and write the cleaned, re-split COCO dataset for training.

The committed ``data/manifest.csv`` is the output of the first two stages. Everything downstream
reads it, so the split can be checked in CI without the images.
"""

CLASSES = (
    "0.5kg",
    "1kg",
    "1.5kg",
    "2kg",
    "2.5kg",
    "5kg",
    "10kg",
    "15kg",
    "20kg",
    "25kg",
    "zacisk",
)
PLATE_CLASSES = CLASSES[:-1]
