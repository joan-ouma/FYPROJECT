#!/usr/bin/env python3
"""Show how many photos you have taken, per session folder (target: about 150 in total).

  python3 capture/dataset_status.py

Photos usually show several products, so the number of times each PRODUCT appears is
only known after labelling: run training/check_dataset.py on the Roboflow export for that.
"""
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = VISION_DIR / "dataset" / "raw"
TARGET_TOTAL = 150


def main():
    if not RAW_DIR.exists():
        print(f"No photos yet ({RAW_DIR} does not exist).")
        return
    total = 0
    print(f"{'session folder':<24}{'photos':>8}")
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        n = len(list(folder.glob("*.jpg")))
        if folder.name == "calib":
            continue
        total += n
        print(f"{folder.name:<24}{n:>8}")
    bar = "#" * min(30, total * 30 // TARGET_TOTAL)
    print(f"{'total':<24}{total:>8}   [{bar:<30}] {min(100, total * 100 // TARGET_TOTAL)}% of {TARGET_TOTAL}")
    calib = RAW_DIR / "calib"
    if calib.exists():
        print(f"\ncalibration photos: {len(list(calib.glob('*.jpg')))} (aim for 20-30)")
    print("\nAfter labelling in Roboflow, check how often each product appears with:\n"
          "  python3 training/check_dataset.py <dataset folder>   (aim for 80+ per product)")


if __name__ == "__main__":
    main()
