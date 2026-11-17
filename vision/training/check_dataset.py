#!/usr/bin/env python3
"""Check a dataset exported from Roboflow (format: "YOLOv8") before training.

  python3 training/check_dataset.py path/to/dataset_folder
  python3 training/check_dataset.py path/to/dataset_folder --write-yaml

It checks that:
  - the class names and their ORDER match products.txt (a wrong order = wrong labels!)
  - every label line is valid and its class number exists
  - how many times each product appears in train / valid / test
and with --write-yaml it writes data_checked.yaml with full paths (this avoids the
common "images not found" error caused by Roboflow's ../ paths).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.config import load_products  # noqa: E402

SPLITS = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PER_PRODUCT = 80   # sightings in train, below this the model will struggle


def find_split(root, names):
    for n in names:
        if (root / n / "images").is_dir():
            return root / n
    return None


def read_names(data_yaml):
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names), data


def check_split(folder, n_classes):
    imgs = [p for p in (folder / "images").iterdir() if p.suffix.lower() in IMG_EXT]
    counts, problems = Counter(), []
    no_label, background = 0, 0
    for img in imgs:
        lab = folder / "labels" / (img.stem + ".txt")
        if not lab.exists():
            no_label += 1
            continue
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            background += 1
        for i, line in enumerate(lines, 1):
            parts = line.split()
            try:
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                problems.append(f"{lab.name} line {i}: not numbers")
                continue
            if len(vals) != 4:
                problems.append(f"{lab.name} line {i}: expected 4 box numbers, got {len(vals)}"
                                f" (export as 'YOLOv8' object detection, not segmentation)")
                continue
            if not 0 <= cid < n_classes:
                problems.append(f"{lab.name} line {i}: class {cid} does not exist")
                continue
            if any(v < 0 or v > 1 for v in vals):
                problems.append(f"{lab.name} line {i}: box numbers must be between 0 and 1")
            counts[cid] += 1
    return len(imgs), counts, no_label, background, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="folder with data.yaml and train/ valid/ test/")
    ap.add_argument("--write-yaml", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.dataset).resolve()
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml in {root}")
    names, _ = read_names(data_yaml)
    products = load_products()
    ok = True

    print(f"Dataset: {root}")
    print(f"Classes in data.yaml: {names}")
    if names != products:
        ok = False
        print(f"PROBLEM: products.txt says {products}")
        if sorted(names) == sorted(products):
            print("  Same names, different ORDER. Fix the order in Roboflow or products.txt,\n"
                  "  otherwise the robot will call one product by another's name.")
        else:
            missing = set(products) - set(names)
            extra = set(names) - set(products)
            if missing:
                print(f"  Missing in dataset: {sorted(missing)}")
            if extra:
                print(f"  Not in products.txt: {sorted(extra)} (rename these classes in Roboflow)")

    found = {}
    table = []
    for split, folder_names in SPLITS.items():
        folder = find_split(root, folder_names)
        if folder is None:
            print(f"\n{split}: not found")
            if split in ("train", "val"):
                ok = False
            continue
        found[split] = folder
        n_img, counts, no_label, bg, problems = check_split(folder, len(names))
        print(f"\n{split} ({folder.name}/): {n_img} images, {no_label} without a label file, "
              f"{bg} with no products (background)")
        for msg in problems[:10]:
            print("  PROBLEM:", msg)
        if problems:
            ok = False
            if len(problems) > 10:
                print(f"  ... and {len(problems) - 10} more")
        table.append((split, counts))

    print(f"\n{'product':<14}" + "".join(f"{s:>8}" for s, _ in table))
    for cid, name in enumerate(names):
        row = "".join(f"{c.get(cid, 0):>8}" for _, c in table)
        warn = ""
        train_counts = dict(table).get("train", Counter())
        if train_counts.get(cid, 0) < MIN_PER_PRODUCT:
            warn = f"   <- fewer than {MIN_PER_PRODUCT} in train"
        print(f"{name:<14}{row}{warn}")
    if "test" not in found:
        print("\nNote: no test split. Keep some of YOUR OWN camera photos as a test set, so the"
              " score in the report is honest.")

    if args.write_yaml:
        out = root / "data_checked.yaml"
        data = {"path": str(root), "train": str(found["train"] / "images"),
                "val": str(found["val"] / "images"), "names": {i: n for i, n in enumerate(names)}}
        if "test" in found:
            data["test"] = str(found["test"] / "images")
        out.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"\nWrote {out} - use this file for training.")

    print("\nRESULT:", "OK - ready to train" if ok else "fix the problems above first")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
