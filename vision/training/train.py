#!/usr/bin/env python3
"""Train the YOLOv8n product detector (best run on Google Colab with a GPU).

  python3 training/train.py --data /content/dataset/data_checked.yaml
  python3 training/train.py --data data_checked.yaml --epochs 150 --name run2

After training it:
  - scores the model on the TEST split (or valid if there is no test split)
  - copies the best weights to vision/models/best.pt
  - writes training/results/<name>_summary.md (table for your report)
  - adds a line to training/training_log.csv
"""
import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

VISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION))
from shopvision.metrics import summary_markdown  # noqa: E402

LOG = VISION / "training" / "training_log.csv"
RESULTS = VISION / "training" / "results"
MODELS = VISION / "models"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data_checked.yaml from check_dataset.py")
    ap.add_argument("--model", default="yolov8n.pt", help="starting model (YOLOv8 nano)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30, help="stop early if no progress for this many epochs")
    ap.add_argument("--name", default=f"run_{datetime.now():%Y%m%d_%H%M}")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU")
    ap.add_argument("--notes", default="", help="what changed in this run (for the log)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    model = YOLO(args.model)
    train_kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        patience=args.patience, project=str(VISION / "runs"), name=args.name,
                        exist_ok=True, plots=True,
                        # extra variety on top of Roboflow's: colour, light and position changes
                        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=5.0, translate=0.1,
                        scale=0.4, fliplr=0.5, mosaic=1.0)
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(best, MODELS / "best.pt")
    print(f"\nBest weights copied to {MODELS / 'best.pt'}")

    split = "test" if data.get("test") else "val"
    trained = YOLO(str(best))
    metrics = trained.val(data=args.data, split=split, imgsz=args.imgsz, plots=True,
                          project=str(VISION / "runs"), name=f"{args.name}_{split}", exist_ok=True)
    md, map50 = summary_markdown(f"Training run {args.name}", metrics, names,
                                 {"Split scored": split, "Epochs": args.epochs, "Image size": args.imgsz,
                                  "Starting model": args.model, "Notes": args.notes or "-"})
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.name}_summary.md"
    out.write_text(md)
    print(md)
    print(f"Saved {out}\nPlots (confusion matrix etc.): {metrics.save_dir}")

    new = not LOG.exists()
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "run", "epochs", "imgsz", "split", "mAP50", "notes"])
        w.writerow([f"{datetime.now():%Y-%m-%d %H:%M}", args.name, args.epochs, args.imgsz, split,
                    round(map50, 4), args.notes])
    return 0


if __name__ == "__main__":
    sys.exit(main())
