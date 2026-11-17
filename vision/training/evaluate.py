#!/usr/bin/env python3
"""Score a trained model on the test photos and write a results table for the report.

  python3 training/evaluate.py --data data_checked.yaml                  # models/best.pt
  python3 training/evaluate.py --weights models/best_ncnn_model --data data_checked.yaml

Use it to check the exported (NCNN) model scores the same as best.pt.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

VISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION))
from shopvision.metrics import summary_markdown  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(VISION / "models" / "best.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default=None, help="test (default if present) or val")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    split = args.split or ("test" if data.get("test") else "val")
    model = YOLO(args.weights, task="detect")
    metrics = model.val(data=args.data, split=split, imgsz=args.imgsz, plots=True,
                        project=str(VISION / "runs"), name=f"eval_{datetime.now():%Y%m%d_%H%M}")
    md, _ = summary_markdown(f"Evaluation of {Path(args.weights).name}", metrics, names,
                             {"Split scored": split, "Image size": args.imgsz})
    out = VISION / "training" / "results" / f"eval_{datetime.now():%Y%m%d_%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"Saved {out}\nConfusion matrix and plots: {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
