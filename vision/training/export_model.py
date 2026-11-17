#!/usr/bin/env python3
"""Convert best.pt into a fast format for the Raspberry Pi.

  python3 training/export_model.py                     # NCNN (fastest on the Pi)
  python3 training/export_model.py --format onnx
  python3 training/export_model.py --imgsz 416         # smaller = faster, a bit less accurate

The result goes to vision/models/ (e.g. models/best_ncnn_model/). Copy that folder to the
Pi and set model.weights in config/vision_config.yaml to it.
If 640 gives fewer than 5 frames per second on the Pi, export at 480 or 416 and run
tools/benchmark_fps.py again (and check the score with evaluate.py --imgsz 416).
"""
import argparse
import shutil
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(VISION / "models" / "best.pt"))
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "openvino", "tflite"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    weights = Path(args.weights)
    out = Path(YOLO(str(weights)).export(format=args.format, imgsz=args.imgsz))
    target = VISION / "models" / out.name
    if out.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(out), str(target))
    print(f"\nExported to {target}")
    print(f"Set this in config/vision_config.yaml:\n  model:\n    weights: models/{target.name}\n"
          f"    image_size: {args.imgsz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
