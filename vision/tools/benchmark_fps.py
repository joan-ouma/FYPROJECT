#!/usr/bin/env python3
"""Measure how many frames per second the model runs at (target: 5 or more on the Pi 5).

  python3 tools/benchmark_fps.py                                  # Pi camera, 100 frames
  python3 tools/benchmark_fps.py --weights models/best_ncnn_model
  python3 tools/benchmark_fps.py --source photos_folder/ --frames 50

It reports the model time alone and the full loop (camera + model + markers),
and appends the result to tools/fps_results.csv for your report.
"""
import argparse
import csv
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.camera import open_camera, ImageFolder  # noqa: E402
from shopvision.config import load_config  # noqa: E402
from shopvision.detector import Detector  # noqa: E402

RESULTS = Path(__file__).with_name("fps_results.csv")
TARGET = 5.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--with-markers", action="store_true", help="include ArUco in the full loop")
    args = ap.parse_args(argv)

    cfg = load_config()
    weights = args.weights or cfg["model"]["weights"]
    imgsz = args.imgsz or cfg["model"].get("image_size", 640)
    det = Detector(weights, imgsz, cfg["model"].get("min_confidence", 0.25))
    finder = None
    if args.with_markers:
        from shopvision.aruco import MarkerFinder, load_calibration
        K, dist, _ = load_calibration(cfg["camera"]["calibration_file"])
        finder = MarkerFinder(K, dist, cfg["aruco"]["marker_size_mm"], cfg["aruco"]["dictionary"])
    cam = open_camera(args.source, (cfg["camera"]["width"], cfg["camera"]["height"]),
                      cfg["camera"].get("lens_position"))
    if isinstance(cam, ImageFolder):
        cam.loop = True
    model_t, loop_t = [], []
    try:
        for i in range(args.warmup + args.frames):
            t0 = time.perf_counter()
            frame = cam.read()
            if frame is None:
                break
            t1 = time.perf_counter()
            det.detect(frame)
            t2 = time.perf_counter()
            if finder:
                finder.find(frame)
            t3 = time.perf_counter()
            if i >= args.warmup:
                model_t.append(t2 - t1)
                loop_t.append(t3 - t0)
    finally:
        cam.close()
    if not model_t:
        raise SystemExit("No frames measured.")
    model_fps = 1.0 / np.mean(model_t)
    loop_fps = 1.0 / np.mean(loop_t)
    print(f"Model: {Path(weights).name} at {imgsz} px, {len(model_t)} frames on {platform.machine()}")
    print(f"  model only : {model_fps:.1f} FPS ({1000 * np.mean(model_t):.0f} ms per frame)")
    print(f"  full loop  : {loop_fps:.1f} FPS  -> {'PASS' if loop_fps >= TARGET else 'FAIL'} "
          f"(target {TARGET:.0f} FPS)")
    if loop_fps < TARGET:
        print("  Tip: export at a smaller size, e.g. training/export_model.py --imgsz 416")
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "machine", "weights", "imgsz", "frames", "model_fps", "loop_fps"])
        w.writerow([f"{datetime.now():%Y-%m-%d %H:%M}", platform.machine(), Path(weights).name, imgsz,
                    len(model_t), round(model_fps, 2), round(loop_fps, 2)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
