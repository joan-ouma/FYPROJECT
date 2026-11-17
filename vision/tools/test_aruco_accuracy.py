#!/usr/bin/env python3
"""Test how accurately the camera measures a marker's position (target: within 15 mm).

Put one marker at a distance you have measured with a tape (from the camera lens to the
marker centre), run this, type the real distance, and it records the error. Repeat at
several distances and angles (e.g. 250, 350, 450 mm; straight on and turned 30 degrees).

  python3 tools/test_aruco_accuracy.py --marker-id 0
  python3 tools/test_aruco_accuracy.py --source photos_folder/   # test on saved photos

Each test averages 30 frames. Results are added to tools/aruco_accuracy_results.csv,
and a summary (mean and worst error) is printed at the end - ready for your report.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.aruco import MarkerFinder, load_calibration  # noqa: E402
from shopvision.camera import open_camera  # noqa: E402
from shopvision.config import load_config  # noqa: E402

RESULTS = Path(__file__).with_name("aruco_accuracy_results.csv")
TARGET_MM = 15.0


def measure(cam, finder, marker_id, frames):
    dists, xyz = [], []
    tries = 0
    while len(dists) < frames and tries < frames * 4:
        tries += 1
        f = cam.read()
        if f is None:
            break
        for p in finder.find(f):
            if p.marker_id == marker_id:
                dists.append(p.distance_mm)
                xyz.append(p.tvec)
    if not dists:
        return None
    return float(np.mean(dists)), float(np.std(dists)), np.mean(xyz, axis=0), len(dists)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--source", default=None, help="webcam number, video or image folder")
    ap.add_argument("--true-mm", type=float, default=None,
                    help="real distance; give it to run one test without questions")
    args = ap.parse_args(argv)

    cfg = load_config()
    K, dist, size = load_calibration(cfg["camera"]["calibration_file"])
    finder = MarkerFinder(K, dist, cfg["aruco"]["marker_size_mm"], cfg["aruco"]["dictionary"])
    cam = open_camera(args.source, (cfg["camera"]["width"], cfg["camera"]["height"]),
                      cfg["camera"].get("lens_position"))
    errors = []
    new_file = not RESULTS.exists()
    try:
        with open(RESULTS, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["time", "marker_id", "true_mm", "measured_mm", "error_mm",
                            "spread_mm", "x_mm", "y_mm", "z_mm", "frames"])
            while True:
                if args.true_mm is None:
                    s = input("\nReal distance in mm (Enter to finish): ").strip()
                    if not s:
                        break
                    true_mm = float(s)
                else:
                    true_mm = args.true_mm
                m = measure(cam, finder, args.marker_id, args.frames)
                if m is None:
                    print(f"Marker {args.marker_id} not seen. Check light, focus and the marker size setting.")
                    if args.true_mm is not None:
                        break
                    continue
                mean, spread, xyz, n = m
                err = mean - true_mm
                errors.append(abs(err))
                ok = "PASS" if abs(err) <= TARGET_MM else "FAIL"
                print(f"measured {mean:.1f} mm (spread {spread:.1f} mm over {n} frames) "
                      f"-> error {err:+.1f} mm  [{ok}]")
                w.writerow([datetime.now().isoformat(timespec="seconds"), args.marker_id, true_mm,
                            round(mean, 1), round(err, 1), round(spread, 1),
                            *[round(float(v), 1) for v in xyz], n])
                if args.true_mm is not None:
                    break
    finally:
        cam.close()
    if errors:
        print(f"\n{len(errors)} tests: mean error {np.mean(errors):.1f} mm, worst {max(errors):.1f} mm "
              f"(target {TARGET_MM:.0f} mm). Saved to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
