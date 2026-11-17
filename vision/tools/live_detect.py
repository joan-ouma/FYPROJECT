#!/usr/bin/env python3
"""Live view: boxes around products, shelf markers, frames per second, and the confirmation rule.

  python3 tools/live_detect.py                      # Pi camera, all products
  python3 tools/live_detect.py --target egg         # also shows the 3-frame confirmation for egg
  python3 tools/live_detect.py --no-markers         # skip ArUco (if not calibrated yet)
  python3 tools/live_detect.py --source 0           # laptop webcam (with models/best.pt)
  python3 tools/live_detect.py --save out.mp4       # also record a video for your presentation

Keys: Q quits, S saves a screenshot.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.camera import open_camera  # noqa: E402
from shopvision.config import load_config  # noqa: E402
from shopvision.confirm import ConfirmTracker  # noqa: E402
from shopvision.detector import Detector, draw_detections  # noqa: E402


def put(img, text, y, colour=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--save", default=None, help="record to this .mp4 file")
    args = ap.parse_args(argv)

    cfg = load_config()
    det = Detector(args.weights or cfg["model"]["weights"], cfg["model"].get("image_size", 640),
                   cfg["model"].get("min_confidence", 0.25))
    finder = None
    if not args.no_markers:
        try:
            from shopvision.aruco import MarkerFinder, load_calibration
            K, dist, _ = load_calibration(cfg["camera"]["calibration_file"])
            finder = MarkerFinder(K, dist, cfg["aruco"]["marker_size_mm"], cfg["aruco"]["dictionary"])
        except FileNotFoundError as err:
            print(f"{err}\nCarrying on without markers.")
    tracker = ConfirmTracker(args.target, cfg["confirm"]["min_confidence"],
                             cfg["confirm"]["frames_in_a_row"]) if args.target else None
    cam = open_camera(args.source, (cfg["camera"]["width"], cfg["camera"]["height"]),
                      cfg["camera"].get("lens_position"))
    writer = None
    fps, last = 0.0, time.time()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            dets = det.detect(frame)
            view = draw_detections(frame, dets, highlight=args.target)
            if finder:
                poses = finder.find(frame)
                view = finder.draw(view, poses)
                for i, p in enumerate(poses[:4]):
                    put(view, f"marker {p.marker_id}: {p.distance_mm:.0f} mm", 120 + 28 * i, (0, 255, 255))
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            put(view, f"{fps:.1f} FPS   {len(dets)} products", 30)
            if tracker:
                done = tracker.update(dets)
                put(view, f"{args.target}: {'CONFIRMED' if done else f'{tracker.streak}/{tracker.frames_in_a_row}'}",
                    60, (0, 255, 0) if done else (0, 200, 255))
            if args.save:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(view)
            cv2.imshow("live detection - Q quits, S saves", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(name, view)
                print("saved", name)
    finally:
        cam.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
