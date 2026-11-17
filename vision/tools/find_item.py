#!/usr/bin/env python3
"""Run the full vision step once: confirm a product and print its position for the arm.

  python3 tools/find_item.py egg
  python3 tools/find_item.py toothpaste --timeout 10 --show
  python3 tools/find_item.py egg --repeat 20      # repeat test: success rate + spread, for the report

This is exactly what the robot's mission control calls (through the ROS 2 node).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.service import VisionService  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--show", action="store_true", help="show the camera view while looking")
    args = ap.parse_args(argv)

    on_frame = None
    if args.show:
        import cv2
        from shopvision.detector import draw_detections

        def on_frame(frame, dets, poses, tracker):
            v = draw_detections(frame, dets, highlight=args.product)
            v = vision.markers.draw(v, poses)
            cv2.imshow("find_item", v)
            cv2.waitKey(1)

    vision = VisionService(source=args.source)
    results = []
    try:
        for i in range(args.repeat):
            r = vision.find_item(args.product, args.timeout, on_frame)
            results.append(r)
            print(json.dumps(r, indent=2) if args.repeat == 1 else
                  f"[{i + 1}/{args.repeat}] {r['status']:<14} {r.get('position_arm_mm', '')}  {r['time_s']} s")
    finally:
        vision.close()
    if args.repeat > 1:
        ok = [r for r in results if r["found"]]
        print(f"\nFound {len(ok)}/{len(results)} times ({100 * len(ok) / len(results):.0f}%)")
        if len(ok) >= 2:
            pts = np.array([r["position_arm_mm"] for r in ok])
            print(f"Position spread (std) x/y/z: {np.round(pts.std(axis=0), 1)} mm")
            print(f"Mean time to find: {np.mean([r['time_s'] for r in ok]):.2f} s")
    return 0 if results and results[-1]["found"] else 1


if __name__ == "__main__":
    sys.exit(main())
