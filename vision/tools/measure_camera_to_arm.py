#!/usr/bin/env python3
"""Work out where the camera is compared with the arm base, and save config/camera_to_arm.yaml.

Agree the arm frame with Joy first: origin at the centre of the arm base at table level,
x forward (towards the shelf), y left, z up, in mm.

Then:
  1. Lay one marker FLAT on the table, printed side up, with its centre at a spot you
     measure in the arm frame (e.g. 200 mm straight in front of the arm base: 200, 0, 0).
  2. Turn it so the marker's TOP edge (the edge farthest from the printed "ID" label)
     points FORWARD (+x of the arm). Then the marker's x points to the arm's RIGHT (-y)
     and the marker's z points UP.
  3. Run:
       python3 tools/measure_camera_to_arm.py --marker-id 0 --marker-at 200 0 0

If you placed the marker another way, give its rotation in the arm frame with
--marker-rpy (roll pitch yaw in degrees). The default (0 0 -90) matches step 2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.aruco import MarkerFinder, load_calibration  # noqa: E402
from shopvision.camera import open_camera  # noqa: E402
from shopvision.config import load_config, resolve  # noqa: E402
from shopvision.transforms import (R_to_rpy_deg, invert, make_T, rpy_deg_to_R)  # noqa: E402


def average_pose(poses):
    """Average several 4x4 poses (rotation averaged through SVD)."""
    t = np.mean([p[:3, 3] for p in poses], axis=0)
    M = np.sum([p[:3, :3] for p in poses], axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return make_T(R, t)


def compute_T_arm_cam(T_cam_marker, marker_at_mm, marker_rpy_deg):
    T_arm_marker = make_T(rpy_deg_to_R(*marker_rpy_deg), marker_at_mm)
    return T_arm_marker @ invert(T_cam_marker)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--marker-at", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="marker centre in the arm frame, mm")
    ap.add_argument("--marker-rpy", type=float, nargs=3, default=[0.0, 0.0, -90.0],
                    metavar=("ROLL", "PITCH", "YAW"), help="marker rotation in the arm frame, degrees")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--source", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the result but do not save it")
    args = ap.parse_args(argv)

    cfg = load_config()
    K, dist, _ = load_calibration(cfg["camera"]["calibration_file"])
    finder = MarkerFinder(K, dist, cfg["aruco"]["marker_size_mm"], cfg["aruco"]["dictionary"])
    cam = open_camera(args.source, (cfg["camera"]["width"], cfg["camera"]["height"]),
                      cfg["camera"].get("lens_position"))
    poses = []
    try:
        for _ in range(args.frames * 4):
            f = cam.read()
            if f is None:
                break
            for p in finder.find(f):
                if p.marker_id == args.marker_id:
                    poses.append(p.T_cam_marker)
            if len(poses) >= args.frames:
                break
    finally:
        cam.close()
    if len(poses) < 5:
        raise SystemExit(f"Marker {args.marker_id} seen in only {len(poses)} frames - need at least 5.")

    T_cam_marker = average_pose(poses)
    T_arm_cam = compute_T_arm_cam(T_cam_marker, args.marker_at, args.marker_rpy)
    t = T_arm_cam[:3, 3]
    rpy = R_to_rpy_deg(T_arm_cam[:3, :3])
    print(f"Used {len(poses)} frames.")
    print(f"Camera position in arm frame: x={t[0]:.1f}  y={t[1]:.1f}  z={t[2]:.1f} mm")
    print(f"Camera rotation (roll, pitch, yaw): {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f} deg")
    print("Sanity check: measure the camera position with a tape. It should match within ~1 cm.")
    if args.dry_run:
        return 0
    out = resolve(cfg["locator"]["camera_to_arm_file"])
    data = {"translation_mm": [round(float(v), 2) for v in t],
            "rotation_rpy_deg": [round(float(v), 3) for v in rpy],
            "measured": True,
            "marker_id": args.marker_id,
            "marker_at_mm": list(args.marker_at),
            "marker_rpy_deg": list(args.marker_rpy)}
    header = ("# Written by tools/measure_camera_to_arm.py - camera pose in the arm frame.\n"
              "# Arm frame: origin at arm base centre, x forward, y left, z up (mm).\n")
    out.write_text(header + yaml.safe_dump(data, sort_keys=False))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
