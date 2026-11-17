#!/usr/bin/env python3
"""Work out the camera's lens settings (calibration) from checkerboard photos.

The ArUco position step (Week 6) needs this file, so do it once for the
camera and focus setting you will use on the robot.

  1. Take 20-30 photos of the printed checkerboard:
       python3 ../capture/capture_images.py --label calib
     Move the board around: near and far, tilted left/right/up/down,
     in the corners of the picture. Keep the whole board in view and sharp.
  2. Run:
       python3 calibrate_camera.py
     (add --square-mm 24.6 if your printed square measured 24.6 mm)

Writes camera_calibration.yaml next to this script and prints the error.
An error (RMS) under about 0.5 px is very good; over 1.0 px means some
photos are blurry or the board was not flat - remove them and run again.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]


def find_corners(paths, pattern, check_dir=None):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, size, used = [], None, []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip {path.name}: could not read")
            continue
        if size is None:
            size = img.shape[1], img.shape[0]
        elif (img.shape[1], img.shape[0]) != size:
            print(f"  skip {path.name}: different photo size {img.shape[1]}x{img.shape[0]}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not ok:
            print(f"  skip {path.name}: board not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        found.append(corners)
        used.append(path)
        if check_dir is not None:
            vis = cv2.drawChessboardCorners(img.copy(), pattern, corners, ok)
            cv2.imwrite(str(check_dir / path.name), vis)
    return found, size, used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(VISION_DIR / "dataset" / "raw" / "calib"))
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square-mm", type=float, default=25.0, help="measured size of one square")
    p.add_argument("--out", default=str(Path(__file__).with_name("camera_calibration.yaml")))
    p.add_argument("--check", action="store_true",
                   help="also save copies of the photos with the found corners drawn on")
    args = p.parse_args(argv)

    folder = Path(args.images)
    paths = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    if not paths:
        raise SystemExit(f"No photos in {folder}")
    pattern = (args.cols, args.rows)
    check_dir = None
    if args.check:
        check_dir = folder.parent / (folder.name + "_check")
        check_dir.mkdir(exist_ok=True)

    print(f"Looking for a {args.cols}x{args.rows} board in {len(paths)} photos...")
    img_pts, size, used = find_corners(paths, pattern, check_dir)
    print(f"Board found in {len(used)} of {len(paths)} photos.")
    if len(used) < 10:
        raise SystemExit("Need at least 10 good photos (20-30 is better). Take more and try again.")

    obj = np.zeros((args.rows * args.cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    obj_pts = [obj] * len(img_pts)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # error per photo, to spot bad ones
    per_image = []
    for o, ip, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        diff = ip.reshape(-1, 2) - proj.reshape(-1, 2)  # shapes differ between OpenCV versions
        per_image.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("rms_px", float(rms))
    fs.write("square_mm", float(args.square_mm))
    fs.write("board_inner_corners", np.array(pattern, dtype=np.int32))
    fs.write("photos_used", len(used))
    fs.release()

    print(f"\nPhoto size: {size[0]} x {size[1]}")
    print(f"Focal length: fx = {K[0, 0]:.1f} px, fy = {K[1, 1]:.1f} px")
    print(f"Centre: cx = {K[0, 2]:.1f} px, cy = {K[1, 2]:.1f} px")
    print(f"Overall error (RMS): {rms:.3f} px", end="  ")
    print("- very good" if rms < 0.5 else "- okay" if rms < 1.0 else "- too high, see notes in this script")
    worst = sorted(zip(per_image, used), reverse=True)[:3]
    print("Worst photos (remove these first if the error is high):")
    for err, path in worst:
        print(f"  {path.name}: {err:.3f} px")
    print(f"\nSaved {args.out}")
    if check_dir:
        print(f"Photos with corners drawn: {check_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
