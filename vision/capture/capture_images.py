#!/usr/bin/env python3
"""Take photos for the product dataset or for camera calibration.

On the Raspberry Pi it uses the Pi camera (Picamera2).
On a laptop it uses a USB webcam through OpenCV, which is handy for testing the script.

Photos are saved to  vision/dataset/raw/<label>/<label>_<date>_<time>.jpg

Plan: about 150 photos in total, with SEVERAL products in each photo. Use the --label to
name the session (e.g. mixed_morning, mixed_night), or a product name for photos of one product.

Examples
  # Preview window: press SPACE to save a photo, Q to quit
  python3 capture_images.py --label mixed_morning

  # No window (e.g. over SSH): save one photo every 1.5 s until 50 are saved
  python3 capture_images.py --label mixed_morning --every 1.5 --count 50

  # Calibration photos with the focus locked (Camera Module 3 only)
  python3 capture_images.py --label calib --lens-position 2.0

  # Test on a laptop with the built-in webcam, or with a video file
  python3 capture_images.py --label test --source 0
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

VISION_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = VISION_DIR / "dataset" / "raw"
TARGET_TOTAL = 150   # photos in total, several products in each


class PiCamera:
    """Raspberry Pi camera through Picamera2 (Raspberry Pi OS Bookworm)."""

    def __init__(self, size, lens_position=None):
        from picamera2 import Picamera2  # only available on the Pi

        self.cam = Picamera2()
        # "RGB888" gives arrays in B, G, R order, which is what OpenCV expects.
        config = self.cam.create_preview_configuration(main={"size": size, "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()
        if lens_position is not None:
            try:
                from libcamera import controls
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                       "LensPosition": float(lens_position)})
                print(f"Focus locked at lens position {lens_position}")
            except Exception as err:  # fixed-focus cameras have no autofocus controls
                print("Note: could not lock the focus (this camera may have fixed focus):", err)
        time.sleep(1.0)  # let exposure and white balance settle
        self.name = "Pi camera"

    def read(self):
        return self.cam.capture_array()

    def close(self):
        self.cam.stop()


class OpenCVCamera:
    """USB webcam (give a number like 0) or a video file (give its path)."""

    def __init__(self, source, size):
        src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera or video: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.name = f"OpenCV source {source}"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def open_camera(args):
    size = (args.width, args.height)
    if args.source is not None:
        return OpenCVCamera(args.source, size)
    try:
        return PiCamera(size, args.lens_position)
    except ImportError:
        print("Picamera2 not found, so using webcam 0 instead (fine for testing on a laptop).")
        return OpenCVCamera(0, size)


def save(frame, out_dir, label, quality):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = out_dir / f"{label}_{stamp}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise SystemExit(f"Could not write {path}")
    return path


def count_existing(out_dir):
    return len(list(out_dir.glob("*.jpg")))


def run_preview(cam, out_dir, label, quality, limit):
    saved = 0
    window = "capture - SPACE: save   Q: quit"
    while True:
        frame = cam.read()
        if frame is None:
            print("Camera stopped giving frames.")
            break
        total = count_existing(out_dir)
        view = frame.copy()
        scale = 960 / view.shape[1] if view.shape[1] > 960 else 1.0
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        text = f"{label}: {total} photos in this folder   saved now: {saved}"
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("s")):
            path = save(frame, out_dir, label, quality)
            saved += 1
            print("saved", path.name)
            if limit and saved >= limit:
                break
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return saved


def run_interval(cam, out_dir, label, quality, every, limit):
    saved = 0
    print(f"Saving one photo every {every} s. Press Ctrl+C to stop.")
    try:
        while not limit or saved < limit:
            start = time.time()
            frame = cam.read()
            if frame is None:
                print("Camera stopped giving frames.")
                break
            path = save(frame, out_dir, label, quality)
            saved += 1
            print(f"[{saved}{'/' + str(limit) if limit else ''}] saved {path.name} "
                  f"- move the product or change the light now")
            time.sleep(max(0.0, every - (time.time() - start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True,
                   help="session name (e.g. mixed_morning), a product name, or 'calib' for calibration")
    p.add_argument("--every", type=float, default=None,
                   help="seconds between photos; turns off the preview window")
    p.add_argument("--count", type=int, default=0, help="stop after this many photos (0 = no limit)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument("--lens-position", type=float, default=None,
                   help="lock the focus (Camera Module 3). Use the same value for calibration and data")
    p.add_argument("--source", default=None,
                   help="use OpenCV instead of the Pi camera: a webcam number (0) or a video file")
    p.add_argument("--out", default=str(RAW_DIR), help="folder to save photos in")
    args = p.parse_args(argv)

    label = args.label.strip().lower().replace(" ", "_")
    known = load_products()
    if known and label not in known and label not in ("calib", "test") and not label.startswith("mixed"):
        print(f"Note: '{label}' is not a product name; photos will be saved as a session called '{label}'.")

    out_dir = Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = open_camera(args)
    print(f"Using {cam.name}, saving to {out_dir}")
    try:
        if args.every:
            saved = run_interval(cam, out_dir, label, args.quality, args.every, args.count)
        else:
            saved = run_preview(cam, out_dir, label, args.quality, args.count)
    finally:
        cam.close()
    print(f"Done: {saved} new photos. '{label}' now has {count_existing(out_dir)} photos.")
    return 0


def load_products():
    path = VISION_DIR / "products.txt"
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


if __name__ == "__main__":
    sys.exit(main())
