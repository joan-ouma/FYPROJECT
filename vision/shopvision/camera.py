"""Picture sources: the Pi camera, a USB webcam, a video file or a folder of images.

All of them give frames as OpenCV BGR arrays through .read(), and None when finished.
"""
import time
from pathlib import Path

import cv2
import numpy as np


class PiCamera:
    """Raspberry Pi camera through Picamera2 (Raspberry Pi OS Bookworm)."""

    def __init__(self, size=(1280, 720), lens_position=None):
        from picamera2 import Picamera2  # only available on the Pi

        self.cam = Picamera2()
        # "RGB888" gives arrays in B, G, R order, which is what OpenCV expects.
        config = self.cam.create_preview_configuration(main={"size": tuple(size), "format": "RGB888"})
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

    def __init__(self, source=0, size=(1280, 720)):
        src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera or video: {source}")
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.name = f"OpenCV source {source}"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class ImageFolder:
    """Plays the .jpg/.png images in a folder as if they were camera frames (for testing)."""

    def __init__(self, folder, loop=False):
        self.paths = sorted([p for p in Path(folder).iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not self.paths:
            raise RuntimeError(f"No images in {folder}")
        self.i = 0
        self.loop = loop
        self.name = f"images in {folder}"

    def read(self):
        if self.i >= len(self.paths):
            if not self.loop:
                return None
            self.i = 0
        img = cv2.imread(str(self.paths[self.i]))
        self.i += 1
        return img

    def close(self):
        pass


class StaticFrames:
    """Returns the given frames in turn (used by the tests)."""

    def __init__(self, frames, loop=True):
        self.frames = [np.asarray(f) for f in frames]
        self.i = 0
        self.loop = loop
        self.name = "static frames"

    def read(self):
        if self.i >= len(self.frames):
            if not self.loop:
                return None
            self.i = 0
        f = self.frames[self.i]
        self.i += 1
        return f.copy()

    def close(self):
        pass


def open_camera(source=None, size=(1280, 720), lens_position=None):
    """source=None: Pi camera if available, else webcam 0.
    A number = webcam, a video file path, or a folder of images."""
    if source is not None:
        p = Path(str(source))
        if p.is_dir():
            return ImageFolder(p)
        return OpenCVCamera(source, size)
    try:
        return PiCamera(size, lens_position)
    except ImportError:
        print("Picamera2 not found, so using webcam 0 instead (fine for testing on a laptop).")
        return OpenCVCamera(0, size)
