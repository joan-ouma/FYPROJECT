"""Find ArUco markers in a frame and work out their 3D pose from the camera.

Needs the camera calibration (calibration/camera_calibration.yaml).
Marker frame (OpenCV convention): origin at the centre of the black square,
x to the right, y up, z out of the marker towards the camera. Units: mm.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import resolve
from .transforms import from_rvec_tvec


@dataclass
class MarkerPose:
    marker_id: int
    corners: np.ndarray       # 4x2 pixel corners (top-left, top-right, bottom-right, bottom-left)
    rvec: np.ndarray
    tvec: np.ndarray          # marker centre in camera frame, mm
    T_cam_marker: np.ndarray  # 4x4
    reprojection_px: float

    @property
    def distance_mm(self):
        return float(np.linalg.norm(self.tvec))


def load_calibration(path):
    fs = cv2.FileStorage(str(resolve(path)), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Calibration file not found: {resolve(path)} - run calibration first")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    fs.release()
    if K is None:
        raise ValueError(f"{path} has no camera_matrix")
    return K, dist, (w, h)


def get_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class MarkerFinder:
    def __init__(self, K, dist, marker_size_mm, dictionary="DICT_4X4_50"):
        self.K = np.asarray(K, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.size = float(marker_size_mm)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(get_dictionary(dictionary), params)
        s = self.size / 2.0
        # corner order used by ArUco: top-left, top-right, bottom-right, bottom-left
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    def find(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = []
        if ids is None:
            return poses
        for c, mid in zip(corners, ids.ravel()):
            img = c.reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(self.obj, img, self.K, self.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(self.obj, rvec, tvec, self.K, self.dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
            poses.append(MarkerPose(int(mid), img, rvec.ravel(), tvec.ravel(),
                                    from_rvec_tvec(rvec, tvec), err))
        poses.sort(key=lambda p: p.marker_id)
        return poses

    def project(self, point_cam_mm):
        """Pixel where a 3D point (camera frame, mm) appears in the image."""
        p = np.asarray(point_cam_mm, dtype=float).reshape(1, 3)
        px, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.ravel()

    def pixel_ray(self, u, v):
        """Unit direction (camera frame) of the ray through pixel (u, v)."""
        pts = np.array([[[u, v]]], dtype=np.float64)
        n = cv2.undistortPoints(pts, self.K, self.dist).ravel()
        d = np.array([n[0], n[1], 1.0])
        return d / np.linalg.norm(d)

    def draw(self, frame, poses, axis_mm=None):
        img = frame.copy()
        if not poses:
            return img
        cv2.aruco.drawDetectedMarkers(img, [p.corners.reshape(1, 4, 2).astype(np.float32) for p in poses],
                                      np.array([[p.marker_id] for p in poses]))
        for p in poses:
            cv2.drawFrameAxes(img, self.K, self.dist, p.rvec, p.tvec, axis_mm or self.size * 0.75)
        return img
