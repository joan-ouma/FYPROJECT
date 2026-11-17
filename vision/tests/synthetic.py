"""Make fake camera pictures with ArUco markers at known 3D positions (for the tests)."""
import cv2
import numpy as np

from shopvision.aruco import get_dictionary
from shopvision.transforms import apply

W, H = 1280, 720
K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
DIST = np.zeros(5)


def marker_texture(marker_id, px_per_mm=8, size_mm=40.0, border_mm=10.0, dictionary="DICT_4X4_50"):
    side = int(size_mm * px_per_mm)
    m = cv2.aruco.generateImageMarker(get_dictionary(dictionary), marker_id, side)
    b = int(border_mm * px_per_mm)
    tex = np.full((side + 2 * b, side + 2 * b), 255, np.uint8)
    tex[b:b + side, b:b + side] = m
    half = size_mm / 2 + border_mm
    return tex, half


def render(markers, background=110, K=K, size_mm=40.0, noise=2.0, seed=0):
    """markers = [(marker_id, T_cam_marker)]. Returns a BGR image."""
    img = np.full((H, W), background, np.float64)
    for mid, T in markers:
        tex, half = marker_texture(mid, size_mm=size_mm)
        h, w = tex.shape
        corners3d = [(-half, half, 0), (half, half, 0), (half, -half, 0), (-half, -half, 0)]
        cam = np.array([apply(T, c) for c in corners3d])
        px = (K @ cam.T).T
        px = px[:, :2] / px[:, 2:3]
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(px))
        warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tex, 255), Hm, (W, H), borderValue=0) > 128
        img[mask] = warped[mask]
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def project(point_cam, K=K):
    p = K @ np.asarray(point_cam, float)
    return p[:2] / p[2]


def write_calibration(path, K=K, dist=DIST, size=(W, H)):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", np.asarray(dist, float).reshape(1, -1))
    fs.release()
