"""Small helpers for 4x4 transforms (rotation + translation, in mm).

T_a_b means "pose of frame b written in frame a": a point p_b in frame b
is p_a = T_a_b @ [p_b, 1].
"""
import cv2
import numpy as np


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_T(R, tvec)


def to_rvec_tvec(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.ravel(), T[:3, 3].copy()


def invert(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def apply(T, p):
    p = np.asarray(p, dtype=float).ravel()
    return (T @ np.append(p, 1.0))[:3]


def rpy_deg_to_R(roll, pitch, yaw):
    """Roll about x, then pitch about y, then yaw about z (R = Rz @ Ry @ Rx)."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R):
    """Inverse of rpy_deg_to_R."""
    R = np.asarray(R, dtype=float)
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def load_camera_to_arm(data):
    """data = the dict from config/camera_to_arm.yaml. Returns T_arm_cam."""
    R = rpy_deg_to_R(*data["rotation_rpy_deg"])
    return make_T(R, data["translation_mm"])
