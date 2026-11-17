"""Tests for transforms, ArUco pose, the item locator and the camera-to-arm tool."""
import numpy as np
import pytest

from shopvision.aruco import MarkerFinder
from shopvision.detector import Detection
from shopvision.locator import ItemLocator
from shopvision.transforms import (R_to_rpy_deg, apply, invert, load_camera_to_arm, make_T,
                                   rpy_deg_to_R)
from synthetic import DIST, K, project, render

SLOTS = {0: {"name": "top_left", "product": "egg", "item_offset_mm": [0.0, 45.0, -30.0]},
         1: {"name": "top_middle", "product": "tea_leaves", "item_offset_mm": [10.0, 60.0, -35.0]}}


def shelf_marker(x, y, z, yaw_deg=0.0, tilt_deg=0.0):
    """A marker facing the camera (marker z towards the camera), turned by yaw and tilt."""
    R_face = np.diag([1.0, -1.0, -1.0])  # marker x = cam x, marker y = cam -y (up), z towards camera
    R = rpy_deg_to_R(tilt_deg, yaw_deg, 0.0) @ R_face
    return make_T(R, [x, y, z])


def test_rpy_round_trip():
    for rpy in ([10, 20, 30], [-90, 0, -90], [5, -40, 170]):
        assert np.allclose(R_to_rpy_deg(rpy_deg_to_R(*rpy)), rpy, atol=1e-6)


def test_invert():
    T = make_T(rpy_deg_to_R(12, -30, 45), [10, -20, 300])
    assert np.allclose(T @ invert(T), np.eye(4))


def test_default_camera_to_arm_points_camera_forward():
    T = load_camera_to_arm({"translation_mm": [-60, 0, 250], "rotation_rpy_deg": [-90, 0, -90]})
    view_dir_arm = T[:3, :3] @ np.array([0, 0, 1.0])      # camera z
    right_arm = T[:3, :3] @ np.array([1.0, 0, 0])         # camera x (image right)
    down_arm = T[:3, :3] @ np.array([0, 1.0, 0])          # camera y (image down)
    assert np.allclose(view_dir_arm, [1, 0, 0], atol=1e-9)   # looks forward
    assert np.allclose(right_arm, [0, -1, 0], atol=1e-9)     # image right = arm right
    assert np.allclose(down_arm, [0, 0, -1], atol=1e-9)      # image down = arm down


@pytest.mark.parametrize("pose", [(0, 0, 350, 0, 0), (-80, 40, 300, 25, 0), (60, -30, 450, -20, 15)])
def test_marker_pose_accuracy(pose):
    T_true = shelf_marker(*pose)
    img = render([(0, T_true)])
    finder = MarkerFinder(K, DIST, 40.0)
    found = finder.find(img)
    assert [p.marker_id for p in found] == [0]
    err = np.linalg.norm(found[0].tvec - T_true[:3, 3])
    assert err < 5.0, f"marker position error {err:.2f} mm"   # proposal target is 15 mm


def test_locator_marker_only_and_refined():
    T_m = shelf_marker(-40, 60, 380, yaw_deg=10)
    img = render([(0, T_m)])
    finder = MarkerFinder(K, DIST, 40.0)
    poses = finder.find(img)
    loc = ItemLocator(finder, SLOTS, np.eye(4))

    true_item = apply(T_m, SLOTS[0]["item_offset_mm"])
    r = loc.locate("egg", poses, [])
    assert not r.refined_by_detection
    assert np.linalg.norm(r.position_cam_mm - true_item) < 6.0

    # the egg sits 20 mm to the right of the slot centre; the detection box shows it there
    shifted = apply(T_m, np.array(SLOTS[0]["item_offset_mm"]) + [20, 0, 0])
    u, v = project(shifted)
    det = Detection("egg", 0.9, (u - 30, v - 40, u + 30, v + 40), 0)
    r2 = loc.locate("egg", poses, [det])
    assert r2.refined_by_detection
    assert np.linalg.norm(r2.position_cam_mm - shifted) < 6.0


def test_locator_ignores_far_or_wrong_detection():
    T_m = shelf_marker(0, 60, 380)
    finder = MarkerFinder(K, DIST, 40.0)
    poses = finder.find(render([(0, T_m)]))
    loc = ItemLocator(finder, SLOTS, np.eye(4), max_match_distance_px=80)
    far = Detection("egg", 0.95, (50, 50, 90, 90), 0)          # far away in the picture
    other = Detection("matchbox", 0.95, (600, 300, 680, 380), 2)
    r = loc.locate("egg", poses, [far, other])
    assert not r.refined_by_detection


def test_locator_no_marker_returns_none_and_unknown_product_raises():
    finder = MarkerFinder(K, DIST, 40.0)
    loc = ItemLocator(finder, SLOTS, np.eye(4))
    assert loc.locate("egg", [], []) is None
    with pytest.raises(KeyError):
        loc.locate("bread", [], [])


def test_locator_converts_to_arm_frame():
    T_m = shelf_marker(0, 50, 400)
    finder = MarkerFinder(K, DIST, 40.0)
    poses = finder.find(render([(0, T_m)]))
    T_arm_cam = load_camera_to_arm({"translation_mm": [-60, 0, 250], "rotation_rpy_deg": [-90, 0, -90]})
    r = ItemLocator(finder, SLOTS, T_arm_cam).locate("egg", poses, [])
    expected = apply(T_arm_cam, apply(T_m, SLOTS[0]["item_offset_mm"]))
    assert np.linalg.norm(r.position_arm_mm - expected) < 6.0
    d = r.to_dict()
    assert len(d["position_arm_m"]) == 3 and abs(d["position_arm_m"][0] - expected[0] / 1000) < 0.01


def test_measure_camera_to_arm_recovers_true_pose():
    from tools.measure_camera_to_arm import average_pose, compute_T_arm_cam
    # true camera pose in arm frame: 250 mm up, 60 mm behind, tilted 30 deg down
    T_arm_cam_true = make_T(rpy_deg_to_R(-90 - 30, 0, -90), [-60, 0, 250])
    marker_at, marker_rpy = [200.0, 0.0, 0.0], [0.0, 0.0, -90.0]
    T_arm_marker = make_T(rpy_deg_to_R(*marker_rpy), marker_at)
    T_cam_marker = invert(T_arm_cam_true) @ T_arm_marker
    finder = MarkerFinder(K, DIST, 40.0)
    poses = [p.T_cam_marker for p in finder.find(render([(0, T_cam_marker)]))]
    assert poses, "marker not found in synthetic image"
    T_est = compute_T_arm_cam(average_pose(poses), marker_at, marker_rpy)
    assert np.linalg.norm(T_est[:3, 3] - T_arm_cam_true[:3, 3]) < 8.0
    ang = np.degrees(np.arccos(np.clip((np.trace(T_est[:3, :3].T @ T_arm_cam_true[:3, :3]) - 1) / 2, -1, 1)))
    assert ang < 2.0
