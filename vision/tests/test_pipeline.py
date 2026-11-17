"""Tests for the confirmation rule, find_item(), the dataset checker, markers and metrics."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from shopvision.camera import StaticFrames
from shopvision.confirm import ConfirmTracker
from shopvision.detector import Detection, FakeDetector
from shopvision.service import VisionService
from synthetic import K, project, render, write_calibration
from test_geometry import SLOTS, shelf_marker

VISION = Path(__file__).resolve().parents[1]


def d(product, conf, box=(0, 0, 10, 10)):
    return Detection(product, conf, box)


def test_confirm_needs_three_in_a_row():
    t = ConfirmTracker("egg", 0.6, 3)
    assert not t.update([d("egg", 0.9)])
    assert not t.update([d("egg", 0.7)])
    assert not t.update([d("egg", 0.5)])          # too low: count restarts
    assert not t.update([d("egg", 0.8)])
    assert not t.update([d("egg", 0.8), d("matchbox", 0.99)])
    assert t.update([d("egg", 0.61)])
    assert t.best.confidence == 0.8


def test_confirm_ignores_other_products():
    t = ConfirmTracker("egg", 0.6, 3)
    for _ in range(5):
        assert not t.update([d("toothpaste", 0.95)])


def make_config(tmp_path):
    cal = tmp_path / "cal.yaml"
    write_calibration(cal)
    slots = tmp_path / "slots.yaml"
    slots.write_text(yaml.safe_dump({"slots": SLOTS}))
    c2a = tmp_path / "c2a.yaml"
    c2a.write_text(yaml.safe_dump({"translation_mm": [0, 0, 0], "rotation_rpy_deg": [0, 0, 0]}))
    return {"camera": {"width": 1280, "height": 720, "calibration_file": str(cal)},
            "model": {"weights": "unused"},
            "confirm": {"min_confidence": 0.6, "frames_in_a_row": 3},
            "aruco": {"dictionary": "DICT_4X4_50", "marker_size_mm": 40.0},
            "locator": {"shelf_slots_file": str(slots), "camera_to_arm_file": str(c2a),
                        "max_match_distance_px": 120},
            "find_item": {"timeout_s": 3.0}}


def test_find_item_end_to_end(tmp_path):
    T_m = shelf_marker(-30, 70, 380)
    frame = render([(0, T_m), (1, shelf_marker(90, 70, 380))])
    item = np.array(SLOTS[0]["item_offset_mm"]) + [15, 0, 0]
    from shopvision.transforms import apply
    true_cam = apply(T_m, item)
    u, v = project(true_cam)
    egg = d("egg", 0.88, (u - 30, v - 40, u + 30, v + 40))
    det = FakeDetector([[egg], [d("egg", 0.4)], [egg], [egg], [egg]])
    vs = VisionService(make_config(tmp_path), camera=StaticFrames([frame]), detector=det)
    r = vs.find_item("egg")
    assert r["status"] == "found" and r["found"]
    assert r["frames"] == 5                        # one low frame reset the count
    assert r["refined_by_detection"]
    assert np.linalg.norm(np.array(r["position_arm_mm"]) - true_cam) < 6.0
    assert r["slot"] == "top_left" and r["marker_id"] == 0


def test_find_item_not_confirmed_and_no_marker(tmp_path):
    frame = render([(1, shelf_marker(0, 70, 380))])       # only tea_leaves' marker visible
    cfg = make_config(tmp_path)
    cfg["find_item"]["timeout_s"] = 0.5
    vs = VisionService(cfg, camera=StaticFrames([frame]), detector=FakeDetector([[d("matchbox", 0.9)]]))
    assert vs.find_item("egg")["status"] == "not_confirmed"
    vs = VisionService(cfg, camera=StaticFrames([frame]), detector=FakeDetector([[d("egg", 0.9)]]))
    r = vs.find_item("egg")
    assert r["status"] == "no_marker" and r["markers_seen"] == [1]


def test_find_item_camera_error(tmp_path):
    vs = VisionService(make_config(tmp_path), camera=StaticFrames([], loop=False), detector=FakeDetector([]))
    assert vs.find_item("egg")["status"] == "camera_error"


def make_roboflow_dataset(root, names, per_split=(6, 2, 2)):
    rng = np.random.default_rng(0)
    for split, n in zip(("train", "valid", "test"), per_split):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / split / "images" / f"img{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
            lines = [f"{c} {rng.uniform(.3, .7):.3f} {rng.uniform(.3, .7):.3f} 0.1 0.1"
                     for c in range(len(names))]
            (root / split / "labels" / f"img{i}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"train": "../train/images", "val": "../valid/images", "test": "../test/images",
         "nc": len(names), "names": names}))


def run_check(root, *extra):
    return subprocess.run([sys.executable, str(VISION / "training" / "check_dataset.py"), str(root), *extra],
                          capture_output=True, text=True)


def test_check_dataset_ok_and_writes_yaml(tmp_path):
    from shopvision.config import load_products
    make_roboflow_dataset(tmp_path, load_products())
    r = run_check(tmp_path, "--write-yaml")
    assert r.returncode == 0, r.stdout + r.stderr
    data = yaml.safe_load((tmp_path / "data_checked.yaml").read_text())
    assert Path(data["train"]).is_dir() and Path(data["test"]).is_dir()


def test_check_dataset_catches_wrong_order(tmp_path):
    from shopvision.config import load_products
    names = load_products()
    make_roboflow_dataset(tmp_path, list(reversed(names)))
    r = run_check(tmp_path)
    assert r.returncode == 1 and "different ORDER" in r.stdout


def test_make_markers_pdf_is_detectable(tmp_path):
    out = tmp_path / "m.pdf"
    r = subprocess.run([sys.executable, str(VISION / "markers" / "make_markers.py"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    png = tmp_path / "page"
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", str(out), str(png)], check=True)
    img = cv2.imread(str(png) + ".png")
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert sorted(ids.ravel().tolist()) == [0, 1, 2, 3, 4]


def test_metrics_summary():
    from shopvision.metrics import summary_markdown
    box = SimpleNamespace(ap_class_index=np.array([0, 1]), map50=0.87, mp=0.9, mr=0.85,
                          class_result=lambda i: [(0.95, 0.9, 0.96, 0.7), (0.85, 0.8, 0.78, 0.5)][i])
    md, m = summary_markdown("t", SimpleNamespace(box=box), {0: "egg", 1: "matchbox"})
    assert m == 0.87 and "DOES NOT MEET" in md and "Take more photos of: matchbox" in md
