"""shopvision - computer vision for the Autonomous Shopping Robot (Objective 1).

Main entry point for the rest of the robot:

    from shopvision.service import VisionService
    vision = VisionService()             # reads config/vision_config.yaml
    result = vision.find_item("egg")     # dict: found, confidence, position_arm_mm, ...

Modules
    config      loads the YAML settings
    camera      Pi camera / webcam / video / image-folder sources
    detector    runs the trained YOLO model
    confirm     the "3 frames in a row at >= 0.6" rule
    aruco       finds ArUco markers and their 3D pose
    transforms  4x4 transform helpers (camera <-> arm)
    locator     turns a marker + a detection into the item's 3D position
    service     find_item(): ties everything together
"""
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[1]

__all__ = ["VISION_DIR"]
