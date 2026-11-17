"""VisionService.find_item(): the one function the rest of the robot calls.

    vision = VisionService()
    result = vision.find_item("egg")
    if result["found"]:
        x, y, z = result["position_arm_m"]   # give this to the arm (Joy)

The result is a plain dict so it can be sent as JSON over ROS 2 or MQTT.
Statuses:
  "found"            confirmed (3 frames >= 0.6) and located
  "not_confirmed"    product not confirmed within the timeout
  "no_marker"        product confirmed but its slot marker is not visible
  "camera_error"     no frames from the camera
"""
import time

from .aruco import MarkerFinder, load_calibration
from .config import load_config, load_shelf_slots, load_yaml
from .confirm import ConfirmTracker
from .locator import ItemLocator
from .transforms import load_camera_to_arm


class VisionService:
    def __init__(self, config=None, camera=None, detector=None, source=None):
        self.cfg = config or load_config()
        c = self.cfg
        if camera is None:
            from .camera import open_camera
            camera = open_camera(source, (c["camera"]["width"], c["camera"]["height"]),
                                 c["camera"].get("lens_position"))
        self.camera = camera
        if detector is None:
            from .detector import Detector
            detector = Detector(c["model"]["weights"], c["model"].get("image_size", 640),
                                c["model"].get("min_confidence", 0.25))
        self.detector = detector
        K, dist, _ = load_calibration(c["camera"]["calibration_file"])
        self.markers = MarkerFinder(K, dist, c["aruco"]["marker_size_mm"], c["aruco"]["dictionary"])
        slots = load_shelf_slots(c["locator"]["shelf_slots_file"])
        T_arm_cam = load_camera_to_arm(load_yaml(c["locator"]["camera_to_arm_file"]))
        self.locator = ItemLocator(self.markers, slots, T_arm_cam,
                                   c["locator"].get("max_match_distance_px", 120))

    def look(self):
        """One frame: returns (frame, detections, marker_poses)."""
        frame = self.camera.read()
        if frame is None:
            return None, [], []
        return frame, self.detector.detect(frame), self.markers.find(frame)

    def find_item(self, product, timeout_s=None, on_frame=None):
        c = self.cfg
        timeout = timeout_s if timeout_s is not None else c["find_item"]["timeout_s"]
        tracker = ConfirmTracker(product, c["confirm"]["min_confidence"],
                                 c["confirm"]["frames_in_a_row"])
        start = time.time()
        frames = 0
        last_markers, last_dets = [], []
        while time.time() - start < timeout:
            frame, dets, poses = self.look()
            if frame is None:
                return self._result(product, "camera_error", frames, start)
            frames += 1
            if on_frame:
                on_frame(frame, dets, poses, tracker)
            last_dets, last_markers = dets, poses
            if tracker.update(dets):
                loc = self.locator.locate(product, poses, dets)
                if loc is None:
                    # keep looking a little: the marker may be blocked for a moment
                    continue
                r = self._result(product, "found", frames, start)
                r.update(loc.to_dict())
                r["found"] = True
                r["confidence"] = round(tracker.best.confidence, 3)
                return r
        status = "no_marker" if tracker.confirmed else "not_confirmed"
        r = self._result(product, status, frames, start)
        r["seen"] = sorted({d.product for d in last_dets})
        r["markers_seen"] = [m.marker_id for m in last_markers]
        return r

    @staticmethod
    def _result(product, status, frames, start):
        elapsed = time.time() - start
        return {"product": product, "status": status, "found": False,
                "frames": frames, "time_s": round(elapsed, 2),
                "fps": round(frames / elapsed, 1) if elapsed > 0 else 0.0}

    def close(self):
        self.camera.close()
