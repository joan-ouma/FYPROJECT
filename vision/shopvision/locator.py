"""Turn a shelf marker + a YOLO detection into the item's 3D position for the arm.

Method (proposal Section 3.5.4):
  1. The slot's ArUco marker gives the exact pose of the slot (camera frame).
  2. The item's expected centre = marker pose + item_offset_mm from shelf_slots.yaml.
  3. If the camera sees the target product near that spot, its box centre refines the
     left/right position (the item may not sit in the middle of its slot): the ray through
     the box centre is cut with a plane at the item's depth, parallel to the marker.
  4. The result is converted from the camera frame to the arm frame (camera_to_arm.yaml).
"""
from dataclasses import dataclass, field

import numpy as np

from .transforms import apply, invert


@dataclass
class ItemLocation:
    product: str
    slot_name: str
    marker_id: int
    position_cam_mm: np.ndarray
    position_arm_mm: np.ndarray
    expected_cam_mm: np.ndarray       # from the marker alone
    refined_by_detection: bool
    detection_confidence: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self):
        r = lambda a: [round(float(v), 1) for v in a]
        return {"product": self.product, "slot": self.slot_name, "marker_id": self.marker_id,
                "position_cam_mm": r(self.position_cam_mm),
                "position_arm_mm": r(self.position_arm_mm),
                "position_arm_m": [round(float(v) / 1000.0, 4) for v in self.position_arm_mm],
                "expected_cam_mm": r(self.expected_cam_mm),
                "refined_by_detection": self.refined_by_detection,
                "detection_confidence": round(self.detection_confidence, 3),
                "notes": list(self.notes)}


class ItemLocator:
    def __init__(self, marker_finder, shelf_slots, T_arm_cam, max_match_distance_px=120,
                 max_refine_shift_mm=40.0):
        self.mf = marker_finder
        self.slots = shelf_slots          # {marker_id: {name, product, item_offset_mm}}
        self.T_arm_cam = np.asarray(T_arm_cam, dtype=float)
        self.max_px = max_match_distance_px
        self.max_shift = max_refine_shift_mm

    def slots_for(self, product):
        return [mid for mid, s in self.slots.items() if s["product"] == product]

    def locate(self, product, marker_poses, detections):
        """Returns ItemLocation, or None if no marker for this product is visible."""
        wanted = set(self.slots_for(product))
        if not wanted:
            raise KeyError(f"No shelf slot for '{product}' in shelf_slots.yaml")
        visible = [m for m in marker_poses if m.marker_id in wanted]
        if not visible:
            return None
        # if the product has several slots, use the nearest visible one
        m = min(visible, key=lambda p: p.distance_mm)
        slot = self.slots[m.marker_id]
        expected = apply(m.T_cam_marker, slot["item_offset_mm"])
        loc = ItemLocation(product, slot["name"], m.marker_id, expected.copy(), None,
                           expected.copy(), False)

        # match a detection of the product to the expected spot
        exp_px = self.mf.project(expected)
        cands = [d for d in detections if d.product == product]
        best = None
        for d in cands:
            dist_px = float(np.hypot(d.center[0] - exp_px[0], d.center[1] - exp_px[1]))
            if dist_px <= self.max_px and (best is None or dist_px < best[0]):
                best = (dist_px, d)
        if best is None:
            loc.notes.append("no detection near the slot; position from marker only")
        else:
            d = best[1]
            refined = self._refine(m, slot, d.center)
            if refined is None:
                loc.notes.append("could not refine (ray parallel to shelf); marker only")
            else:
                shift = float(np.linalg.norm(refined - expected))
                if shift > self.max_shift:
                    loc.notes.append(f"detection {shift:.0f} mm from slot centre - too far, ignored")
                else:
                    loc.position_cam_mm = refined
                    loc.refined_by_detection = True
            loc.detection_confidence = d.confidence
        loc.position_arm_mm = apply(self.T_arm_cam, loc.position_cam_mm)
        return loc

    def _refine(self, marker, slot, center_px):
        """Cut the ray through the box centre with the plane z = item depth (marker frame)."""
        ray_cam = self.mf.pixel_ray(*center_px)
        T_mc = invert(marker.T_cam_marker)               # camera -> marker frame
        origin_m = T_mc[:3, 3]                           # camera centre, marker frame
        dir_m = T_mc[:3, :3] @ ray_cam
        z_item = slot["item_offset_mm"][2]
        if abs(dir_m[2]) < 1e-6:
            return None
        s = (z_item - origin_m[2]) / dir_m[2]
        if s <= 0:
            return None
        hit_m = origin_m + s * dir_m
        # Use the detection only for left/right (x). Height (y) and depth (z) come from the
        # slot: items stand on the shelf, so their height is known, while the box centre
        # moves up and down when the top or bottom of the item is cut off or hidden.
        refined_m = np.array([hit_m[0], slot["item_offset_mm"][1], z_item])
        return apply(marker.T_cam_marker, refined_m)
