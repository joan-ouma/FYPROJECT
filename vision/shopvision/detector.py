"""Run the trained YOLO model on a frame and return simple Detection objects."""
from dataclasses import dataclass, asdict

from .config import resolve


@dataclass
class Detection:
    product: str          # class name, e.g. "egg"
    confidence: float     # 0..1
    box: tuple            # (x1, y1, x2, y2) in pixels
    class_id: int = -1

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self):
        d = asdict(self)
        d["center"] = self.center
        return d


class Detector:
    """Wraps an Ultralytics YOLO model (.pt, .onnx or an NCNN folder)."""

    def __init__(self, weights, image_size=640, min_confidence=0.25):
        from ultralytics import YOLO  # imported here so other modules work without it

        path = resolve(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Train one (training/train.py) and export it "
                f"(training/export_model.py), or set model.weights in config/vision_config.yaml")
        self.model = YOLO(str(path), task="detect")
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.names = self.model.names  # {class_id: name}

    def detect(self, frame):
        results = self.model.predict(frame, imgsz=self.image_size, conf=self.min_confidence,
                                     verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                       r.boxes.cls.tolist()):
                cid = int(cls)
                out.append(Detection(product=self.names[cid], confidence=float(conf),
                                     box=tuple(float(v) for v in xyxy), class_id=cid))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


class FakeDetector:
    """Returns detections you give it, one list per frame (for tests and dry runs)."""

    def __init__(self, per_frame):
        self.per_frame = list(per_frame)
        self.i = 0
        self.names = {}

    def detect(self, frame):
        if not self.per_frame:
            return []
        dets = self.per_frame[min(self.i, len(self.per_frame) - 1)]
        self.i += 1
        return list(dets)


def draw_detections(frame, detections, highlight=None):
    """Draw boxes and labels on a copy of the frame."""
    import cv2

    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        colour = (0, 200, 0) if d.product == highlight else (255, 160, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = f"{d.product} {d.confidence:.2f}"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
    return img
