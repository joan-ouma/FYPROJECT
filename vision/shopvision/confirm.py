"""The confirmation rule from the proposal (Section 3.5.3):

A product is confirmed only when it is detected with confidence >= 0.6
in 3 frames in a row. One frame without it starts the count again.
"""


class ConfirmTracker:
    def __init__(self, target, min_confidence=0.6, frames_in_a_row=3):
        self.target = target
        self.min_confidence = min_confidence
        self.frames_in_a_row = frames_in_a_row
        self.streak = 0
        self.best = None  # best detection of the target in the current streak

    def reset(self):
        self.streak = 0
        self.best = None

    def update(self, detections):
        """Give it the detections of one frame. Returns True once the target is confirmed."""
        hits = [d for d in detections
                if d.product == self.target and d.confidence >= self.min_confidence]
        if not hits:
            self.reset()
            return False
        top = max(hits, key=lambda d: d.confidence)
        self.streak += 1
        if self.best is None or top.confidence >= self.best.confidence:
            self.best = top
        return self.streak >= self.frames_in_a_row

    @property
    def confirmed(self):
        return self.streak >= self.frames_in_a_row
