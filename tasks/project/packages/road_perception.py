"""Road-surface perception: the red stop line before an intersection.

(Obstacle/duckie detection lives in object_detection.py, which reuses the
trained model.)
"""

import cv2
import numpy as np


class StopLineDetector:
    """Detects the red band painted on the floor just before an intersection."""

    def __init__(self, cfg):
        s = cfg.get("stopline", {})
        self.red1_lo = np.array(s.get("red1_lo", [0,   90, 70]))
        self.red1_hi = np.array(s.get("red1_hi", [10, 255, 255]))
        self.red2_lo = np.array(s.get("red2_lo", [170, 90, 70]))
        self.red2_hi = np.array(s.get("red2_hi", [180,255, 255]))
        self.roi_top = float(s.get("roi_top", 0.65))
        self.trigger = float(s.get("trigger_area_frac", 0.045))

    def detect(self, frame_bgr):
        """Return (at_line: bool, area_frac: float)."""
        h, w = frame_bgr.shape[:2]
        y0 = int(h * self.roi_top)
        roi = frame_bgr[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.red1_lo, self.red1_hi) | \
               cv2.inRange(hsv, self.red2_lo, self.red2_hi)
        frac = float(np.count_nonzero(mask)) / mask.size
        return frac >= self.trigger, frac
