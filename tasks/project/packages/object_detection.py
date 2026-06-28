"""Obstacle stopping via the trained object detector.

Reuses the model + logic from the `object_detection` task:
  * `ObjectDetectionAgent` runs the YOLO ONNX model (best.onnx) and returns
    detections already filtered to duckies (its `filter_by_classes` keeps class 0).
  * `should_stop` decides whether a duckie is close enough and in our lane.

Inference is run on a background thread so it never stalls the control loop; the
agent just reads `blocked`. If the model or its code can't be loaded, obstacle
detection disables itself and the rest of the agent keeps working.

The model file ships with the project at tasks/project/models/best.onnx (the
object_detection copy is git-ignored), and we point the detector at that copy.
"""

import os
import time
import threading

import cv2

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MODEL_PATH = os.path.join(_ROOT, "tasks", "project", "models", "best.onnx")


class ObstacleStopper:
    """Background duckie detector. Exposes `blocked` / `reason`."""

    def __init__(self, cfg):
        ocfg = (cfg or {}).get("obstacle", {})
        self.enabled = bool(ocfg.get("enabled", False))
        # duck bbox bottom past this fraction of frame height -> stop. Lower =
        # stops from farther away (needed at higher cruise speed). Default 0.72
        # matches the shared object_detection task.
        self.stop_y_frac = float(ocfg.get("stop_y_frac", 0.72))
        # duck bbox height must exceed this fraction of frame size — filters out
        # distant ducks whose bbox is tiny. Raise to ignore more far-away ducks.
        self.min_height_frac = float(ocfg.get("min_height_frac", 0.06))
        # duck center-x must be at or to the right of this fraction of frame width
        # to count as being in our lane. Ducks whose center is left of this are
        # assumed to be in the oncoming lane.
        self.lane_x_min_frac = float(ocfg.get("lane_x_min_frac", 0.30))
        self.blocked = False
        self.reason = ""

        # right-of-way: which detections count as crossing traffic, and the
        # vertical band of the frame to watch for them (used by yield/stop).
        ycfg = (cfg or {}).get("yield_traffic", {})
        self.traffic_enabled  = bool(ycfg.get("enabled", True))
        self.traffic_classes  = set(ycfg.get("classes", []) or [])      # [] = any object
        self.traffic_top_frac = float(ycfg.get("zone_top_frac", 0.30))
        self.traffic_bot_frac = float(ycfg.get("zone_bottom_frac", 0.80))
        self.traffic_min_score = float(ycfg.get("min_score", 0.5))      # ignore weak boxes
        self._dets = []          # latest detections (for traffic_present)
        self._size = 0

        self._agent = None
        self._should_stop = None
        self._camera = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.load_error = None

        if not self.enabled:
            return

        try:
            import tasks.object_detection.packages.integration_activity as ia
            # use the model copy that ships with the project task
            if os.path.isfile(_MODEL_PATH):
                ia.MODEL_PATH = _MODEL_PATH
            from tasks.object_detection.packages.agent import ObjectDetectionAgent
            from tasks.object_detection.packages.stop_activity import should_stop

            self._agent = ObjectDetectionAgent()
            self._should_stop = should_stop
            if not self._agent.model_loaded:
                self.load_error = self._agent.load_error or "model not loaded"
                print(f"[project] obstacle detection off: {self.load_error}")
                self.enabled = False
        except Exception as e:
            self.load_error = str(e)
            print(f"[project] obstacle detection unavailable: {e}")
            self.enabled = False

    def start(self, camera):
        if not self.enabled:
            return
        self._camera = camera
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ObstacleThread")
        self._thread.start()
        print("[project] obstacle detection running (duckies)")

    def _loop(self):
        size = self._agent.img_size
        while not self._stop_event.is_set():
            ok, frame = self._camera.read()          # BGR
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            # detect() + should_stop() both work in img_size x img_size space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            square = cv2.resize(rgb, (size, size))
            try:
                dets = self._agent.detect(square)
            except Exception as e:
                print(f"[project] detection error: {e}")
                time.sleep(0.05)
                continue
            if dets is None:                         # frame skipped by the agent
                continue
            blocked, reason = self._evaluate(dets, size)
            with self._lock:
                self.blocked = blocked
                self.reason = reason
                self._dets = dets
                self._size = size

    def traffic_present(self):
        """True if crossing traffic is visible right now (for yield/stop right-
        of-way). Counts detections whose center falls in the configured vertical
        band; if `traffic_classes` is set, only those classes count (1=other
        bot, 0=duckie). Returns False when obstacle detection is unavailable, so
        callers fall back to a plain timed pause."""
        if not (self.enabled and self.traffic_enabled):
            return False
        with self._lock:
            dets = list(self._dets)
            size = self._size
        if size <= 0:
            return False
        top = size * self.traffic_top_frac
        bot = size * self.traffic_bot_frac
        for (x1, y1, x2, y2), score, cls_id in dets:
            if score < self.traffic_min_score:
                continue
            if self.traffic_classes and cls_id not in self.traffic_classes:
                continue
            cy = 0.5 * (y1 + y2)
            if top <= cy <= bot:
                return True
        return False

    def draw(self, frame):
        """Overlay the current detections + the traffic watch-band on the frame
        (BGR, modified in place). Red box = counts as crossing traffic right now;
        amber box = detected but ignored (out of band / wrong class / low score).
        Use it to see exactly what the right-of-way logic is reacting to."""
        if not self.enabled:
            return frame
        with self._lock:
            dets = list(self._dets)
            size = self._size
        if size <= 0:
            return frame
        h, w = frame.shape[:2]
        sx, sy = w / float(size), h / float(size)
        y_top, y_bot = int(h * self.traffic_top_frac), int(h * self.traffic_bot_frac)
        cv2.line(frame, (0, y_top), (w, y_top), (255, 180, 0), 1)
        cv2.line(frame, (0, y_bot), (w, y_bot), (255, 180, 0), 1)
        names = {0: "duckie", 1: "truck/bot", 2: "sign"}
        for (x1, y1, x2, y2), score, cls_id in dets:
            cy = 0.5 * (y1 + y2)
            counts = ((not self.traffic_classes) or (cls_id in self.traffic_classes)) \
                     and score >= self.traffic_min_score \
                     and (size * self.traffic_top_frac) <= cy <= (size * self.traffic_bot_frac)
            color = (0, 0, 255) if counts else (0, 200, 255)
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"{names.get(cls_id, cls_id)} {score:.2f}",
                        (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def _evaluate(self, dets, size):
        """Stop if any duckie is close ahead AND in our lane.

        Three filters before triggering a stop:
          1. y2 > stop_y_frac  — duck must be low enough (close enough) in frame.
          2. bbox height > min_height_frac — filters tiny/far-away detections.
          3. center_x > lane_x_min_frac — ignores ducks clearly in oncoming lane.
        """
        stop_y    = size * self.stop_y_frac
        min_h     = size * self.min_height_frac
        lane_x    = size * self.lane_x_min_frac

        for (x1, y1, x2, y2), score, cls_id in dets:
            if (y2 - y1) < min_h:
                continue        # too far away (bbox too small)
            cx = 0.5 * (x1 + x2)
            if cx < lane_x:
                continue        # in oncoming lane
            if y2 > stop_y:
                return True, "duckie detected ahead"
        return False, ""

    def status(self):
        with self._lock:
            return self.blocked, self.reason

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
