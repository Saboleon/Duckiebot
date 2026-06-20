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
        self.blocked = False
        self.reason = ""

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
            blocked, reason = self._should_stop(dets, size)
            with self._lock:
                self.blocked = blocked
                self.reason = reason

    def status(self):
        with self._lock:
            return self.blocked, self.reason

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
