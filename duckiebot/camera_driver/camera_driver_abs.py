from abc import ABC, abstractmethod
import threading
import numpy as np
import cv2
import os
import yaml
from typing import Tuple, Optional


class CameraDriverAbs(ABC):
    def __init__(self, config_file: str = None):
        if config_file is None:
            current_dir = os.path.dirname(__file__)
            config_file = os.path.join(current_dir, 'config/camera_config.yaml')

        self._load_config(config_file)
        self.config_file = config_file

        self._running = False
        self._frame_count = 0
        self._device = None

        # Single-producer pattern: one background thread reads GStreamer/camera
        # and stores the latest frame here. All consumers call read() which just
        # returns a copy — no thread ever blocks on GStreamer directly.
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None

    @abstractmethod
    def _initialize_camera(self):
        pass

    @abstractmethod
    def _capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def _release_camera(self):
        pass

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._running:
            print("[Camera] Already running")
            return

        self._initialize_camera()
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name='CameraCapture'
        )
        self._capture_thread.start()

        print(f"Camera started successfully at {self.width}x{self.height} @ {self.framerate}fps")

    def stop(self):
        if not self._running:
            print("[Camera] Already stopped")
            return

        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._release_camera()
        with self._frame_lock:
            self._latest_frame = None
        print("Camera stopped")

    # -- capture loop (single producer thread) -----------------------------

    def _capture_loop(self):
        """Continuously read from the hardware into _latest_frame."""
        consecutive_failures = 0
        while self._running:
            ok, frame = self._capture_frame()
            if ok and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    print("[Camera] Too many consecutive read failures — stopping capture")
                    break

    # -- consumer API (safe to call from any thread) -----------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            if not hasattr(self, '_warned_not_running'):
                print("[Camera] Warning: Camera not running, call start() first")
                self._warned_not_running = True
            return False, None

        with self._frame_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        return False, None

    def read_jpeg(self) -> Tuple[bool, Optional[bytes]]:
        ok, frame = self.read()
        if not ok or frame is None:
            return False, None
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (True, jpeg.tobytes()) if ret else (False, None)

    # -- config ------------------------------------------------------------

    def _load_config(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)

            res = config.get('resolution', {})
            self.width = res.get('width', 640)
            self.height = res.get('height', 480)

            self.framerate = config.get('framerate', 30)
            self.sensor_mode = config.get('sensor_mode', 0)
            self.use_hw_acceleration = config.get('use_hw_acceleration', True)

            self.maker = config.get('maker', 'Unknown')
            self.model = config.get('model', 'Unknown')
            self.fov = config.get('fov', 160)
            self.exposure_mode = config.get('exposure_mode', 'sports')

            print(f"[CameraDriver] Loaded config from {filepath}")

        except FileNotFoundError:
            print(f"[CameraDriver] Warning: Config not found: {filepath}")

    # -- properties --------------------------------------------------------

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def is_active(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __del__(self):
        if self._running:
            self.stop()
