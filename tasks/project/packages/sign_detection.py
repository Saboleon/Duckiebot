"""AprilTag-based traffic-sign detection for the Traffic Signs project.

Duckietown traffic signs each carry an AprilTag (tag36h11 family). We use
OpenCV's built-in ArUco module to read those tags - no extra library needed.

A detected tag is turned into a `SignObservation`: the sign category, the turns
it permits (for intersection signs), and how big/where it is in the frame so the
state machine can decide when we are close enough to act.
"""

import os
import yaml
import cv2
import numpy as np


# sign categories
STOP       = "stop"
YIELD      = "yield"
PEDESTRIAN = "pedestrian"
T_RIGHT    = "t_right"      # side road on the right  -> straight | right
T_LEFT     = "t_left"       # side road on the left   -> straight | left
T_JUNCTION = "t_junction"   # T intersection          -> left | right
UNKNOWN    = "unknown"

# turns
LEFT     = "left"
RIGHT    = "right"
STRAIGHT = "straight"

# which turns each intersection sign permits
ALLOWED_TURNS = {
    T_RIGHT:    [STRAIGHT, RIGHT],
    T_LEFT:     [STRAIGHT, LEFT],
    T_JUNCTION: [LEFT, RIGHT],
}

INTERSECTION_SIGNS = set(ALLOWED_TURNS.keys())

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml"
))


class SignObservation:
    """One detected traffic sign in one frame."""

    __slots__ = ("tag_id", "sign_type", "height_px", "distance_m",
                 "cx", "cy", "corners")

    def __init__(self, tag_id, sign_type, height_px, distance_m, cx, cy, corners):
        self.tag_id     = tag_id
        self.sign_type  = sign_type
        self.height_px  = height_px
        self.distance_m = distance_m
        self.cx         = cx          # tag centre x (px)
        self.cy         = cy          # tag centre y (px)
        self.corners    = corners     # 4x2 float array

    @property
    def allowed_turns(self):
        return list(ALLOWED_TURNS.get(self.sign_type, []))

    @property
    def is_intersection(self):
        return self.sign_type in INTERSECTION_SIGNS

    def __repr__(self):
        return (f"<Sign {self.sign_type} tag={self.tag_id} "
                f"h={self.height_px:.0f}px d={self.distance_m:.2f}m>")


class SignDetector:
    """Reads traffic-sign AprilTags and maps them to sign categories."""

    def __init__(self, config=None):
        cfg = config if config is not None else _load_config()
        self.cfg = cfg

        # build tag-id -> sign category lookup from the config
        self.id_to_sign = {}
        for sign_type, ids in (cfg.get("signs") or {}).items():
            for tag_id in ids:
                self.id_to_sign[int(tag_id)] = sign_type

        cam = cfg.get("camera", {})
        self.tag_size_m = float(cam.get("tag_size_m", 0.065))
        self.focal_px   = float(cam.get("focal_px", 320.0))

        det = cfg.get("detection", {})
        self.observe_px = float(det.get("observe_px", 28))
        self.act_px     = float(det.get("act_px", 72))

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    # -- detection ---------------------------------------------------------
    def detect(self, frame_bgr):
        """Return a list of SignObservation for every known tag in the frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        out = []
        if ids is None:
            return out

        for quad, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            sign_type = self.id_to_sign.get(tag_id, UNKNOWN)
            if sign_type == UNKNOWN:
                continue
            pts = quad.reshape(-1, 2)
            h = float(pts[:, 1].max() - pts[:, 1].min())
            if h <= 0:
                continue
            dist = self.focal_px * self.tag_size_m / h
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            out.append(SignObservation(tag_id, sign_type, h, dist, cx, cy, pts))
        return out

    def closest_actionable(self, observations):
        """The nearest sign that is close enough (height >= act_px) to act on,
        or None. Nearest == largest apparent tag."""
        actionable = [o for o in observations if o.height_px >= self.act_px]
        if not actionable:
            return None
        return max(actionable, key=lambda o: o.height_px)

    def in_view(self, observations):
        """Nearest sign that is at least within 'observe' range, or None."""
        visible = [o for o in observations if o.height_px >= self.observe_px]
        if not visible:
            return None
        return max(visible, key=lambda o: o.height_px)

    # -- visualization -----------------------------------------------------
    def draw(self, frame_bgr, observations):
        """Draw boxes + labels for detected signs (returns the same frame)."""
        for o in observations:
            pts = o.corners.astype(int)
            close = o.height_px >= self.act_px
            color = (0, 215, 255) if not close else (0, 0, 255)
            cv2.polylines(frame_bgr, [pts], True, color, 2)
            label = f"{o.sign_type} #{o.tag_id} {o.distance_m:.2f}m"
            x, y = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(frame_bgr, label, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_bgr


def _load_config(path=None):
    path = path or _CONFIG_FILE
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
