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
TURN_LEFT  = "turn_left"    # mandatory left  -> left only
TURN_RIGHT = "turn_right"   # mandatory right -> right only
GO_STRAIGHT = "go_straight" # mandatory straight ahead
YIELD_LEFT = "yield_left"   # yield, THEN mandatory left (combined sign)
PARKING    = "parking"      # parking area (not an intersection)
UNKNOWN    = "unknown"

# turns
LEFT     = "left"
RIGHT    = "right"
STRAIGHT = "straight"

# which turns each intersection sign permits. Single-direction signs list one
# turn, so random.choice() over the list always yields that direction.
ALLOWED_TURNS = {
    T_RIGHT:     [STRAIGHT, RIGHT],
    T_LEFT:      [STRAIGHT, LEFT],
    T_JUNCTION:  [LEFT, RIGHT],
    TURN_LEFT:   [LEFT],
    TURN_RIGHT:  [RIGHT],
    GO_STRAIGHT: [STRAIGHT],
    YIELD_LEFT:  [LEFT],   # turn part; the yield part is gated separately
}

INTERSECTION_SIGNS = set(ALLOWED_TURNS.keys())

# signs that require a yield (brief pause) before proceeding through the
# intersection. YIELD_LEFT both yields AND forces a left turn.
YIELD_SIGNS = {YIELD, YIELD_LEFT}

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

    def __init__(self, config=None, sim=True):
        cfg = config if config is not None else _load_config()
        self.cfg = cfg

        # Pick the right sign-ID table: sim vs real bot, falling back to the
        # other section if the preferred one is absent or all-empty.
        if sim:
            sign_cfg = cfg.get("signs_sim") or cfg.get("signs_real") or cfg.get("signs") or {}
        else:
            sign_cfg = cfg.get("signs_real") or cfg.get("signs_sim") or cfg.get("signs") or {}

        # build tag-id -> sign category lookup from the config
        self.id_to_sign = {}
        for sign_type, ids in sign_cfg.items():
            for tag_id in (ids or []):
                self.id_to_sign[int(tag_id)] = sign_type

        env = "sim" if sim else "real"
        print(f"[SignDetector] mode={env}, {len(self.id_to_sign)} tag IDs loaded")

        # Discovery mode: show unknown tag IDs on the video overlay so you can
        # read them off the browser and fill in signs_real in the config.
        # Auto-enabled on the real bot when signs_real has no IDs yet.
        self.discovery_mode = (not sim) and (len(self.id_to_sign) == 0)
        self._reported_ids = set()  # tracks IDs already printed, avoids spam

        cam = cfg.get("camera", {})
        self.tag_size_m = float(cam.get("tag_size_m", 0.065))
        self.focal_px   = float(cam.get("focal_px", 320.0))

        det = cfg.get("detection", {})
        self.observe_px = float(det.get("observe_px", 28))
        self.act_px     = float(det.get("act_px", 72))

        # Support both OpenCV 4.7+ (ArucoDetector class) and older versions
        # (the Jetson JetPack SDK ships OpenCV 4.1-4.6).
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._legacy_aruco = False
        except AttributeError:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            params = cv2.aruco.DetectorParameters_create()
            self._aruco_dict   = dictionary
            self._aruco_params = params
            self._detector     = None
            self._legacy_aruco = True

    # -- detection ---------------------------------------------------------
    def detect(self, frame_bgr):
        """Return a list of SignObservation for every known tag in the frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._legacy_aruco:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)
        else:
            corners, ids, _ = self._detector.detectMarkers(gray)
        out = []
        if ids is None:
            return out

        for quad, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            sign_type = self.id_to_sign.get(tag_id, UNKNOWN)
            if sign_type == UNKNOWN and not self.discovery_mode:
                continue
            if sign_type == UNKNOWN and tag_id not in self._reported_ids:
                print(f"[SignDetector] DISCOVERY: tag_id={tag_id} (add to signs_real in project_config.yaml)")
                self._reported_ids.add(tag_id)
            pts = quad.reshape(-1, 2)
            h = float(pts[:, 1].max() - pts[:, 1].min())
            if h <= 0:
                continue
            dist = self.focal_px * self.tag_size_m / h
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            out.append(SignObservation(tag_id, sign_type, h, dist, cx, cy, pts))
        return out

    def closest_actionable(self, observations, frame_w=640):
        """The actionable sign (height >= act_px) that is best-centered in the
        frame, broken by size.

        Strategy: discard any sign whose centre-x is outside the middle half of
        the frame (clearly a side-road sign).  Among the remaining, pick the
        largest (closest).  If nothing survives the zone filter, fall back to
        the same zone-then-size logic on the full set."""
        actionable = [o for o in observations if o.height_px >= self.act_px]
        if not actionable:
            return None
        center = frame_w / 2.0
        half   = frame_w / 4.0          # middle-half zone: center ± 25 % of width
        forward = [o for o in actionable if abs(o.cx - center) <= half]
        pool    = forward if forward else actionable
        return max(pool, key=lambda o: o.height_px)

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
            if o.sign_type == UNKNOWN:
                # discovery mode: grey box, just show the raw ID so it can be
                # copied into signs_real in project_config.yaml
                color = (160, 160, 160)
                label = f"UNKNOWN tag_id={o.tag_id}"
            else:
                close = o.height_px >= self.act_px
                color = (0, 215, 255) if not close else (0, 0, 255)
                label = f"{o.sign_type} #{o.tag_id} {o.distance_m:.2f}m"
            cv2.polylines(frame_bgr, [pts], True, color, 2)
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
