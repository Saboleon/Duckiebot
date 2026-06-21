"""Traffic Signs project - autonomous agent.

The robot follows the lane and reacts to traffic signs read from their
AprilTags:

  * STOP sign   -> come to a full stop, hold, give right-of-way, then go.
  * YIELD sign  -> brief slow-down before proceeding.
  * pedestrian  -> slow / stop if something is in the way, then continue.
  * intersection signs (side-road-left/right, T-junction) -> at the
    intersection pick ONE of the turns the sign permits, at random, and drive it.
  * obstacle ahead (optional) -> stop until it clears.

Behaviour is a small state machine. All numbers come from
config/project_config.yaml so they can be tuned live from the dashboard.

Entry point (called once by the server on its own thread):

    def main(camera, wheels, leds, stop_event): ...
"""

import os
import time
import random
import threading

import cv2
import yaml

from tasks.project.packages.sign_detection import (
    SignDetector, STOP, YIELD, PEDESTRIAN, LEFT, RIGHT, STRAIGHT,
)
from tasks.project.packages.lane_following import LaneFollower
from tasks.project.packages.road_perception import StopLineDetector
from tasks.project.packages.object_detection import ObstacleStopper

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml"
))

# ---- shared state, read by the server for /status and the video overlay -----
_lock = threading.Lock()
_status = {"state": "init"}
_overlay = None          # latest annotated BGR frame
_cfg = {}                # live config (mutable via apply_command)
_paused = False          # when True the agent holds still (sim convenience)
_force_turn = None       # set to "left"/"right"/"straight" to trigger one turn


def get_status():
    with _lock:
        return dict(_status)


def get_overlay():
    with _lock:
        return None if _overlay is None else _overlay.copy()


def set_paused(value):
    global _paused
    _paused = bool(value)
    return _paused


def apply_command(key, value):
    """Live-tune config / control the agent from the dashboard.

    Control keys: 'pause'/'resume' (hold or release the motors) and
    'force_turn' = left|right|straight (manually trigger one turn maneuver -
    handy in the simulator, which has no AprilTags to react to).
    Anything else is treated as a dotted config path, e.g.
    'speed.cruise' -> 0.4, or 'obstacle.enabled' -> true."""
    global _force_turn
    k = key.strip().lower()
    if k in ("pause", "stop_driving"):
        return f"paused = {set_paused(True)}"
    if k in ("resume", "go"):
        return f"paused = {set_paused(False)}"
    if k == "force_turn":
        turn = str(value).strip().lower()
        if turn not in (LEFT, RIGHT, STRAIGHT):
            raise ValueError("force_turn must be left, right or straight")
        _force_turn = turn
        return f"force_turn queued: {turn}"

    parsed = _parse_value(value)
    node = _cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = parsed
    return f"{key} = {parsed!r}"


# ============================================================================
def main(camera, wheels, leds, stop_event):
    global _cfg
    _cfg = _load_config()

    signs    = SignDetector(_cfg)
    lane     = LaneFollower(_cfg)
    stopline = StopLineDetector(_cfg)
    obstacle = ObstacleStopper(_cfg)
    obstacle.start(camera)

    leds_ctl = _Leds(leds)
    random.seed()

    pending_turn = None           # turn queued (by a sign or a button), run at the red line
    pending_src = None            # "sign" or "command" - for the status display
    pending_since = 0.0           # when pending_turn was set (for no-red-line fallback)
    stop_sign_since = 0.0         # when a STOP sign was first seen at act distance
    yield_sign_since = 0.0        # when a YIELD sign was first seen at act distance
    cooldown_until = 0.0          # ignore the stop line again until this time
    frame_i = 0

    _set_status(state="cruise", note="started")
    print("[project] traffic-signs agent running")

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_i += 1

            # paused: hold still but keep the video/overlay alive
            if _paused:
                _drive(wheels, 0.0, 0.0)
                leds_ctl.off()
                _set_status(state="paused")
                _annotate(signs, lane, frame, [], {}, "paused")
                time.sleep(0.03)
                continue

            cfg_speed = _cfg.get("speed", {})
            cruise_v   = float(cfg_speed.get("cruise", 0.32))
            approach_v = float(cfg_speed.get("approach", 0.18))

            every = max(1, int(_cfg.get("detection", {}).get("every_n_frames", 1)))
            observations = signs.detect(frame) if (frame_i % every == 0) else []

            blocked, block_reason = obstacle.status()
            at_line, line_frac = stopline.detect(frame)
            now = time.time()

            # ---- obstacle overrides everything ----
            if blocked:
                leds_ctl.hazard()
                _drive(wheels, 0.0, 0.0)
                _set_status(state="obstacle", note=block_reason or "obstacle ahead")
                _annotate(signs, lane, frame, observations, {}, "obstacle: duckie ahead")
                continue

            # ---- queue a turn from a button press (force_turn command) ----
            if _force_turn is not None:
                pending_turn = _consume_force_turn()
                pending_src = "command"
                pending_since = now

            # ---- queue a turn from a traffic sign (real bot) ----
            # an intersection sign seen before the line decides which way to go
            if pending_turn is None and now >= cooldown_until:
                sign = signs.closest_actionable(observations)
                if sign is not None and sign.is_intersection:
                    pending_turn = random.choice(sign.allowed_turns)
                    pending_src = "sign"
                    pending_since = now
                    print(f"[project] {sign.sign_type} sign -> queued turn: {pending_turn}")

            # ---- track STOP/YIELD proximity for no-red-line fallback ----
            close_types = {o.sign_type for o in observations if o.height_px >= signs.act_px}
            if STOP in close_types:
                if stop_sign_since == 0.0:
                    stop_sign_since = now
            else:
                stop_sign_since = 0.0
            if YIELD in close_types:
                if yield_sign_since == 0.0:
                    yield_sign_since = now
            else:
                yield_sign_since = 0.0

            # ---- the red stop line is the trigger: execute the queued turn here ----
            if at_line and now >= cooldown_until:
                _drive(wheels, 0.0, 0.0)
                handled = _handle_intersection(
                    wheels, leds_ctl, observations, pending_turn, stop_event)
                pending_turn = None
                pending_src = None
                pending_since = 0.0
                stop_sign_since = 0.0
                yield_sign_since = 0.0
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last=handled)
                continue

            # ---- no-red-line fallbacks: act on sign proximity alone ----
            sign_timeout = float(_cfg.get("sign_timeout_s", 4.0))

            if stop_sign_since > 0.0 and now - stop_sign_since > sign_timeout and now >= cooldown_until:
                print("[project] STOP sign timeout (no red line) -> stopping")
                _drive(wheels, 0.0, 0.0)
                leds_ctl.stop()
                _set_status(state="stop", note="STOP sign (no red line)")
                timing = _cfg.get("timing", {})
                _wait(stop_event, float(timing.get("stop_dwell", 2.0)))
                _wait(stop_event, float(timing.get("clear_time", 2.0)))
                stop_sign_since = 0.0
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last="stop(no-line)")
                continue

            if yield_sign_since > 0.0 and now - yield_sign_since > sign_timeout and now >= cooldown_until:
                print("[project] YIELD sign timeout (no red line) -> yielding")
                _drive(wheels, 0.0, 0.0)
                leds_ctl.yield_()
                _set_status(state="yield", note="YIELD sign (no red line)")
                timing = _cfg.get("timing", {})
                _wait(stop_event, float(timing.get("yield_dwell", 0.6)))
                yield_sign_since = 0.0
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last="yield(no-line)")
                continue

            if pending_turn is not None and pending_since > 0.0 and now - pending_since > sign_timeout and now >= cooldown_until:
                print(f"[project] intersection timeout (no red line) -> turn: {pending_turn}")
                _drive(wheels, 0.0, 0.0)
                handled = _handle_intersection(wheels, leds_ctl, observations, pending_turn, stop_event)
                pending_turn = None
                pending_src = None
                pending_since = 0.0
                stop_sign_since = 0.0
                yield_sign_since = 0.0
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last=f"{handled}(no-line)")
                continue

            # ---- otherwise just follow the lane (slower if a turn is pending) ----
            base = approach_v if pending_turn is not None else cruise_v
            left, right, dbg = lane.compute(frame, base)
            _drive(wheels, left, right)
            leds_ctl.cruise()

            trig = stopline.trigger
            line_txt = f"line={line_frac:.3f}/{trig:.3f}"
            if pending_turn is not None:
                _set_status(state="approach", turn=pending_turn, src=pending_src,
                            line=round(line_frac, 3), **dbg)
                _annotate(signs, lane, frame, observations, dbg,
                          f"approach -> {pending_turn}  {line_txt}")
            else:
                _set_status(state="cruise", line=round(line_frac, 3), **dbg)
                _annotate(signs, lane, frame, observations, dbg, f"cruise  {line_txt}")

    except Exception as e:
        print(f"[project] agent error: {e}")
    finally:
        obstacle.stop()
        _drive(wheels, 0.0, 0.0)
        leds_ctl.off()
        _set_status(state="stopped")
        print("[project] agent stopped, motors off")


# ----------------------------------------------------------------------------
def _handle_intersection(wheels, leds_ctl, observations, pending_turn, stop_event):
    """We have reached the red stop line. Apply stop/yield right-of-way, then
    execute the queued turn (from a sign or a button). Returns a short
    description of what we did."""
    timing = _cfg.get("timing", {})

    # what signs do we see right now?
    types = {o.sign_type for o in observations}

    # --- right-of-way gating ---
    if STOP in types:
        leds_ctl.stop()
        print("[project] STOP sign: full stop")
        if not _wait(stop_event, float(timing.get("stop_dwell", 2.0))):
            return "stop(interrupted)"
        # give way to crossing traffic ("from the right has precedence")
        if not _wait(stop_event, float(timing.get("clear_time", 2.0))):
            return "stop(interrupted)"
    elif YIELD in types:
        leds_ctl.yield_()
        print("[project] YIELD sign: slowing")
        if not _wait(stop_event, float(timing.get("yield_dwell", 0.6))):
            return "yield(interrupted)"
    elif PEDESTRIAN in types and pending_turn is None:
        leds_ctl.yield_()
        print("[project] pedestrian crossing: pausing")
        if not _wait(stop_event, float(timing.get("yield_dwell", 0.6))):
            return "pedestrian(interrupted)"
        return "pedestrian"        # not an intersection: just continue cruising

    # --- the turn: use the queued one, else pick a random allowed/straight ---
    turn = pending_turn or random.choice(_allowed_turns(observations))
    print(f"[project] stop line reached -> turn: {turn}")
    _execute_turn(wheels, leds_ctl, turn, stop_event)
    return f"turn:{turn}"


def _allowed_turns(observations):
    """Union of turns permitted by any intersection sign in view; default
    to straight if none is present."""
    turns = []
    for o in observations:
        for t in o.allowed_turns:
            if t not in turns:
                turns.append(t)
    return turns or [STRAIGHT]


def _execute_turn(wheels, leds_ctl, turn, stop_event):
    """Open-loop timed maneuver - a gradual forward arc, not a pivot.

    Each direction (left/right/straight) has its own speed, sharpness and
    duration so they can be tuned independently: a left turn is a wider, deeper
    arc (crosses to the far lane), a right turn is tighter. Both wheels keep
    moving forward; `sharpness` is the small speed difference between them
    (radius ~ speed/sharpness, angle ~ sharpness x duration)."""
    tcfg = _cfg.get("turn", {})
    creep_s = float(tcfg.get("creep_s", 0.5))

    p = tcfg.get(turn, {}) or {}
    speed     = float(p.get("speed", 0.30))
    duration  = float(p.get("duration", 1.8))
    sharpness = float(p.get("sharpness", 0.05))   # unused for straight

    # ease straight into the intersection first
    leds_ctl.cruise()
    _drive(wheels, speed, speed)
    if not _wait(stop_event, creep_s):
        _drive(wheels, 0.0, 0.0)
        return

    if turn == LEFT:
        leds_ctl.signal_left()
        _drive(wheels, speed - sharpness, speed + sharpness)
    elif turn == RIGHT:
        leds_ctl.signal_right()
        _drive(wheels, speed + sharpness, speed - sharpness)
    else:  # straight
        leds_ctl.cruise()
        _drive(wheels, speed, speed)

    _wait(stop_event, duration)
    _drive(wheels, 0.0, 0.0)


# ---- LED signaling ---------------------------------------------------------
class _Leds:
    """Thin wrapper that tolerates leds == None and ignores hardware errors."""
    FL, FR, BL, BR = 0, 2, 3, 4

    def __init__(self, leds):
        self.leds = leds

    def _set(self, idx, rgb):
        if not self.leds:
            return
        try:
            self.leds.set_rgb(idx, rgb)
        except Exception:
            pass

    def off(self):
        for i in (self.FL, self.FR, self.BL, self.BR):
            self._set(i, [0, 0, 0])

    def cruise(self):
        self._set(self.FL, [0.4, 0.4, 0.4]); self._set(self.FR, [0.4, 0.4, 0.4])
        self._set(self.BL, [0.2, 0, 0]);     self._set(self.BR, [0.2, 0, 0])

    def stop(self):
        for i in (self.FL, self.FR, self.BL, self.BR):
            self._set(i, [1, 0, 0])

    def yield_(self):
        for i in (self.FL, self.FR, self.BL, self.BR):
            self._set(i, [1, 0.5, 0])

    def hazard(self):
        for i in (self.FL, self.FR, self.BL, self.BR):
            self._set(i, [1, 0.4, 0])

    def signal_left(self):
        self.off()
        self._set(self.FL, [1, 0.5, 0]); self._set(self.BL, [1, 0.5, 0])

    def signal_right(self):
        self.off()
        self._set(self.FR, [1, 0.5, 0]); self._set(self.BR, [1, 0.5, 0])


# ---- small helpers ---------------------------------------------------------
def _drive(wheels, left, right):
    try:
        wheels.set_wheels_speed(left, right)
    except Exception as e:
        print(f"[project] wheel error: {e}")


def _wait(stop_event, secs):
    """Sleep up to `secs`, returning False if a stop was requested."""
    end = time.time() + secs
    while time.time() < end:
        if stop_event.is_set():
            return False
        time.sleep(0.02)
    return not stop_event.is_set()


def _set_status(**kw):
    with _lock:
        _status.clear()
        _status.update(kw)


def _annotate(signs, lane, frame, observations, dbg, state):
    global _overlay
    img = frame.copy()
    signs.draw(img, observations)
    if dbg:
        lane.draw(img, dbg)
    cv2.putText(img, f"state: {state}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    with _lock:
        _overlay = img


def _consume_force_turn():
    global _force_turn
    turn = _force_turn
    _force_turn = None
    return turn


def _parse_value(value):
    if isinstance(value, (int, float, bool)):
        return value
    s = str(value).strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _load_config(path=None):
    path = path or _CONFIG_FILE
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[project] could not load config ({e}); using defaults")
        return {}
