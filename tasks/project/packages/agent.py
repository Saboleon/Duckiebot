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
    SignDetector, STOP, YIELD, PEDESTRIAN, PARKING, YIELD_SIGNS,
    LEFT, RIGHT, STRAIGHT,
)
from tasks.project.packages.lane_following import LaneFollower
from tasks.project.packages.road_perception import StopLineDetector
from tasks.project.packages.object_detection import ObstacleStopper

_CONFIG_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config"))
_CONFIG_FILE      = os.path.join(_CONFIG_DIR, "project_config.yaml")
_CONFIG_FILE_SIM  = os.path.join(_CONFIG_DIR, "project_config_sim.yaml")

# ---- shared state, read by the server for /status and the video overlay -----
_lock = threading.Lock()
_status = {"state": "init"}
_overlay = None          # latest annotated BGR frame
_overlay_ts = 0.0        # when _overlay was last refreshed (for staleness checks)
_obstacle = None         # ObstacleStopper ref, so the overlay can draw detections
_cfg = {}                # live config (mutable via apply_command)
_paused = False          # when True the agent holds still (sim convenience)
_force_turn = None       # set to "left"/"right"/"straight" to trigger one turn
_manual_wheels = None    # (left, right) set by 'drive' command; None = autonomous
_discovered_ids = {}     # {tag_id: sign_type} seen in discovery mode — shown in status


def get_status():
    with _lock:
        st = dict(_status)
    if _discovered_ids:
        st["discovered_sign_ids"] = sorted(_discovered_ids.keys())
    return st


def get_overlay(max_age=None):
    """Latest annotated frame, or None. With max_age set, also returns None when
    the overlay is older than max_age seconds - the control loop stops refreshing
    it during blocking maneuvers (turns, yield/stop waits), so the video stream
    can fall back to the live camera frame instead of freezing on a stale image."""
    with _lock:
        if _overlay is None:
            return None
        if max_age is not None and (time.time() - _overlay_ts) > max_age:
            return None
        return _overlay.copy()


def set_paused(value):
    global _paused
    _paused = bool(value)
    return _paused


def apply_command(key, value):
    """Live-tune config / control the agent from the dashboard.

    Control keys: 'pause'/'resume' (hold or release the motors) and
    'force_turn' = left|right|straight (manually trigger one turn maneuver -
    handy in the simulator, which has no AprilTags to react to).
    Manual drive: 'drive' = 'left,right' (e.g. '0.3,0.3') sets wheel speeds
    directly; 'drive_stop' clears manual drive and returns to autonomous.
    Anything else is treated as a dotted config path, e.g.
    'speed.cruise' -> 0.4, or 'obstacle.enabled' -> true."""
    global _force_turn, _manual_wheels
    k = key.strip().lower()
    if k in ("pause", "stop_driving"):
        return f"paused = {set_paused(True)}"
    if k in ("resume", "go"):
        _manual_wheels = None   # exit manual drive
        _force_turn    = None   # cancel any queued turn
        return f"paused = {set_paused(False)}"
    if k == "force_turn":
        turn = str(value).strip().lower()
        if turn not in (LEFT, RIGHT, STRAIGHT):
            raise ValueError("force_turn must be left, right or straight")
        _force_turn = turn
        _manual_wheels = None   # exit manual mode so the turn actually executes
        return f"force_turn: {turn}"
    if k == "drive":
        parts = str(value).split(',')
        if len(parts) == 2:
            l = max(-1.0, min(1.0, float(parts[0])))
            r = max(-1.0, min(1.0, float(parts[1])))
            _manual_wheels = (l, r)
            return f"manual drive: l={l:.2f} r={r:.2f}"
        raise ValueError("drive value must be 'left,right' e.g. '0.3,0.3'")
    if k in ("drive_stop", "manual_stop"):
        _manual_wheels = (0.0, 0.0)   # stay stopped in manual mode; 'resume' exits
        return "manual stopped"

    parsed = _parse_value(value)
    node = _cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = parsed
    return f"{key} = {parsed!r}"


# ============================================================================
def main(camera, wheels, leds, stop_event, sim=False):
    global _cfg, _obstacle
    _cfg = _load_config(_CONFIG_FILE_SIM if sim else _CONFIG_FILE)

    _set_status(state="init", note="loading signs")
    try:
        signs = SignDetector(_cfg, sim=sim)
    except Exception as e:
        print(f"[project] SignDetector failed: {e}")
        _set_status(state="init", note=f"signs error: {e}")
        signs = _DummySigns()

    _set_status(state="init", note="loading lane follower")
    try:
        lane = LaneFollower(_cfg, sim=sim)
    except Exception as e:
        print(f"[project] LaneFollower failed: {e}")
        _set_status(state="error", note=f"lane error: {e}")
        return

    _set_status(state="init", note="loading stopline detector")
    try:
        stopline = StopLineDetector(_cfg)
    except Exception as e:
        print(f"[project] StopLineDetector failed: {e}")
        stopline = _DummyStopline()

    _set_status(state="init", note="loading obstacle detector")
    try:
        obstacle = ObstacleStopper(_cfg)
    except Exception as e:
        print(f"[project] ObstacleStopper failed: {e}")
        obstacle = _DummyObstacle()
    _obstacle = obstacle

    _set_status(state="init", note="starting obstacle thread")
    try:
        obstacle.start(camera)
    except Exception as e:
        print(f"[project] obstacle.start failed: {e}")

    leds_ctl = _Leds(leds)
    random.seed()

    pending_turn = None           # turn queued (by a sign or a button), run at the red line
    pending_src = None            # "sign" or "command" - for the status display
    pending_since = 0.0           # when pending_turn was set (for no-red-line fallback)
    pending_gate = None           # sign category that queued the turn (e.g. yield_left),
                                  # so we still yield/stop even after the sign leaves view
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

            # manual drive: bypass the autonomous agent entirely
            if _manual_wheels is not None:
                l, r = _manual_wheels
                _drive(wheels, l, r)
                leds_ctl.off()
                _set_status(state="manual", left=round(l, 2), right=round(r, 2))
                _annotate(signs, lane, frame, [], {}, f"manual  L={l:.2f} R={r:.2f}")
                time.sleep(0.02)
                continue

            # paused: hold still but still run sign detection so IDs can be discovered
            if _paused:
                _drive(wheels, 0.0, 0.0)
                leds_ctl.off()
                obs_paused = signs.detect(frame)
                for o in obs_paused:
                    if o.sign_type == "unknown" and o.tag_id not in _discovered_ids:
                        _discovered_ids[o.tag_id] = "unknown"
                        print(f"[project] SIGN DISCOVERED: tag_id={o.tag_id}")
                _set_status(state="paused")
                _annotate(signs, lane, frame, obs_paused, {}, "paused")
                time.sleep(0.03)
                continue

            cfg_speed = _cfg.get("speed", {})
            cruise_v   = float(cfg_speed.get("cruise", 0.32))
            approach_v = float(cfg_speed.get("approach", 0.18))

            every = max(1, int(_cfg.get("detection", {}).get("every_n_frames", 1)))
            observations = signs.detect(frame) if (frame_i % every == 0) else []

            # accumulate discovered IDs (unknown tags seen in discovery mode)
            for obs in observations:
                if obs.sign_type == "unknown" and obs.tag_id not in _discovered_ids:
                    _discovered_ids[obs.tag_id] = "unknown"
                    print(f"[project] SIGN DISCOVERED: tag_id={obs.tag_id}")

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

            # ---- force_turn command: execute immediately (don't wait for red line) ----
            if _force_turn is not None:
                forced = _consume_force_turn()
                print(f"[project] force_turn: {forced} — executing now")
                _drive(wheels, 0.0, 0.0)
                _execute_turn(wheels, leds_ctl, obstacle, forced, stop_event)
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last=f"manual:{forced}")
                continue

            # ---- queue a turn from a traffic sign (real bot) ----
            # an intersection sign seen before the line decides which way to go
            if pending_turn is None and now >= cooldown_until:
                sign = signs.closest_actionable(observations)
                if sign is not None and sign.is_intersection:
                    pending_turn = random.choice(sign.allowed_turns)
                    pending_src = "sign"
                    pending_since = now
                    pending_gate = sign.sign_type   # remember it (e.g. yield_left)
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

            # ---- parking sign: this is the destination. Come to a full stop
            #      and stay parked (hold still). Send 'resume' from the
            #      dashboard if you want to drive on afterwards. ----
            if PARKING in close_types and now >= cooldown_until:
                print("[project] PARKING sign -> parked (full stop)")
                leds_ctl.stop()
                _drive(wheels, 0.0, 0.0)
                _annotate(signs, lane, frame, observations, {}, "parked")
                _set_status(state="parked", note="parking sign")
                set_paused(True)
                continue

            # ---- the red stop line is the trigger: execute the queued turn here ----
            if at_line and now >= cooldown_until:
                _drive(wheels, 0.0, 0.0)
                handled = _handle_intersection(
                    wheels, leds_ctl, obstacle, observations, pending_turn, pending_gate, stop_event)
                pending_turn = None
                pending_src = None
                pending_since = 0.0
                pending_gate = None
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
                _wait_for_clear(obstacle, leds_ctl, stop_event, timing)
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
                if obstacle.traffic_present():
                    _wait_for_clear(obstacle, leds_ctl, stop_event, timing)
                else:
                    _wait(stop_event, float(timing.get("yield_dwell", 0.6)))
                yield_sign_since = 0.0
                cooldown_until = time.time() + 4.0
                _set_status(state="cruise", last="yield(no-line)")
                continue

            if pending_turn is not None and pending_since > 0.0 and now - pending_since > sign_timeout and now >= cooldown_until:
                print(f"[project] intersection timeout (no red line) -> turn: {pending_turn}")
                _drive(wheels, 0.0, 0.0)
                handled = _handle_intersection(wheels, leds_ctl, obstacle, observations, pending_turn, pending_gate, stop_event)
                pending_turn = None
                pending_src = None
                pending_since = 0.0
                pending_gate = None
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
def _handle_intersection(wheels, leds_ctl, obstacle, observations, pending_turn, pending_gate, stop_event):
    """We have reached the red stop line. Apply stop/yield right-of-way, then
    execute the queued turn (from a sign or a button). Returns a short
    description of what we did."""
    timing = _cfg.get("timing", {})

    # signs we see right now, plus the category that queued this turn. The gate
    # matters because a combined sign (e.g. yield_left) is usually behind us by
    # the time we reach the line, so it won't be in `observations` anymore.
    types = {o.sign_type for o in observations}
    if pending_gate:
        types.add(pending_gate)

    # --- right-of-way gating ---
    if STOP in types:
        leds_ctl.stop()
        print("[project] STOP sign: full stop")
        if not _wait(stop_event, float(timing.get("stop_dwell", 2.0))):
            return "stop(interrupted)"
        # right-of-way: hold until crossing traffic has actually cleared
        if _wait_for_clear(obstacle, leds_ctl, stop_event, timing) == "interrupted":
            return "stop(interrupted)"
    elif types & YIELD_SIGNS:
        leds_ctl.yield_()
        if obstacle.traffic_present():
            # there IS crossing traffic -> give way until the lane is clear
            print("[project] YIELD: crossing traffic -> holding")
            if _wait_for_clear(obstacle, leds_ctl, stop_event, timing) == "interrupted":
                return "yield(interrupted)"
        else:
            # clear -> just a brief slow-down, then proceed
            print("[project] YIELD: clear -> brief slow")
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
    _execute_turn(wheels, leds_ctl, obstacle, turn, stop_event)
    return f"turn:{turn}"


def _wait_for_clear(obstacle, leds_ctl, stop_event, timing):
    """Right-of-way: stay stopped (amber LEDs) until no crossing traffic has been
    seen for `clear_time` continuously, or until `yield_max_wait` elapses (a
    safety cap so we never hang). Returns 'clear', 'timeout', or 'interrupted'.

    Relies on the camera-based detector via obstacle.traffic_present(); if that's
    unavailable it returns clear immediately, so behaviour degrades to the plain
    dwell the caller already applied."""
    clear_needed = float(timing.get("clear_time", 1.0))
    max_wait     = float(timing.get("yield_max_wait", 8.0))
    start = time.time()
    clear_since = None
    leds_ctl.yield_()
    while not stop_event.is_set():
        if time.time() - start > max_wait:
            print("[project] right-of-way wait timed out -> proceeding")
            return "timeout"
        if obstacle.traffic_present():
            clear_since = None                 # traffic in view -> reset the clear timer
        elif clear_since is None:
            clear_since = time.time()          # just went clear; start counting
        elif time.time() - clear_since >= clear_needed:
            return "clear"                     # clear long enough -> go
        time.sleep(0.05)
    return "interrupted"


def _allowed_turns(observations):
    """Union of turns permitted by any intersection sign in view; default
    to straight if none is present."""
    turns = []
    for o in observations:
        for t in o.allowed_turns:
            if t not in turns:
                turns.append(t)
    return turns or [STRAIGHT]


def _drive_for(wheels, left, right, secs, stop_event, obstacle, leds_ctl, on_move=None):
    """Drive at (left, right) for `secs` of actual MOVING time.
    Pauses for obstacles ahead; aborts if a manual drive command arrives."""
    remaining = float(secs)
    was_blocked = True
    while remaining > 0.0:
        if stop_event.is_set():
            _drive(wheels, 0.0, 0.0)
            return False
        if _manual_wheels is not None:   # manual override → abort turn
            _drive(wheels, 0.0, 0.0)
            return False
        blocked, _ = obstacle.status()
        if blocked:
            if not was_blocked:
                _drive(wheels, 0.0, 0.0)
                leds_ctl.hazard()
            was_blocked = True
            time.sleep(0.05)
            continue
        if was_blocked:
            if on_move:
                on_move()
            _drive(wheels, left, right)
            was_blocked = False
        time.sleep(0.05)
        remaining -= 0.05
    _drive(wheels, 0.0, 0.0)
    return True


def _execute_turn(wheels, leds_ctl, obstacle, turn, stop_event):
    """Open-loop timed maneuver - a gradual forward arc, not a pivot.

    Each direction (left/right/straight) has its own speed, sharpness and
    duration so they can be tuned independently: a left turn is a wider, deeper
    arc (crosses to the far lane), a right turn is tighter. Both wheels keep
    moving forward; `sharpness` is the small speed difference between them
    (radius ~ speed/sharpness, angle ~ sharpness x duration).

    The maneuver pauses for obstacles ahead (via _drive_for) so a duck that
    appears mid-turn freezes the bot instead of being run over."""
    tcfg = _cfg.get("turn", {})

    p = tcfg.get(turn, {}) or {}
    speed     = float(p.get("speed", 0.30))
    duration  = float(p.get("duration", 1.8))
    sharpness = float(p.get("sharpness", 0.05))   # unused for straight
    creep_s   = float(p.get("creep_s", tcfg.get("creep_s", 0.5)))

    # ease into the intersection first; left turn gets a slight left lean during
    # creep to counteract the rightward drift that builds up at equal wheel speeds
    creep_lean = float(p.get("creep_lean", 0.0))
    if not _drive_for(wheels, speed - creep_lean, speed + creep_lean, creep_s,
                      stop_event, obstacle, leds_ctl, on_move=leds_ctl.cruise):
        return

    if turn == LEFT:
        l, r, on_move = speed - sharpness, speed + sharpness, leds_ctl.signal_left
    elif turn == RIGHT:
        l, r, on_move = speed + sharpness, speed - sharpness, leds_ctl.signal_right
    else:  # straight
        l, r, on_move = speed, speed, leds_ctl.cruise

    _drive_for(wheels, l, r, duration, stop_event, obstacle, leds_ctl, on_move=on_move)
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
    global _overlay, _overlay_ts
    img = frame.copy()
    signs.draw(img, observations)
    if _obstacle is not None:
        _obstacle.draw(img)        # detection boxes + traffic watch-band
    if dbg:
        lane.draw(img, dbg)
    cv2.putText(img, f"state: {state}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    with _lock:
        _overlay = img
        _overlay_ts = time.time()


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


class _DummySigns:
    act_px = 999
    def detect(self, frame): return []
    def closest_actionable(self, obs, **kw): return None
    def in_view(self, obs): return None
    def draw(self, frame, obs): return frame

class _DummyStopline:
    trigger = 0.0
    def detect(self, frame): return False, 0.0

class _DummyObstacle:
    blocked = False
    reason = ""
    enabled = False
    def start(self, camera): pass
    def status(self): return False, ""
    def traffic_present(self): return False
    def draw(self, frame): return frame
    def stop(self): pass
