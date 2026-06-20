"""Offline self-test for the Traffic Signs agent.

Runs everything that does NOT need the robot, against the recorded camera
frames in tasks/object_detection/dataset/raw. Use it to sanity-check the
perception + decision logic before deploying to the bot:

    env/bin/python tasks/project/packages/test_offline.py

It checks:
  1. AprilTag sign detection over the whole dataset (which IDs / signs appear).
  2. The turn-choice logic for each intersection sign.
  3. Lane following + stop-line detection run without error on real frames.
  4. A no-hardware smoke run of agent.main() with fake camera/wheels/leds.
"""

import os
import sys
import glob
import time
import threading
import collections

import cv2

# make `tasks.project.packages...` importable when run as a script
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tasks.project.packages import agent as agent_mod
from tasks.project.packages.sign_detection import (
    SignDetector, ALLOWED_TURNS, INTERSECTION_SIGNS,
)
from tasks.project.packages.lane_following import LaneFollower
from tasks.project.packages.road_perception import StopLineDetector
from tasks.project.packages.object_detection import ObstacleStopper

_DATASET = os.path.join(_ROOT, "tasks", "object_detection", "dataset", "raw")

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def _frames(limit=None):
    files = sorted(glob.glob(os.path.join(_DATASET, "*.jpg")))
    return files if limit is None else files[:limit]


# ---------------------------------------------------------------------------
def test_sign_detection(cfg):
    print("\n[1] AprilTag sign detection over the dataset")
    det = SignDetector(cfg)
    files = _frames()
    check("dataset present", len(files) > 0, f"{len(files)} frames")

    by_sign = collections.Counter()
    by_tag = collections.Counter()
    for f in files:
        for o in det.detect(cv2.imread(f)):
            by_sign[o.sign_type] += 1
            by_tag[o.tag_id] += 1

    print("      tags seen :", dict(sorted(by_tag.items())))
    print("      signs seen:", dict(by_sign))
    check("at least one sign detected", sum(by_sign.values()) > 0)
    # every configured tag id should be mapped to a sign category
    configured = {int(t) for ids in cfg["signs"].values() for t in ids}
    mapped = set(det.id_to_sign)
    check("all configured tags are mapped", configured == mapped,
          f"configured={sorted(configured)}")


def test_turn_logic(cfg):
    print("\n[2] Turn-choice logic")
    for sign, turns in ALLOWED_TURNS.items():
        check(f"{sign} permits {turns}", len(turns) >= 1)
    # _allowed_turns falls back to straight when no intersection sign is seen
    agent_mod._cfg = cfg
    fallback = agent_mod._allowed_turns([])
    check("no sign -> default straight", fallback == ["straight"], str(fallback))
    # union across multiple signs
    det = SignDetector(cfg)
    obs = []
    for f in _frames():
        obs = det.detect(cv2.imread(f))
        if any(o.is_intersection for o in obs):
            break
    if obs:
        turns = agent_mod._allowed_turns(obs)
        check("intersection frame yields turns", len(turns) >= 1, str(turns))


def test_perception(cfg):
    print("\n[3] Lane following + stop-line on real frames")
    lane = LaneFollower(cfg)
    line = StopLineDetector(cfg)
    n_line = 0
    err = None
    for f in _frames(120):
        img = cv2.imread(f)
        try:
            l, r, dbg = lane.compute(img, cfg["speed"]["cruise"])
            assert -1.0 <= l <= 1.0 and -1.0 <= r <= 1.0
            at, frac = line.detect(img)
            n_line += int(at)
        except Exception as e:
            err = e
            break
    check("lane/line run without error", err is None, str(err) if err else "")
    print(f"      stop-line triggered on {n_line}/120 sampled frames")


def test_obstacle(cfg):
    print("\n[4] Obstacle detector (trained model)")
    stopper = ObstacleStopper(cfg)
    if not cfg.get("obstacle", {}).get("enabled", False):
        check("obstacle disabled in config (skipped)", True)
        return
    check("model loaded", stopper.enabled and stopper._agent is not None,
          stopper.load_error or "")
    if not stopper.enabled:
        return
    # run detection + should_stop on a duckie frame to confirm the pipeline runs
    size = stopper._agent.img_size
    err = None
    fired = 0
    for f in _frames(40):
        try:
            rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            dets = stopper._agent.detect(cv2.resize(rgb, (size, size)))
            if dets is None:
                continue
            blocked, _ = stopper._should_stop(dets, size)
            fired += int(blocked)
        except Exception as e:
            err = e
            break
    check("detect + should_stop run without error", err is None, str(err) if err else "")
    print(f"      would-stop on {fired}/40 sampled frames")


def test_smoke_run(cfg):
    print("\n[5] No-hardware smoke run of agent.main()")
    cam = _FakeCamera(_frames())
    wheels = _FakeWheels()
    leds = _FakeLeds()
    stop = threading.Event()

    t = threading.Thread(target=agent_mod.main, args=(cam, wheels, leds, stop))
    t.start()
    time.sleep(2.5)
    stop.set()
    t.join(timeout=5)

    check("agent thread exited cleanly", not t.is_alive())
    check("wheels received commands", wheels.calls > 0, f"{wheels.calls} calls")
    check("motors stopped on exit", wheels.last == (0.0, 0.0), str(wheels.last))
    check("status is reachable", "state" in agent_mod.get_status(),
          str(agent_mod.get_status()))
    # live-tuning a config value via the dashboard command path
    agent_mod.apply_command("speed.cruise", "0.4")
    check("apply_command updates config",
          abs(agent_mod._cfg["speed"]["cruise"] - 0.4) < 1e-9)


# ---- fakes -----------------------------------------------------------------
class _FakeCamera:
    def __init__(self, files):
        self.files = files or []
        self.i = 0
        self.frame_count = 0
        self.resolution = (640, 480)

    def read(self):
        if not self.files:
            return False, None
        img = cv2.imread(self.files[self.i % len(self.files)])
        self.i += 1
        self.frame_count += 1
        return (img is not None), img


class _FakeWheels:
    def __init__(self):
        self.calls = 0
        self.last = None

    def set_wheels_speed(self, left, right):
        self.calls += 1
        self.last = (float(left), float(right))


class _FakeLeds:
    def set_rgb(self, idx, rgb):
        pass

    def all_off(self):
        pass


# ---------------------------------------------------------------------------
def main():
    cfg = agent_mod._load_config()
    if not cfg:
        print("ERROR: could not load config/project_config.yaml")
        return 1
    print("=" * 64)
    print("Traffic Signs agent - offline self-test")
    print("=" * 64)
    test_sign_detection(cfg)
    test_turn_logic(cfg)
    test_perception(cfg)
    test_obstacle(cfg)
    test_smoke_run(cfg)
    print("\n" + "=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
