"""Standalone AprilTag diagnostic — run ON THE BOT (Jetson) to find out why signs
aren't being detected.

    python3 tools/check_apriltags.py

It checks three things in order and prints a clear verdict:
  1. Is cv2.aruco available at all, and does DICT_APRILTAG_36h11 exist?
  2. Can the camera grab a frame?
  3. Running detection on that frame — how many tags, and which IDs?

Hold a sign ~30-50cm in front of the camera while it runs.
"""

import sys
import time

import cv2
import numpy as np

print("=" * 60)
print("APRILTAG DIAGNOSTIC")
print("=" * 60)
print(f"OpenCV version: {cv2.__version__}")

# ---- 1. pick an AprilTag backend (aruco, or the apriltag library) ----------
backend = None
detector = None
dictionary = params = None
apriltag_det = None

if hasattr(cv2, "aruco"):
    print("OK: cv2.aruco is available")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        backend = "aruco_new"
        print("OK: using NEW aruco API (ArucoDetector)")
    except Exception:
        try:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            params = cv2.aruco.DetectorParameters_create()
            backend = "aruco_legacy"
            print("OK: using LEGACY aruco API (Dictionary_get)")
        except Exception:
            print("WARN: cv2.aruco lacks DICT_APRILTAG_36h11")
else:
    print("WARN: cv2.aruco is NOT in this OpenCV build")

if backend is None:
    try:
        try:
            from pupil_apriltags import Detector
            libname = "pupil_apriltags"
        except ImportError:
            from dt_apriltags import Detector
            libname = "dt_apriltags"
        apriltag_det = Detector(families="tag36h11", nthreads=2)
        backend = "apriltag_lib"
        print(f"OK: using apriltag library ({libname})")
    except ImportError:
        print("\nFAIL: no AprilTag backend available (no aruco, no pupil_apriltags/dt_apriltags).")
        print("  -> install one on the bot:  pip3 install pupil-apriltags")
        sys.exit(1)

# ---- 2. camera -------------------------------------------------------------
sys.path.insert(0, ".")
try:
    from duckiebot.camera_driver.camera_driver import CameraDriver
    cam = CameraDriver()
    cam.start()
    print("OK: camera started")
except Exception as e:
    print(f"\nFAIL: could not start camera: {e}")
    sys.exit(1)

time.sleep(1.0)  # let the capture thread fill a frame

# ---- 3. detection loop -----------------------------------------------------
print("\nWatching for tags for 20s — hold a sign in front of the camera...")
seen = set()
end = time.time() + 20.0
frames = 0
while time.time() < end:
    ok, frame = cam.read()
    if not ok or frame is None:
        continue
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tag_ids = []
    if backend == "apriltag_lib":
        tag_ids = [int(r.tag_id) for r in apriltag_det.detect(gray)]
    elif backend == "aruco_legacy":
        _, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        tag_ids = [int(t) for t in ids.flatten()] if ids is not None else []
    else:
        _, ids, _ = detector.detectMarkers(gray)
        tag_ids = [int(t) for t in ids.flatten()] if ids is not None else []
    for tid in tag_ids:
        if tid not in seen:
            seen.add(tid)
            print(f"  >>> DETECTED tag_id = {tid}")
    time.sleep(0.05)

cam.stop()
print("\n" + "=" * 60)
print(f"Processed {frames} frames.")
if seen:
    print(f"VERDICT: detection WORKS. Tag IDs seen: {sorted(seen)}")
    print("  -> put these into signs_real in config/project_config.yaml")
else:
    print("VERDICT: aruco works but NO tags were detected.")
    print("  Likely causes: tag too far/small, blurry, bad lighting, or the")
    print("  signs aren't the tag36h11 family. Try holding a sign closer/steadier.")
print("=" * 60)
