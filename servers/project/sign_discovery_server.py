"""Standalone SIGN-DISCOVERY server — camera feed + AprilTag detection ONLY.

Run this directly on the bot when you just want to find sign tag IDs, without
the full driving agent (no wheels, no lane following, no ONNX obstacle model).
Nothing here touches real_server.py or the agent, so it can't break anything.

    python3 servers/project/sign_discovery_server.py
    # then open  http://glados.local:5000  (or the printed port)

The page shows the live camera with a box on every AprilTag it sees and a running
list of discovered tag IDs. Hold a sign in front of the camera; note the IDs, then
put them into signs_real in config/project_config.yaml.
"""

import os
import sys
import time
import threading

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response

from duckiebot.camera_driver import CameraDriver
from launcher.ports import find_available_port
from tasks.project.packages.sign_detection import SignDetector, _load_config

app = Flask(__name__)

_camera   = None
_signs    = None
_overlay  = None
_lock     = threading.Lock()
_seen_ids = {}          # tag_id -> first-seen time
_running  = True
_backend  = "?"


def _detect_loop():
    """Background: grab frames, detect tags, draw boxes, remember IDs."""
    global _overlay
    while _running:
        ok, frame = _camera.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        try:
            observations = _signs.detect(frame)
        except Exception as e:
            observations = []
            print(f"[discovery] detect error: {e}")
        for o in observations:
            if o.tag_id not in _seen_ids:
                _seen_ids[o.tag_id] = time.time()
                print(f"[discovery] >>> tag_id={o.tag_id}  (sign_type={o.sign_type})")
        img = frame.copy()
        try:
            _signs.draw(img, observations)
        except Exception:
            pass
        cv2.putText(img, f"backend={_backend}  tags={len(observations)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ids_txt = "IDs seen: " + (", ".join(str(i) for i in sorted(_seen_ids)) or "none yet")
        cv2.putText(img, ids_txt, (8, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        with _lock:
            _overlay = img
        time.sleep(0.02)


_PAGE = """<!doctype html><html><head><title>Sign Discovery</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:12px}
img{max-width:100%;border:2px solid #444;border-radius:6px}
#ids{font-size:20px;color:#0ff;margin:12px;min-height:24px}</style></head>
<body><h2>Sign Discovery — hold a sign in front of the camera</h2>
<img id="v"><div id="ids">loading...</div>
<script>
function poll(){const i=new Image();i.onload=()=>{document.getElementById('v').src=i.src;setTimeout(poll,150)};
i.onerror=()=>setTimeout(poll,500);i.src='/snapshot?'+Date.now();}
poll();
setInterval(()=>fetch('/ids').then(r=>r.json()).then(d=>{
  document.getElementById('ids').textContent='Discovered tag IDs: '+(d.ids.length?d.ids.join(', '):'none yet');
}),1000);
</script></body></html>"""


@app.route('/')
def index():
    return _PAGE


@app.route('/snapshot')
def snapshot():
    with _lock:
        img = None if _overlay is None else _overlay.copy()
    if img is None:
        return '', 204
    ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return '', 204
    return Response(jpeg.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})


@app.route('/ids')
def ids():
    from flask import jsonify
    return jsonify({'ids': sorted(_seen_ids), 'backend': _backend})


def main():
    global _camera, _signs, _running, _backend

    print('=' * 60)
    print('SIGN DISCOVERY SERVER (camera + AprilTags only)')
    print('=' * 60)

    print('\n[1/2] Starting camera...')
    _camera = CameraDriver()
    _camera.start()
    print('  Camera: ok')

    print('\n[2/2] Starting sign detector...')
    cfg = _load_config()
    try:
        _signs = SignDetector(cfg, sim=False)
        _signs.discovery_mode = True   # always show ALL tags, even ones already in config
        _backend = getattr(_signs, '_backend', '?')
        print(f'  SignDetector: ok (backend={_backend}, discovery=forced-on)')
    except Exception as e:
        print(f'  SignDetector FAILED: {e}')
        print('  -> the Jetson has no working AprilTag backend.')
        print('     install one:  pip3 install dt-apriltags')
        sys.exit(1)

    threading.Thread(target=_detect_loop, daemon=True, name='DetectLoop').start()

    port = find_available_port(5000)
    print(f'\nOpen:  http://localhost:{port}   (or http://glados.local:{port})')
    print('Press Ctrl+C to stop\n')
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        _running = False
        try:
            _camera.stop()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
