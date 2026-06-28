import sys
import os
import signal
import time
import threading
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request
import numpy as np
import cv2

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.project import get_template

import tasks.project.packages.agent as agent

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()

# ---- DISCOVERY-ONLY mode ----------------------------------------------------
# Set `discovery_only: true` in config/project_config.yaml to run camera + sign
# detection ONLY (no agent, no wheels, no ONNX). Lets you read sign tag IDs from
# the browser without launching the full driving pipeline. Default off.
_discovery_active  = False
_discovery_overlay = None
_discovery_ids     = {}
_discovery_backend = "?"
_discovery_lock    = threading.Lock()


def _visualize(frame):
    if _discovery_active:
        with _discovery_lock:
            if _discovery_overlay is not None:
                return _discovery_overlay.copy()
        return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    # On real hardware don't apply a max_age — the overlay goes stale during
    # blocking maneuvers (turns, stops) and then cam.read() blocks too, which
    # freezes the browser feed. Use whatever overlay we have; fall back to the
    # raw frame only when no overlay has been produced yet.
    overlay = agent.get_overlay()
    if overlay is not None:
        return overlay
    if frame is not None:
        return frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=False)


@app.route('/')
def index():
    return get_template(subtitle='Real Duckiebot', show_reset=False)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    if _discovery_active:
        with _discovery_lock:
            return jsonify({
                'state': 'discovery',
                'backend': _discovery_backend,
                'discovered_sign_ids': sorted(_discovery_ids.keys()),
            })
    return jsonify(agent.get_status())


@app.route('/command', methods=['POST'])
def command():
    data = request.get_json(silent=True) or {}
    key  = (data.get('key') or '').strip()
    if not key:
        return jsonify({'status': 'error', 'message': 'key required'}), 400
    try:
        msg = agent.apply_command(key, data.get('value'))
        return jsonify({'status': 'ok', 'message': msg})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/ping')
def ping():
    return 'ok'


@app.route('/snapshot')
def snapshot():
    if _discovery_active:
        with _discovery_lock:
            overlay = None if _discovery_overlay is None else _discovery_overlay.copy()
    else:
        overlay = agent.get_overlay()
        if overlay is None and camera is not None:
            ok, frame = camera.read()
            if ok and frame is not None:
                overlay = frame
    if overlay is None:
        return '', 204
    ret, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return '', 204
    return Response(jpeg.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def _discovery_loop():
    """Background: detect AprilTags, draw boxes, remember IDs. Camera + signs only."""
    global _discovery_overlay
    from tasks.project.packages.sign_detection import SignDetector, _load_config
    signs = SignDetector(_load_config(), sim=False)
    signs.discovery_mode = True   # show ALL tags regardless of config
    globals()['_discovery_backend'] = getattr(signs, '_backend', '?')
    print(f"  SignDetector: backend={_discovery_backend}, discovery=forced-on")
    while not stop_event.is_set():
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        try:
            observations = signs.detect(frame)
        except Exception as e:
            observations = []
            print(f"[discovery] detect error: {e}")
        img = frame.copy()
        try:
            signs.draw(img, observations)
        except Exception:
            pass
        with _discovery_lock:
            for o in observations:
                if o.tag_id not in _discovery_ids:
                    _discovery_ids[o.tag_id] = True
                    print(f"[discovery] >>> tag_id={o.tag_id} (sign_type={o.sign_type})")
            ids_txt = "IDs: " + (", ".join(str(i) for i in sorted(_discovery_ids)) or "none yet")
        cv2.putText(img, f"DISCOVERY  backend={_discovery_backend}  tags={len(observations)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, ids_txt, (8, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        with _discovery_lock:
            _discovery_overlay = img
        time.sleep(0.02)


def main():
    global camera, wheels, leds, stop_event, _discovery_active

    ap = argparse.ArgumentParser(description='Project Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()

    # discovery-only flag from config: camera + sign detection, no agent/wheels
    try:
        from tasks.project.packages.sign_detection import _load_config
        _discovery_active = bool((_load_config() or {}).get('discovery_only', False))
    except Exception:
        _discovery_active = False

    print('=' * 60)
    print('PROJECT SERVER — REAL HARDWARE' + ('  [DISCOVERY-ONLY MODE]' if _discovery_active else ''))
    print('=' * 60)

    if _discovery_active:
        print('\n[1/2] Initializing camera driver...')
        camera = CameraDriver()
        camera.start()
        print('  Camera: ok')
        print('\n[2/2] Starting sign discovery (no agent, no wheels)...')
        stop_event.clear()
        threading.Thread(target=_discovery_loop, daemon=True, name='DiscoveryLoop').start()

        def _shutdown_d(signum, frame):
            shutdown_cleanup(None, camera, stop_event)
            sys.exit(0)
        signal.signal(signal.SIGTERM, _shutdown_d)
        signal.signal(signal.SIGINT,  _shutdown_d)

        web_port = find_available_port(args.port)
        print(f'\nDISCOVERY: open http://<bot>:{web_port}  — hold signs in front of the camera')
        print('Set discovery_only: false in project_config.yaml to drive normally.\n')
        try:
            app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True, use_reloader=False)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            shutdown_cleanup(None, camera, stop_event)
        return

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting agent...')
    stop_event.clear()

    def _run_agent():
        try:
            agent.main(camera, wheels, leds, stop_event)
        except Exception as e:
            import traceback
            print(f'\n[AgentThread] CRASHED: {e}')
            traceback.print_exc()

    threading.Thread(target=_run_agent, daemon=True, name='AgentThread').start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nVideo stream: http://localhost:{web_port}/video')
    print('Press Ctrl+C to stop\n')

    try:
        # use_reloader=False avoids the double-process that breaks signals;
        # threaded=True lets the snapshot endpoint serve while agent is running
        app.run(host='0.0.0.0', port=web_port, debug=False,
                threaded=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
