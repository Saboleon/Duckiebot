"""Project server — Godot simulation.

Runs the same `tasks/project/packages/agent.py` as the real bot, but wired to
the Godot camera/wheels and a virtual LED driver. Use it to test and tune lane
following (and the turn maneuvers) before deploying to hardware.

  python launch.py --sim --task project

Notes:
  * Three AprilTag signs (tag36h11, IDs 9/10/11) are placed along the road in
    intersection.tscn so sign recognition can be tested in sim. Each sign face
    points toward the bot's approach direction. Adjust positions in the Godot
    editor (node AprilTagSigns/*) if the bot doesn't see them clearly.
  * `force_turn` (key) with value `left`/`right`/`straight` still works as a
    manual override from the dashboard "Send Command" box.
  * `pause` / `resume` (as a command key) hold or release the motors.
"""

import sys
import os
import argparse
import threading

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from launcher.config import GODOT_SCENES
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.project import get_template

import tasks.project.packages.agent as agent

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()


class _SimLeds:
    """Tiny stand-in for the LED driver in simulation.

    The real `duckiebot.led_driver` package imports `smbus2` (hardware only) at
    import time, which isn't installed off-bot - so we keep a local no-op driver
    that just records the last colour set on each corner (handy for /status)."""

    def __init__(self):
        self.state = {0: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0]}

    def set_rgb(self, led, color):
        if led in self.state:
            self.state[led] = list(color)

    def all_on(self):
        for i in self.state:
            self.set_rgb(i, [1, 1, 1])

    def all_off(self):
        for i in self.state:
            self.set_rgb(i, [0, 0, 0])

    def release(self):
        self.all_off()


def _visualize(frame):
    # Prefer the agent's annotated overlay, but only if it's recent. During a
    # blocking maneuver (turn / yield / stop wait) the control loop stops
    # refreshing it; falling back to the live camera frame keeps the video
    # moving instead of freezing on the last annotated image.
    overlay = agent.get_overlay(max_age=0.3)
    if overlay is not None:
        return overlay
    return frame


# the Godot camera decodes to BGR, so read() (rgb=False) matches the real bot
generate_frames = make_frame_generator(lambda: camera, _visualize, quality=60, rgb=False)


@app.route('/')
def index():
    return get_template(subtitle='Godot Simulation', show_reset=True)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    st = agent.get_status()
    if wheels is not None:
        st['game_over'] = wheels.is_game_over()
    return jsonify(st)


@app.route('/command', methods=['POST'])
def command():
    data = request.get_json(silent=True) or {}
    key = (data.get('key') or '').strip()
    if not key:
        return jsonify({'status': 'error', 'message': 'key required'}), 400
    try:
        msg = agent.apply_command(key, data.get('value'))
        return jsonify({'status': 'ok', 'message': msg})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/reset', methods=['POST'])
def reset():
    if wheels:
        wheels.reset_game()
    agent.set_paused(False)
    return jsonify({'status': 'reset'})


@app.route('/remove_duck', methods=['POST'])
def remove_duck():
    """Remove one duck obstacle from the Godot scene. Handy when the bot has
    stopped for a duck: clear it and the bot resumes once the camera no longer
    sees it. The 'duck_' filter matches the Duck_* nodes only (not DuckieBot or
    the Ducks container), and Godot frees the first match, so one duck per call.
    Also un-pauses, in case the bot was holding (e.g. parked)."""
    if not wheels:
        return jsonify({'status': 'error', 'message': 'no sim connection'}), 400
    wheels.remove_objects('duck_')
    agent.set_paused(False)
    return jsonify({'status': 'ok', 'message': 'Removed a duck'})


@app.route('/switch_scene', methods=['POST'])
def switch_scene():
    target = (request.json or {}).get('scene', '')
    if target not in GODOT_SCENES:
        return jsonify({'error': f'unknown scene {target!r}'}), 400
    if wheels:
        wheels.change_scene(GODOT_SCENES[target])
    return jsonify({'scene': target})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds

    ap = argparse.ArgumentParser(description='Project Server — Godot Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — GODOT SIMULATION (traffic signs agent)')
    print('=' * 60)

    print('\n[1/3] Initializing wheels...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )

    print('\n[2/3] Initializing camera...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    leds = _SimLeds()

    print('\n[3/3] Starting agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event),
        kwargs={'sim': True},
        daemon=True,
        name='AgentThread',
    ).start()
    print('  agent.main() running')

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
