"""Dashboard / deploy server — runs on the real Duckiebot at startup.

Listens on port 8000 (default) and handles three requests from launch.py:

  POST /deploy   — receives a task tar.gz, extracts it into the project root
  POST /start    — launches the task's real_server.py as a subprocess
  POST /stop     — kills the currently running task

Started automatically by the systemd service (duckiebot/dashboard.service).
"""

import os
import sys
import io
import signal
import subprocess
import tarfile
import threading

from flask import Flask, request, jsonify

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

_task_proc: subprocess.Popen = None
_task_lock = threading.Lock()


@app.route('/deploy', methods=['POST'])
def deploy():
    pkg = request.files.get('package')
    if pkg is None:
        return jsonify({'error': 'no package file'}), 400

    data = pkg.read()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            tar.extractall(path=PROJECT_ROOT)
    except Exception as e:
        return jsonify({'error': f'extraction failed: {e}'}), 500

    return jsonify({'message': 'deployed ok'})


@app.route('/start', methods=['POST'])
def start():
    global _task_proc

    body     = request.get_json(silent=True) or {}
    task     = body.get('task', '')
    port     = int(body.get('port', 5000))
    debug    = bool(body.get('debug', False))

    server_path = os.path.join(PROJECT_ROOT, 'servers', task, 'real_server.py')
    if not os.path.isfile(server_path):
        return jsonify({'error': f'no real server for task {task!r}'}), 400

    with _task_lock:
        _kill_task()
        cmd = [sys.executable, server_path, '--port', str(port)]
        kwargs = {} if debug else {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
        _task_proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, **kwargs)

    return jsonify({'pid': _task_proc.pid, 'port': port, 'task': task})


@app.route('/stop', methods=['POST'])
def stop():
    with _task_lock:
        _kill_task()
    return jsonify({'message': 'task stopped'})


def _kill_task():
    global _task_proc
    if _task_proc is not None and _task_proc.poll() is None:
        try:
            _task_proc.send_signal(signal.SIGTERM)
            _task_proc.wait(timeout=5)
        except Exception:
            _task_proc.kill()
    _task_proc = None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    print(f'[Dashboard] listening on port {args.port}  (project root: {PROJECT_ROOT})')
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
