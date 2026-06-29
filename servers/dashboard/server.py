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
import shutil
import signal
import subprocess
import tarfile
import threading

from flask import Flask, request, jsonify, Response

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(PROJECT_ROOT, 'last_task.log')

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

    # Purge stale bytecode. The tar restores the dev machine's file mtimes, so a
    # redeployed .py can look "unchanged" to Python's .pyc freshness check and a
    # stale __pycache__ entry gets loaded instead of the new source (this caused
    # "cannot import name 'SignDetector'" even after redeploying). Removing the
    # caches forces a clean recompile of whatever we just extracted.
    removed = _purge_bytecode(PROJECT_ROOT)

    return jsonify({'message': f'deployed ok ({removed} bytecode caches cleared)'})


def _purge_bytecode(root):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if '__pycache__' in dirnames:
            shutil.rmtree(os.path.join(dirpath, '__pycache__'), ignore_errors=True)
            dirnames.remove('__pycache__')
            count += 1
        for name in filenames:
            if name.endswith('.pyc'):
                try:
                    os.remove(os.path.join(dirpath, name))
                except OSError:
                    pass
    return count


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
        if debug:
            kwargs = {}
            log_path = None
        else:
            # Don't discard output to DEVNULL — when the task crashes on startup
            # that swallows the traceback and it just looks like "ran, then died".
            # Tee it to a log file so the crash reason is always recoverable.
            log_path = LOG_PATH
            log_f = open(log_path, 'wb')
            kwargs = {'stdout': log_f, 'stderr': subprocess.STDOUT}
        _task_proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, **kwargs)

    resp = {'pid': _task_proc.pid, 'port': port, 'task': task}
    if log_path:
        resp['log'] = log_path
    return jsonify(resp)


@app.route('/log')
def log():
    """Return the most recent task's captured output (its crash traceback lives
    here when a task dies on startup). Open http://<bot>:8000/log in a browser.
    Use ?tail=N to get only the last N bytes; add ?status=1 for JSON with the
    task's alive/exit state alongside the log."""
    if not os.path.isfile(LOG_PATH):
        return Response('no task log yet — start a task first\n',
                        mimetype='text/plain'), 404
    try:
        tail = int(request.args.get('tail', 0))
    except (TypeError, ValueError):
        tail = 0
    with open(LOG_PATH, 'rb') as f:
        if tail > 0:
            try:
                f.seek(-tail, os.SEEK_END)
            except OSError:
                f.seek(0)
        body = f.read().decode('utf-8', 'replace')

    if request.args.get('status'):
        if _task_proc is None:
            state = 'no task started'
        elif _task_proc.poll() is None:
            state = f'running (pid {_task_proc.pid})'
        else:
            state = f'exited (code {_task_proc.returncode})'
        return jsonify({'task': state, 'log': body})
    return Response(body, mimetype='text/plain')


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
