from .base import render_template

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img id="videoStream" class="stream">
        </div>

        <div class="controls-section">

            <div class="card">
                <div class="card-header">
                    Status
                    <span id="statusDot" style="width:8px;height:8px;border-radius:50%;
                        background:var(--accent-green);display:inline-block;"></span>
                </div>
                <div id="statusTable" style="font-size:12px;">
                    <div style="color:var(--text-muted);text-align:center;padding:12px 0;">
                        Waiting for data...
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">Intersection (queues for next sign)</div>
                <div style="display:flex;gap:6px;">
                    <button class="button" onclick="sendTurn('left')">&#8592; Turn Left</button>
                    <button class="button" onclick="sendTurn('right')">Turn Right &#8594;</button>
                </div>
                <button class="button" onclick="sendTurn('straight')">&#8593; Go Straight</button>
                <div style="display:flex;gap:6px;">
                    <button class="button danger" onclick="sendCtl('pause')">Pause</button>
                    <button class="button success" onclick="sendCtl('resume')">Resume</button>
                </div>
                <!--RESET-->
                <!--REMOVE_DUCK-->
                <div id="driveStatus" class="status"></div>
            </div>

            <div class="card">
                <div class="card-header">Manual Drive (hold to move, WASD / arrows)</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;max-width:180px;margin:0 auto;">
                    <div></div>
                    <button class="button" id="btn-fwd"
                        onmousedown="startDrive(0.28,0.28)" onmouseup="stopDrive()"
                        ontouchstart="startDrive(0.28,0.28)" ontouchend="stopDrive()">&#8593;</button>
                    <div></div>
                    <button class="button" id="btn-left"
                        onmousedown="startDrive(0.10,0.28)" onmouseup="stopDrive()"
                        ontouchstart="startDrive(0.10,0.28)" ontouchend="stopDrive()">&#8592;</button>
                    <button class="button danger" id="btn-stop"
                        onmousedown="stopDrive()" ontouchstart="stopDrive()">&#9632;</button>
                    <button class="button" id="btn-right"
                        onmousedown="startDrive(0.28,0.10)" onmouseup="stopDrive()"
                        ontouchstart="startDrive(0.28,0.10)" ontouchend="stopDrive()">&#8594;</button>
                    <div></div>
                    <button class="button" id="btn-bwd"
                        onmousedown="startDrive(-0.22,-0.22)" onmouseup="stopDrive()"
                        ontouchstart="startDrive(-0.22,-0.22)" ontouchend="stopDrive()">&#8595;</button>
                    <div></div>
                </div>
                <div id="manualStatus" class="status"></div>
            </div>

            <div class="card">
                <div class="card-header">Send Command</div>
                <div style="display:flex;flex-direction:column;gap:8px;">
                    <div style="display:flex;gap:6px;">
                        <input id="cmdKey" type="text" placeholder="key"
                            style="flex:1;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                        <input id="cmdValue" type="text" placeholder="value"
                            style="flex:2;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                    </div>
                    <button class="button" onclick="sendCommand()">Send</button>
                    <div id="cmdStatus" class="status"></div>
                </div>
            </div>

        </div>
    </div>
'''

_EXTRA_CSS = '''
#statusTable .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
    align-items: baseline;
}
#statusTable .row:last-child { border-bottom: none; }
#statusTable .key  { color: var(--text-secondary); font-size: 12px; }
#statusTable .val  { color: var(--text-primary);   font-weight: 500; font-size: 13px; font-family: monospace; }
'''

_EXTRA_JS = '''
function sendTurn(dir) {
    postJSON('/command', {key: 'force_turn', value: dir})
        .then(r => showStatus('driveStatus', r.status === 'ok' ? ('Turn: ' + dir) : r.message,
                              r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('driveStatus', 'Error: ' + e, 'error'));
}

function sendCtl(key) {
    postJSON('/command', {key: key, value: ''})
        .then(r => showStatus('driveStatus', r.message, r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('driveStatus', 'Error: ' + e, 'error'));
}

function resetScene() {
    postJSON('/reset', {})
        .then(r => showStatus('driveStatus', 'Scene reset', 'success'))
        .catch(e => showStatus('driveStatus', 'Error: ' + e, 'error'));
}

function removeDuck() {
    postJSON('/remove_duck', {})
        .then(r => showStatus('driveStatus', r.message || 'Removed duck',
                              r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('driveStatus', 'Error: ' + e, 'error'));
}

function refreshStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            const table = document.getElementById('statusTable');
            const keys = Object.keys(data);
            if (keys.length === 0) {
                table.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:12px 0;">get_ui_data() returned {}</div>';
                return;
            }
            table.innerHTML = keys.map(k =>
                `<div class="row">
                    <span class="key">${k}</span>
                    <span class="val">${JSON.stringify(data[k])}</span>
                </div>`
            ).join('');
            document.getElementById('statusDot').style.background = 'var(--accent-green)';
        })
        .catch(() => {
            document.getElementById('statusDot').style.background = 'var(--accent-red)';
        });
}

function sendCommand() {
    const key   = document.getElementById('cmdKey').value.trim();
    const value = document.getElementById('cmdValue').value.trim();
    if (!key) {
        showStatus('cmdStatus', 'Key cannot be empty', 'error');
        return;
    }
    postJSON('/command', {key, value})
        .then(r => showStatus('cmdStatus', r.status === 'ok' ? 'Sent' : r.message, r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('cmdStatus', 'Error: ' + e, 'error'));
}

document.getElementById('cmdValue').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendCommand();
});

refreshStatus();
setInterval(refreshStatus, 1000);

// Manual drive
function startDrive(l, r) {
    postJSON('/command', {key: 'drive', value: l + ',' + r})
        .then(rsp => showStatus('manualStatus', rsp.message || 'driving', 'success'))
        .catch(e  => showStatus('manualStatus', 'Error: ' + e, 'error'));
}
function stopDrive() {
    postJSON('/command', {key: 'drive_stop', value: ''})
        .catch(() => {});
    showStatus('manualStatus', 'stopped', '');
}
// Keyboard WASD / arrows
const _keys = {};
document.addEventListener('keydown', e => {
    if (_keys[e.key]) return;
    _keys[e.key] = true;
    const spd = 0.28, turn = 0.05;
    if (e.key === 'w' || e.key === 'ArrowUp')    startDrive(spd,  spd);
    if (e.key === 's' || e.key === 'ArrowDown')  startDrive(-0.22, -0.22);
    if (e.key === 'a' || e.key === 'ArrowLeft')  startDrive(turn, spd);
    if (e.key === 'd' || e.key === 'ArrowRight') startDrive(spd,  turn);
    if (e.key === ' ') stopDrive();
});
document.addEventListener('keyup', e => {
    delete _keys[e.key];
    if (['w','s','a','d','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))
        stopDrive();
});

// Video polling — 200ms (5fps) is enough and doesn't flood the WiFi link
let _videoErrors = 0;
(function pollVideo() {
    const next = new Image();
    next.onload = function() {
        document.getElementById('videoStream').src = this.src;
        _videoErrors = 0;
        setTimeout(pollVideo, 200);
    };
    next.onerror = function() {
        _videoErrors++;
        // after 30 consecutive failures (~15 s) reload the page to reconnect
        if (_videoErrors >= 30) { location.reload(); return; }
        setTimeout(pollVideo, 500);
    };
    next.src = '/snapshot?' + Date.now();
})();
'''


_RESET_BUTTON = ('<button class="button" onclick="resetScene()" '
                 'style="background:var(--accent-orange);">&#8635; Reset Scene</button>')

_REMOVE_DUCK_BUTTON = ('<button class="button" onclick="removeDuck()" '
                       'style="background:var(--accent-orange);">&#128036; Remove Duck</button>')


def get_template(title='Project', subtitle='Real Duckiebot', show_reset=False):
    # the Reset Scene / Remove Duck buttons only make sense in the simulator
    content = _CONTENT.replace('<!--RESET-->', _RESET_BUTTON if show_reset else '')
    content = content.replace('<!--REMOVE_DUCK-->', _REMOVE_DUCK_BUTTON if show_reset else '')
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=content,
        extra_css=_EXTRA_CSS,
        extra_js=_EXTRA_JS,
    )
