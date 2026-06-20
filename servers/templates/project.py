from .base import render_template

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img src="/video" class="stream" id="videoStream">
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
                <div class="card-header">Drive Controls</div>
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
                <div id="driveStatus" class="status"></div>
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
setInterval(refreshStatus, 500);
'''


_RESET_BUTTON = ('<button class="button" onclick="resetScene()" '
                 'style="background:var(--accent-orange);">&#8635; Reset Scene</button>')


def get_template(title='Project', subtitle='Real Duckiebot', show_reset=False):
    # the Reset Scene button only makes sense in the simulator
    content = _CONTENT.replace('<!--RESET-->', _RESET_BUTTON if show_reset else '')
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=content,
        extra_css=_EXTRA_CSS,
        extra_js=_EXTRA_JS,
    )
