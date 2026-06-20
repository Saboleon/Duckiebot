# Traffic Signs project — status & tuning notes

## What this does

The robot follows the lane and reacts to traffic signs read from their
AprilTags (tag36h11, via OpenCV's `cv2.aruco` — no extra library needed):

| Sign | Tag IDs (our town) | Behaviour |
|------|--------------------|-----------|
| STOP | 24, 25, 26 | full stop, hold, give right-of-way, then go |
| YIELD | 39 | brief slow-down, then go |
| Pedestrian crossing | 20 | pause / stop if blocked, then continue |
| Side road on the right | 9 | at intersection: random **straight or right** |
| Side road on the left | 10 | at intersection: random **straight or left** |
| T-intersection (road ends) | 11 | at intersection: random **left or right** |

It also **stops for duckies** in its lane using the trained object-detection
model (`tasks/project/models/best.onnx`), reusing the `object_detection` task's
detector + `should_stop` logic. Runs on a background thread so it never stalls
lane following; toggle with `obstacle.enabled` in the config.

> The tag→sign map was recovered by running the detector over
> `tasks/object_detection/dataset/raw/*.jpg` (frames recorded in our own
> Duckietown). **Confirm each ID against the physical signs before the demo.**

## Files

- `packages/agent.py` — state machine (`cruise → approach → intersection → turn`).
- `packages/sign_detection.py` — AprilTag reader + sign/turn semantics.
- `packages/lane_following.py` — thin wrapper that reuses the proven
  `visual_lane_servoing` controller for lane keeping.
- `packages/road_perception.py` — red stop-line detector.
- `packages/object_detection.py` — duckie obstacle stopping (background thread;
  reuses the `object_detection` model + `should_stop`).
- `models/best.onnx` — the trained detector, copied here so it ships with the
  task (the `object_detection` copy is git-ignored).
- `packages/test_offline.py` — runs the whole pipeline against the dataset, no bot.
- `../../config/project_config.yaml` — **all** tunable numbers live here.
- `servers/project/virtual_server.py` — runs the agent in the Godot simulator.
- `servers/project/real_server.py` — runs the agent on the bot (now also serves
  `/status` + `/command` and overlays the video).

## Run

```bash
# offline self-test (no robot)
env/bin/python tasks/project/packages/test_offline.py

# in the Godot simulator (lane following + turn maneuvers)
python launch.py --sim --task project        # dashboard at http://localhost:5000

# on the robot
python launch.py --run --bot <bot-name> --task project
# dashboard (live video overlay + status + live tuning): http://<bot-name>.local:5000
python launch.py --stop --bot <bot-name>
```

## Simulation testing — what it can and can't show

`python launch.py --sim --task project` runs the *same* agent against the Godot
lane-following loop (`servers/project/virtual_server.py`).

Validated in sim:
- **Lane following** — the agent reuses the `visual_lane_servoing` controller, so
  it tracks the yellow centre + white edge and stays centred in the lane. Tuned
  by `config/lane_servoing_config.yaml` + `lane_servoing_hsv_config.yaml`.
- **Turn maneuvers** — click a **Drive Controls** button (Turn Left / Go Straight
  / Turn Right) to *queue* a turn; it fires when the bot reaches the red stop
  line. Plus Pause / Resume and Reset Scene (sim). They POST to `/command`
  (`/reset` for reset), so the key/value box works too.
- The state machine, video overlay, status and live config tuning.

### Intersection test scene

`launcher/config.py` points the `project` task at
`GodotSimulation/.../scenes/maps/intersection.tscn` — a cleaned copy of the
`introduction` map (the `KiuPathObj` road, which has a real intersection) with the
ducks and the parked vehicle removed. The bot starts on the road and lane-follows
toward the intersection. A turn is now **queued** and then triggered by the red
stop line: click a **Drive Controls** button (Turn Left / Go Straight / Turn
Right) any time while approaching — the bot keeps lane-following (a bit slower)
and runs the maneuver when it reaches the red line at the intersection, then
resumes lane following on the new arm. No timing by hand needed. (To test the
plain lane-following loop instead, set the `project` scene back to
`lane_follower.tscn` in `launcher/config.py`.)

This mirrors the real robot exactly: the **sign decides which way** (queues the
turn), the **red line decides when** (triggers it). Only the queue source differs
— a button in sim, an AprilTag sign on the bot — so the maneuver tuning
(`turn.left/right/straight.*`) carries straight over.

Each turn direction is tuned independently under `turn:` — `left`, `right`,
`straight`, each with its own `speed`, `sharpness` (left/right only) and
`duration`, plus a shared `turn.creep_s`. Left is set wider/deeper (lower
sharpness, longer) and right tighter, matching a real intersection. Live-tune
e.g. key `turn.left.sharpness` value `0.03`, or `turn.right.duration` value `1.3`.

The overlay shows `line=<now>/<trigger>` so you can watch the red-line fraction
rise as you approach; if it never reaches the trigger, lower
`stopline.trigger_area_frac` (or widen the red HSV range) live from the command box.

**Cannot** be tested in sim: AprilTag *sign recognition* — the simulation signs
carry no AprilTags. That perception step is the one item left for the real bot.

**Obstacle stopping in sim:** the intersection scene has no ducks (we removed
them), so it won't trigger there. To exercise it, point the `project` scene at
`lane_detect.tscn` (has ducks) or add a duck. If the model false-stops in sim
(it was trained on real images), disable live with command `obstacle.enabled`
`false`.

Note the sim camera renders at 1280×960 while the bot is 640×480; lane error is
normalised by image width so the same tuning works at either resolution.

The dashboard video shows the annotated feed (sign boxes, distance, current
state). The Status card shows the state machine live. The "Send Command" box
live-tunes any config value by dotted key, e.g. key `speed.cruise` value `0.4`,
or key `obstacle.enabled` value `true`.

## Tomorrow's tuning checklist (needs the real bot)

These were set from recorded images / sensible defaults and **must** be verified:

1. **Sign IDs** — drive past each sign, watch the overlay label matches reality.
2. **Lane following** (`lane.*`) — the HSV ranges and `steer_gain` depend on the
   room lighting. Tune so the bot tracks a straight lane before testing turns.
3. **Turn maneuvers** (`turn.left/right/straight.{speed,sharpness,duration}` +
   `turn.creep_s`) — open-loop, depend on floor grip + battery. Tune each
   direction; left wider/deeper, right tighter.
4. **`detection.act_px`** — how close the bot gets before acting on a sign.
5. **`stopline.trigger_area_frac`** — so the red line triggers reliably without
   false positives on other red surfaces.
6. **Obstacle stopping** is `enabled: false` by default (untested). Turn it on
   and tune `obstacle.area_frac` only after lane following is solid, or wire in
   the trained detector from the object_detection task.

## Known limitations / things to discuss

- Turns are **open-loop timed**, not closed-loop — good enough for the demo but
  sensitive to battery level.
- "Right-of-way / which robot goes first" is handled as a fixed wait
  (`timing.clear_time`); detecting the *other* robot to truly arbitrate would
  need a second perception step (the project lists this as a two-robot scenario).
- The video overlay freezes during a turn maneuver (the handler runs
  synchronously); status text resumes after the turn.
