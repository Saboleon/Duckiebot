# DuckieTown Rewritten — Project Overview

A study/reference guide for understanding and presenting this project.
Not a line-by-line walkthrough — deep enough to answer questions confidently.

---

## 1. What this project is

A **robotics education platform** built around the **Duckiebot DB21J** — a small
autonomous car robot from the [Duckietown](https://www.duckietown.org/) project
(Jetson Nano computer, one camera, two motors, LEDs, encoders).

Students write Python code that makes the robot drive itself. The exact same
student code runs in **two interchangeable places**:

1. **A Godot 4.6 simulation** — a 3D virtual Duckietown that runs on a laptop.
2. **The real physical robot** — over the network.

Only the *hardware drivers* underneath swap out. The brain (the student code)
is identical. The repo is organized as a series of **tasks** (lessons), each
with theory notebooks, Python packages to implement, and a live web dashboard
to watch and tune behavior.

---

## 2. Repository layout (the parts that matter)

```
Duckiebot/
├── launch.py              # single entry point: run a task in sim or on the bot
├── launcher/              # picks sim vs real, wires ports, transfers code to bot
├── requirements.txt       # Python deps (OpenCV, numpy, Flask, onnxruntime, ...)
│
├── duckiebot/             # HARDWARE DRIVERS (real + virtual)
│   ├── camera_driver/     #   camera (real) + godot_camera_driver (sim)
│   ├── wheel_driver/      #   motors (real) + godot_wheels_driver (sim)
│   ├── led_driver/        #   LEDs (real) + virtual_led_driver (sim)
│   ├── encoder_driver/    #   wheel encoders
│   └── hat_driver/        #   motor HAT / PWM low-level board control
│
├── servers/               # one small Flask web server per task
│   ├── project/
│   │   ├── real_server.py        # runs the agent on the real bot
│   │   ├── virtual_server.py     # runs the agent in the Godot sim
│   │   └── sign_discovery_server.py
│   └── dashboard/         # the boot dashboard (camera, start/stop tasks, battery)
│
├── tasks/                 # the lessons; each has notebooks/ + packages/
│   ├── introduction/      #   LEDs, keyboard driving
│   ├── braitenberg/       #   reactive light-following
│   ├── modcon/            #   kinematics, odometry, PID control
│   ├── visual_lane_servoing/  # lane following from the camera (reused later!)
│   ├── object_detection/  #   train a neural net to spot ducks
│   └── project/           # ★ THE CAPSTONE: Traffic Signs (the main deliverable)
│
├── config/                # YAML config files — ALL tunable numbers live here
├── GodotSimulation/       # the Godot 4.6 simulator project (GDScript, 3D maps)
└── docs/                  # extra docs (map maker, etc.)
```

The **tasks build on each other**: the final `project` reuses the lane-following
controller from `visual_lane_servoing` and the neural-net detector from
`object_detection`.

---

## 3. The capstone: the Traffic Signs project

This is `tasks/project/` — the most important part. The robot drives a
Duckietown course autonomously and obeys traffic signs.

### 3.1 Behaviors

| Sign / event | Detected by | Behavior |
|--------------|-------------|----------|
| **STOP** | AprilTag (IDs 20, 24) | Full stop, hold, give right-of-way, then go |
| **YIELD** | AprilTag (ID 39) | Brief slow-down, then proceed |
| **Pedestrian crossing** | AprilTag | Pause / stop if blocked, then continue |
| **Side road right** (ID 9) | AprilTag | At intersection: random **straight or right** |
| **Side road left** (ID 10) | AprilTag | At intersection: random **straight or left** |
| **T-junction** (ID 11) | AprilTag | At intersection: random **left or right** |
| **Parking** | AprilTag | Stop and stay parked — this is the destination |
| **Duck / robot ahead** | Neural net (ONNX) | Stop until it clears (creep past a stalled car) |

### 3.2 The key design idea — "sign decides WHICH way, red line decides WHEN"

This is the cleverest part of the design, and worth understanding well:

- A **traffic sign** seen while approaching → *queues* a turn ("I should go left").
- A **red stop-line** painted on the floor → *triggers* that queued turn at the
  exact right moment (when the bot actually reaches the intersection).

Why split it? Because it lets the **simulation test almost everything** even
though the simulated signs carry no AprilTags. In sim you queue the turn with a
**dashboard button** instead of a sign; the red-line trigger and the turn
maneuver are then identical to the real bot. Only the *queue source* differs
(button in sim, AprilTag on the real bot).

### 3.3 The state machine (`packages/agent.py`)

`main(camera, wheels, leds, stop_event, sim)` runs one big loop. Each camera
frame, in priority order:

1. **Manual override?** — if a "drive l,r" command is active, just obey it.
2. **Paused?** — hold still (but keep detecting signs so IDs can be discovered).
3. **Obstacle ahead?** — overrides everything. A **duck** = permanent stop (ducks
   don't move). A **car** = stop, but after `max_block_s` seconds creep past it
   (it might be parked).
4. **Force-turn command?** — execute a turn immediately (sim convenience).
5. **Intersection sign in view?** — queue a turn (`pending_turn`).
6. **STOP / YIELD / PARKING close?** — track them for the no-red-line fallback.
7. **At the red line?** — execute the queued turn, applying stop/yield
   right-of-way first.
8. **No red line but a sign has been close too long?** — fallback: act on the
   sign alone after a timeout.
9. **Otherwise** — just follow the lane (slower if a turn is pending).

States you'll see on the dashboard: `init → cruise → approach → obstacle / stop /
yield / parked → (turn) → cruise`.

### 3.4 The supporting modules

```
agent.py  ───────────────  the brain / state machine
   │
   ├── sign_detection.py     reads AprilTags, maps tag ID → sign type → allowed turns
   ├── lane_following.py     thin wrapper around the visual_lane_servoing controller
   ├── road_perception.py    detects the red stop-line band on the floor
   └── object_detection.py   runs the neural net on a BACKGROUND THREAD for ducks/cars
```

- **`sign_detection.py`** — reads AprilTags (tag36h11) via OpenCV's `cv2.aruco`.
  A `SignObservation` records the tag ID, sign type, pixel height (→ distance
  estimate), and center position. `act_px` decides "we are close enough to act."
- **`lane_following.py`** — reuses the proven `LaneServoingAgent` from the
  earlier lane task, so lane keeping isn't reinvented. It tracks the yellow
  center line + white edge line and computes left/right wheel speeds.
- **`road_perception.py`** — looks for a red band in the lower part of the frame;
  when the red fraction crosses `trigger_area_frac`, that's the stop line.
- **`object_detection.py`** — runs the trained ONNX model on its own thread so
  vision never stalls the driving loop. Also used to watch for crossing traffic
  at yields (right-of-way).

### 3.5 Turn maneuvers

Turns are **open-loop timed arcs** — drive both wheels forward at fixed speeds
for a set duration (not a pivot, a gradual arc). Each direction (left / right /
straight) is tuned independently in the config:

- `speed` — how fast
- `sharpness` — the small speed difference between wheels (sets the arc radius)
- `duration` — how long
- `creep_s` — a short straight run-in before the arc begins
- `trim` — corrects the bot's systematic drift (no closed-loop correction)

Left is set wider/deeper (crosses to the far lane); right is tighter. This is
**known to be battery- and floor-sensitive** — it must be tuned on the real bot.

---

## 4. Technologies used

| Area | Tech |
|------|------|
| Robot logic | **Python** |
| Computer vision | **OpenCV** (`cv2`) + **NumPy** |
| Sign recognition | **AprilTags** (tag36h11) via `cv2.aruco`, fallback `dt_apriltags` / `pupil_apriltags` |
| Obstacle detection | **ONNX neural network** (`best.onnx`) via `onnxruntime` |
| Web servers | **Flask** (one tiny server per task) |
| Video stream | **MJPEG** over HTTP (`/video`) — shows in a plain `<img>` tag |
| Config | **YAML** files in `config/` |
| Simulator | **Godot 4.6** (GDScript), 3D Duckietown maps |
| Robot OS | Jetson Nano (the real bot); code transferred over HTTP |

**Why AprilTags instead of training a sign classifier?** They're reliable, fast,
need no training data, and Duckietown signs are designed to carry them. The
neural net is saved for ducks/cars, where there's no tag to read.

**Why is OpenCV's aruco wrapped in fallbacks?** The Jetson's stock OpenCV is
often built *without* the aruco module, so the code tries new-aruco → legacy-aruco
→ a dedicated apriltag library, whichever is present.

---

## 5. How it all connects (request flow)

```
   launch.py  ──►  launcher/  ──►  chooses sim or real
                                        │
            ┌────────────────────────────┴───────────────────────────┐
   virtual_server.py (Godot)                          real_server.py (the bot)
            └──────────────── both import & run the SAME ─────────────┘
                                        │
                          tasks/project/packages/agent.py
                                        │
            (camera frames in → wheel/LED commands out, each loop)
```

### The web API (how the browser talks to the robot)

Each task server is a **Flask app**; the browser dashboard talks to it purely
over HTTP, so the robot needs no display of its own:

- **`GET /video`** — live MJPEG stream of the annotated camera feed (sign boxes,
  state label, lane markers).
- **`GET /status`** — polled every few hundred ms; returns JSON of the current
  state (state machine status, frame count, current maneuver, discovered tag IDs…).
- **`POST /command`** or **`/update_config`** — sliders/buttons send values here;
  the server updates its in-memory config *immediately* (effective next frame)
  **and** writes it back to the YAML so it persists across restarts.

This is why you can **tune the robot live** while it drives — e.g. send key
`speed.cruise` value `0.4`, or `turn.left.duration` value `1.3`.

---

## 6. How to run it

```bash
# Offline self-test against recorded images (no robot at all)
python tasks/project/packages/test_offline.py

# In the Godot simulator (downloads Godot automatically on first run)
python launch.py --sim --task project

# On the real robot (by hostname or IP)
python launch.py --run --bot <bot-name> --task project
python launch.py --run --host 192.168.1.100 --task project

# Stop a running task on the bot
python launch.py --stop --bot <bot-name>
```

Then open the printed dashboard URL (`http://<bot>.local:8000` or a localhost
port) for the live video, status, and tuning controls.

### Discovery mode

Set `discovery_only: true` (or leave `signs_real` empty) to run **camera + sign
detection only — no driving**. Hold each physical sign in front of the camera,
read its tag ID off the video / the `discovered_sign_ids` status, and fill the
IDs into `config/project_config.yaml`. Turn it off to drive normally.

---

## 7. Configuration — everything tunable lives in YAML

`config/project_config.yaml` holds **every** number the agent uses, so behavior
can be tuned without touching code. Main sections:

- **`signs_real` / `signs_sim`** — the tag-ID → sign-type lookup tables.
- **`detection`** — `observe_px` / `act_px`: how close (in tag pixel height) a
  sign must be before the bot reacts.
- **`speed`** — `cruise` and `approach` wheel speeds.
- **`stopline`** — red HSV color range + `trigger_area_frac` (when a red blob
  counts as the stop line).
- **`timing`** — how long to dwell at stop/yield, how long the intersection must
  read clear before going (right-of-way), and a safety timeout cap.
- **`turn`** — per-direction arc tuning (`speed`, `sharpness`, `duration`,
  `creep_s`, `trim`).
- **`obstacle`** — duck/car stopping thresholds (how big/where in the lane a duck
  must be to trigger a stop), and `max_block_s` (creep-past timeout for cars).
- **`yield_traffic`** — which detector classes count as crossing traffic, and
  which band of the frame to watch.

---

## 8. Likely Q&A (for a presentation / defense)

**Q: How does the robot recognize signs?**
Each Duckietown sign carries an AprilTag (a QR-like marker). OpenCV reads the
tag's ID and corners; a config lookup maps the ID to a sign type. Distance is
estimated from the tag's apparent pixel height (bigger = closer).

**Q: How can the same code run in sim and on the real robot?**
The student logic (`agent.py`) is identical. Only the hardware abstraction layer
differs — `godot_*` drivers for the sim vs the real hardware drivers — chosen by
the `sim` flag and the server that launches it.

**Q: Are the turns precise?**
No — they're open-loop timed arcs, not closed-loop. Good enough for a demo but
sensitive to battery level and floor grip, so they must be tuned on the real bot.
A closed-loop (encoder/odometry-based) turn would be the next improvement.

**Q: How is right-of-way / "which robot goes first" handled?**
At a stop/yield the bot watches a band of the camera frame (via the object
detector) for crossing traffic and waits until it's been clear long enough,
with a safety timeout. True multi-robot negotiation is *not* implemented — it's
a fixed-wait approximation, and a known limitation to mention.

**Q: Why a neural net for ducks but AprilTags for signs?**
Ducks/cars carry no markers and vary in appearance, so they need a learned
detector. Signs are standardized and carry tags, so tag reading is far more
reliable and needs no training data.

**Q: Why does obstacle detection run on a separate thread?**
So the neural-net inference (which is slower) never stalls the fast driving /
lane-following loop. The main loop just reads the latest result.

**Q: What happens if sign detection fails to initialize on the bot?**
The agent falls back to a dummy detector and reports `sign_detector: FAILED` in
the status, so the bot still lane-follows instead of crashing. The code tries
several AprilTag backends precisely because the Jetson's OpenCV may lack aruco.

**Q: How do live config changes take effect without restarting?**
The Flask server holds the config in memory and the agent re-reads it each loop.
A `/command` or `/update_config` POST mutates that in-memory object (next frame)
and is also written back to the YAML for persistence.

---

## 9. Known limitations (good to mention proactively)

- **Open-loop turns** — battery/grip sensitive; not closed-loop.
- **Right-of-way is a fixed wait**, not true robot-to-robot arbitration.
- **The video overlay freezes during a turn maneuver** (the turn handler runs
  synchronously); status text resumes after the turn finishes.
- **Sign-ID table must be verified on the physical signs** before a demo — the
  IDs in the config were recovered from recorded images and could differ.
- **Sim cannot test AprilTag sign recognition** — the simulated signs carry no
  tags, so that one perception step is only exercisable on the real bot.

---

*Generated as a study reference. The authoritative details live in
`tasks/project/NOTES.md`, `README.md`, and `config/project_config.yaml`.*
