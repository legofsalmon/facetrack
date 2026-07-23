# facetrack — live face tracking to NDI

Points a camera at a crowd, finds and follows faces, and sends the result
over **NDI** to your vision mixer — with a browser control panel for
everything. Built for live events.

**What it does / doesn't do:** face *detection and tracking* only — boxes,
stable numbers, optional expression labels. It does **not** identify
people and stores nothing.

---

## Quick start

### Get it onto the machine

Either clone with git:

```bash
git clone https://github.com/legofsalmon/facetrack.git
```

…or on GitHub click **Code → Download ZIP** and unzip it anywhere.
To update later: `git pull` (or re-download), then run Setup again.

### First time (once per machine)

- **Mac** — double-click **`Setup Mac.command`**. If macOS blocks it,
  right-click → Open. When it asks for camera access, click **Allow**.
- **Windows** — install Python 3.10+ from python.org (tick *"Add python.exe
  to PATH"*), then double-click **`Setup Windows.bat`**.

Setup installs everything, downloads any missing model files, and runs a
self-check that tells you in plain language if something needs fixing.

### Every show

Double-click **`Start Mac.command`** / **`Start Windows.bat`**.

The **control panel opens in your browser** automatically. Your mixer will
see two NDI sources named like `MAC (FaceTracker)` — the panel shows the
exact names. That's it.

### The control panel

- Reachable from **any phone/laptop/tablet on the same network** — the
  terminal window shows the address (e.g. `http://192.168.1.20:8089`).
- **Presets** across the top: *Wide crowd*, *Mid crowd*, *Stage close-up*,
  *Power saver*. Start with one of these; fine-tune below if needed.
- **Video source** lists every connected camera by name (webcams, capture
  cards, virtual cameras) and every NDI feed on the network — switching is
  instant, no restart. Plug in a device and hit the ↻ rescan button; the
  camera currently in use is shown as "(in use)" and left undisturbed.
- Every change applies live **and is remembered for next launch**.
- If the input dies or is missing, the app keeps running and shows a
  NO INPUT slate — fix the source from the panel.

### If something's wrong

Run the Setup script again (safe any time), or from a terminal:

```bash
.venv/bin/python main.py --doctor
```

It checks Python, packages, models, camera, NDI and the panel port, and
prints the fix for anything broken.

---

## Technical reference

### Pipeline

| Stage | Implementation | Notes |
|---|---|---|
| Capture | OpenCV (camera/capture card/file) or NDI in | threaded, latest-frame-wins for low latency |
| Detection | YuNet (OpenCV, CPU) or SCRFD-10G (ONNX Runtime, GPU) | auto-selected per machine |
| Tracking | SORT-style IoU + velocity tracker | stable IDs, sub-ms for hundreds of faces |
| Expression | FER+ (8 classes), budgeted round-robin | cost stays flat as crowd grows |
| Output | NDI via cyndilib (+ local preview window) | NDI runtime bundled |
| Control | FastAPI + WebSocket panel on :8089 | settings persist in `settings.json` |

### Machine notes

- **macOS (testing)**: YuNet on CPU — ~4-5 ms detection at 640 px on an
  M2 Max; 100+ fps at 720p.
- **Windows + NVIDIA (production, RTX 5080)**: `requirements.txt` installs
  `onnxruntime-gpu`; the SCRFD-10G detector runs on CUDA/TensorRT
  automatically — markedly better on dense crowds. There's headroom for
  *Search detail: Maximum (1280)* at show resolution. First TensorRT run
  compiles an engine (can take a minute). For capture cards try
  `--capture-backend dshow`, then `msmf`.

### Two-feed keying workflow

Send a **clean camera feed** plus a **graphics-only overlay with real
alpha**, and key downstream in vMix / Resolume / TriCaster / OBS+DistroAV:

```bash
python main.py --clean-main --ndi-overlay "FaceTracker Overlay"
```

(Or toggle *Clean camera feed* in the panel; the overlay feed needs the
flag at launch.) NDI's codec is lossy, so keyed edges gain a pixel of soft
fringe — normal for all NDI alpha sources. The two feeds are emitted
together; your mixer's frame sync aligns them.

### CLI flags

Everything in the panel is also a flag (`python main.py --help`). Flags
override saved settings for that run. Non-panel flags:

| Flag | Purpose |
|---|---|
| `--source` | camera index / file / URL / `ndi:<name>` |
| `--width --height --fps` | capture request (default 1280x720@30) |
| `--backend yunet\|scrfd` | force a detector |
| `--ndi-name` / `--ndi-overlay` / `--no-ndi` | feed naming |
| `--out-width` | downscale the NDI send |
| `--no-web` / `--web-host` / `--web-port` / `--no-browser` | panel control |
| `--no-preview` | no local window (headless/rack use) |
| `--doctor` | self-check and exit |

### Layout

```
main.py                  entry point: flags, saved settings, banner
facetrack/
  capture.py             camera / file / NDI-in / NO-INPUT slate, camera probe
  detectors.py           YuNet + SCRFD backends, live-tunable
  tracker.py             SORT-style multi-face tracker
  emotion.py             FER+ expression estimation (budgeted)
  overlay.py             boxes/labels/stats + alpha overlay rendering
  ndi_io.py              NDI output + NDI input (cyndilib)
  pipeline.py            the frame loop, hot source-swap, stats, preview JPEGs
  params.py              validated live parameters
  settings.py            auto-persistence (settings.json)
  webui.py               FastAPI app: panel, WebSocket, MJPEG, /sources
  static/index.html      the control panel
  doctor.py              self-check (python -m facetrack.doctor)
models/                  ONNX models (doctor --fix re-downloads)
```

### Known quirks

- The installed NDI HX driver makes OpenCV's ffmpeg print an `objc`
  duplicate-class warning at startup on this Mac. Harmless.
- The panel has no authentication — it's meant for a closed production
  LAN. Use `--web-host 127.0.0.1` to keep it local-only.
- macOS camera permission belongs to the *terminal app* that launches
  facetrack; grant it once when prompted.
