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

### Run it

Double-click **`Facetrack Mac.command`** (Mac) or **`Facetrack Windows.bat`**
(Windows). One file does everything:

- **First launch** sets itself up — installs components, downloads the
  model files, runs a self-check (a few minutes; on Mac, if macOS blocks
  the file, right-click → Open). Click **Allow** on the camera prompt.
- **Every launch after that** starts in seconds.
- **After an update** (`git pull` / re-download) it notices and re-runs
  just the setup steps that are needed.

Windows first: install Python 3.13 from python.org (tick *"Add python.exe
to PATH"*). Mac: nothing to pre-install (having Homebrew's `uv` gives the
most reliable setup: `brew install uv`).

The **control panel opens in your browser** automatically. Your mixer will
see NDI sources named like `MAC (FaceTracker)` — the panel shows the exact
names. That's it.

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

Delete the `.venv` folder and double-click the Facetrack file again — it
rebuilds everything. Or, from a terminal:

```bash
.venv/bin/python main.py --doctor
```

It checks Python, packages, models, camera, NDI, Syphon/Spout and the
panel port, and prints the fix for anything broken.

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

### Output feeds

Everything below is a live toggle in the panel's **Output feeds** card and
persists across restarts:

| Output | What it carries | Notes |
|---|---|---|
| Main NDI feed | the program picture (annotated, or clean with *Clean camera feed* on) | panel shows connected-receiver count |
| Overlay NDI feed | graphics only, real alpha (premultiplied) | key it in vMix / Resolume / TriCaster / OBS+DistroAV |
| Syphon (macOS) / Spout (Windows) | program picture **or** overlay-with-alpha | zero-compression, GPU-to-GPU, same machine only |
| Output size | Match input / 1920 / 1280 / 960 wide | applies to all feeds; lowers network load |

Feed *names* are fixed at launch (`--ndi-name`, `--ndi-overlay`; the
overlay defaults to "<name> Overlay") because renaming mid-show would drop
receivers. NDI's codec is lossy, so keyed NDI graphics gain a pixel of
soft fringe — normal for all NDI alpha sources; the Syphon/Spout path is
uncompressed and keys perfectly. Feeds are emitted together; your mixer's
frame sync aligns them.

**Syphon/Spout notes:** the texture share appears as `facetrack` in
Resolume / VDMX / MadMapper / TouchDesigner on the same machine. *Share
overlay only* switches it to the graphics-with-alpha layer — VJ keying
with no network hop. Python version matters: Syphon needs Python 3.12,
Spout ≤ 3.13 — the Setup scripts pick a compatible Python automatically
(on macOS they prefer a uv-managed 3.12; Homebrew's python@3.12 bottle is
currently broken on macOS 26.1).

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
