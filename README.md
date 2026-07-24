# facetrack — live face tracking to NDI

Points a camera at a crowd, finds and follows faces, and sends the result
over **NDI** to your vision mixer — with a browser control panel for
everything. Built for live events.

**What it does / doesn't do:** face *detection and tracking* only — boxes,
stable numbers, optional expression labels. It does **not** identify
people and stores nothing.

![sample output](assets/sample_output.png)

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

- **Mac** — double-click **`Facetrack.app`** (drag it to your Dock for
  one-click starts). It opens Terminal running the launcher.
- **Windows** — double-click **`Facetrack Windows.bat`**, or run
  **`Create Desktop Icon.bat`** once to get a proper desktop icon.

One launcher does everything:

- **First launch** sets itself up — installs components, downloads the
  model files, runs a self-check (a few minutes; on Mac, if macOS blocks
  the file, right-click → Open). Click **Allow** on the camera prompt.
- **Every launch after that** starts in seconds.
- **After an update** (`git pull` / re-download) it notices and re-runs
  just the setup steps that are needed.

**Nothing to pre-install on either platform.** If no suitable Python is
found, the launcher downloads `uv` (official installer, into your user
folder, no admin password) which fetches a self-contained Python — 3.12
on Mac, 3.13 on Windows. First run needs internet.

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
- **Previews** (the panel thumbnail and the window on the facetrack
  machine) can be switched off during the show to save processing — the
  NDI/Syphon/Spout feeds keep running. With *Clean camera feed* on and
  previews off, the annotation pass is skipped entirely.
- **Process card**: *Pause* keeps the feeds up with a STANDBY slate (the
  overlay feed goes fully transparent, so keyed graphics vanish cleanly);
  *Restart* relaunches the app in place with saved settings and the panel
  reconnects itself; *Quit* shuts down — start again with the launcher on
  the machine (the panel reconnects automatically when you do).
- **If the input dies mid-show** (unplugged camera, dead NDI feed) the
  outputs cut to plain black — graceful on a live screen — while the
  panel preview shows a NO SIGNAL slate and the header a red pill.
  facetrack reconnects automatically the moment the source returns.
- **Stat chips change colour** when things get tight: processing time
  turns amber past 20 ms and red past the 30fps frame budget (33 ms);
  fps warns below 27 and alarms below 20.
- **PIN protection**: on a shared production network, launch with
  `--pin 4721` (or add `"pin": "4721"` to `settings.json`). The panel then
  asks once per browser; without it, controls, preview and source listing
  are locked. No PIN set = open panel (fine at home).
- If the input dies or is missing, the app keeps running and shows a
  NO INPUT slate — fix the source from the panel.

### If something's wrong

- **It crashed?** The launcher restarts it automatically after 3 seconds
  (clean quits don't restart). Everything the app printed is also in
  `logs/facetrack.log` for after-the-fact diagnosis.
- **It's wedged?** Delete the `.venv` folder and double-click the
  Facetrack file again — it rebuilds everything. Or, from a terminal:

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
| Faces cutout feed | the picture **only inside detected face boxes**, transparent everywhere else | *Cutout margin* slider adds headroom around each box |
| Syphon (macOS) / Spout (Windows) | full picture, graphics overlay, **or** faces cutout — pick in the panel | zero-compression, GPU-to-GPU, same machine only |
| Output size | Match input / 1920 / 1280 / 960 wide | applies to all feeds; lowers network load |
| Test card | SMPTE-style bars, ramp, feed identity, clock + moving block | motion proves the chain is live, not frozen; alpha feeds get a bracket/crosshair pattern instead; works with no input connected |

Feed *names* are fixed at launch (`--ndi-name`, `--ndi-overlay`; the
overlay defaults to "<name> Overlay", the cutout to "<name> Faces")
because renaming mid-show would drop receivers. NDI's codec is lossy, so keyed NDI graphics gain a pixel of
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

### Running two instances (e.g. two cameras)

Each instance needs its own panel port and NDI name:

```bash
python main.py --source 0 --ndi-name "FaceTracker A" --web-port 8089
python main.py --source 1 --ndi-name "FaceTracker B" --web-port 8090
```

Settings are shared per folder — for fully independent settings, keep a
second clone of the repo.

### Privacy

facetrack detects and follows faces; it performs no identity recognition,
no matching against any database, and records nothing — frames are
processed and discarded in memory. Expression labels are a cosmetic
overlay estimate. For public events, follow your usual venue practice on
camera signage, and keep this paragraph handy for client conversations.

### Known quirks

- The installed NDI HX driver makes OpenCV's ffmpeg print an `objc`
  duplicate-class warning at startup on this Mac. Harmless.
- The panel has no authentication — it's meant for a closed production
  LAN. Use `--web-host 127.0.0.1` to keep it local-only.
- macOS camera permission belongs to the *terminal app* that launches
  facetrack; grant it once when prompted.
