"""Web control panel: FastAPI app served from the pipeline process.

- GET  /             the control page (single static HTML file)
- WS   /ws           client -> {"type":"set","data":{param:value}} or
                     {"type":"source","data":"<spec>"};
                     server -> {"type":"tick","stats":{...},"params":{...}}
                     every ~0.5s (keeps multiple clients in sync)
- GET  /preview.mjpg throttled MJPEG preview (~12 fps, 640px wide)

NOTE: no `from __future__ import annotations` here — the WebSocket type
hint must be a real class (FastAPI resolves it for dependency injection,
and the import lives inside create_app).
"""
import asyncio
import json
import threading
from pathlib import Path

from .params import LiveParams
from .pipeline import Pipeline

STATIC_DIR = Path(__file__).parent / "static"


def create_app(pipeline: Pipeline, params: LiveParams, on_params_change=None,
               pin: str = ""):
    import hmac

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    from .capture import _camera_names, camera_authorization, probe_cameras

    app = FastAPI(title="facetrack")
    index_html = (STATIC_DIR / "index.html").read_text()

    def _pin_ok(request: Request) -> bool:
        if not pin:
            return True
        supplied = request.query_params.get("pin", "")
        return hmac.compare_digest(supplied, pin)

    @app.get("/")
    def index():
        return HTMLResponse(index_html)

    @app.get("/sources")
    def sources(request: Request):
        """Selectable inputs for the panel, scanned live on each call:
        connected cameras/system video devices (with real names where the
        OS provides them) + NDI sources on the network (minus our own
        outputs). The camera the pipeline is using is reported without
        being re-opened."""
        if not _pin_ok(request):
            return PlainTextResponse("PIN required", status_code=401)
        current = pipeline.source_spec
        in_use = int(current) if current.isdigit() else None
        try:
            cameras = probe_cameras(
                backend=params.snapshot().get("cap_backend", "any"),
                skip=in_use)
        except Exception:
            cameras = []
        # If nothing opened because macOS blocks this process, still list
        # the devices the OS knows about so the picker can explain itself.
        camera_auth = camera_authorization()
        blocked = []
        if not cameras and camera_auth in ("denied", "undetermined", "restricted"):
            try:
                blocked = [n for n in _camera_names() if n]
            except Exception:
                blocked = []
        ndi_names = []
        try:
            from cyndilib.finder import Finder
            f = Finder()
            f.open()
            f.wait_for_sources(timeout=1.5)
            own = {n for n in (getattr(pipeline.args, "ndi_name", ""),
                               getattr(pipeline.args, "ndi_overlay", "")) if n}
            ndi_names = [n for n in f.get_source_names()
                         if not any(o in n for o in own)]
            f.close()
        except Exception:
            pass
        return JSONResponse({"cameras": cameras, "ndi": ndi_names,
                             "current": current,
                             "camera_auth": camera_auth,
                             "blocked_cameras": blocked})

    @app.get("/logs")
    def logs(request: Request):
        if not _pin_ok(request):
            return PlainTextResponse("PIN required", status_code=401)
        log_path = STATIC_DIR.parent.parent / "logs" / "facetrack.log"
        try:
            lines = log_path.read_text(errors="replace").splitlines()[-200:]
            return PlainTextResponse("\n".join(lines) or "log is empty")
        except OSError:
            return PlainTextResponse("no log file yet")

    @app.get("/preview.mjpg")
    def preview(request: Request):
        if not _pin_ok(request):
            return PlainTextResponse("PIN required", status_code=401)
        boundary = b"--frame"

        def gen():
            last = -1
            pipeline.preview_clients += 1  # JPEG encoding pauses at zero viewers
            try:
                while not pipeline.stopped:
                    item = pipeline.wait_preview(last, timeout=1.0)
                    if item is None:
                        continue
                    last, jpg = item
                    yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                           + jpg + b"\r\n")
            finally:
                pipeline.preview_clients -= 1

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if pin:
            await sock.send_text(json.dumps({"type": "auth_required"}))
            try:
                msg = json.loads(await asyncio.wait_for(sock.receive_text(), timeout=60))
            except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
                await sock.close(code=4001)
                return
            if msg.get("type") != "auth" or not hmac.compare_digest(
                    str(msg.get("data", "")), pin):
                await sock.close(code=4001)
                return
        try:
            while not pipeline.stopped:
                try:
                    msg = await asyncio.wait_for(sock.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    msg = None
                if msg is not None:
                    try:  # a malformed message must not kill the socket
                        data = json.loads(msg)
                        kind = data.get("type")
                    except (ValueError, AttributeError):
                        data, kind = {}, None
                    if kind == "set":
                        changed = False
                        for k, v in dict(data.get("data", {}) or {}).items():
                            try:
                                params.set(k, v)
                                changed = True
                            except (KeyError, TypeError, ValueError):
                                pass
                        if changed and on_params_change is not None:
                            on_params_change(params.snapshot())
                    elif kind == "source":
                        pipeline.request_source(str(data.get("data", "")))
                    elif kind == "control":
                        action = data.get("data")
                        if action == "pause":
                            pipeline.paused = True
                        elif action == "resume":
                            pipeline.paused = False
                        elif action == "restart":
                            pipeline.request_restart()
                        elif action == "quit":
                            pipeline.stop()
                await sock.send_text(json.dumps({
                    "type": "tick",
                    "stats": pipeline.get_stats(),
                    "params": params.snapshot(),
                }))
        except WebSocketDisconnect:
            pass

    return app


def start_in_thread(app, host: str, port: int):
    import socket
    import time

    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def run():
        # After an in-place Restart the previous process may still be
        # releasing the port; wait for it briefly instead of losing the
        # panel to a bind race.
        for _ in range(40):
            try:
                probe = socket.socket()
                probe.bind(("" if host == "0.0.0.0" else host, port))
                probe.close()
                break
            except OSError:
                time.sleep(0.25)
        server.run()

    thread = threading.Thread(target=run, daemon=True, name="facetrack-web")
    thread.start()
    return server
