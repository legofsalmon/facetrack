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


def create_app(pipeline: Pipeline, params: LiveParams, on_params_change=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    from .capture import probe_cameras

    app = FastAPI(title="facetrack")
    index_html = (STATIC_DIR / "index.html").read_text()

    @app.get("/")
    def index():
        return HTMLResponse(index_html)

    @app.get("/sources")
    def sources():
        """Selectable inputs for the panel, scanned live on each call:
        connected cameras/system video devices (with real names where the
        OS provides them) + NDI sources on the network (minus our own
        outputs). The camera the pipeline is using is reported without
        being re-opened."""
        current = pipeline.source_spec
        in_use = int(current) if current.isdigit() else None
        try:
            cameras = probe_cameras(
                backend=getattr(pipeline.args, "capture_backend", "any"),
                skip=in_use)
        except Exception:
            cameras = []
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
                             "current": current})

    @app.get("/preview.mjpg")
    def preview():
        boundary = b"--frame"

        def gen():
            last = -1
            while not pipeline.stopped:
                item = pipeline.wait_preview(last, timeout=1.0)
                if item is None:
                    continue
                last, jpg = item
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while not pipeline.stopped:
                try:
                    msg = await asyncio.wait_for(sock.receive_text(), timeout=0.5)
                    data = json.loads(msg)
                    kind = data.get("type")
                    if kind == "set":
                        changed = False
                        for k, v in dict(data.get("data", {})).items():
                            try:
                                params.set(k, v)
                                changed = True
                            except (KeyError, TypeError, ValueError):
                                pass
                        if changed and on_params_change is not None:
                            on_params_change(params.snapshot())
                    elif kind == "source":
                        pipeline.request_source(str(data.get("data", "")))
                except asyncio.TimeoutError:
                    pass
                await sock.send_text(json.dumps({
                    "type": "tick",
                    "stats": pipeline.get_stats(),
                    "params": params.snapshot(),
                }))
        except WebSocketDisconnect:
            pass

    return app


def start_in_thread(app, host: str, port: int):
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="facetrack-web")
    thread.start()
    return server
