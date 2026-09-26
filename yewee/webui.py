"""Web control panel: FastAPI app served from the pipeline process.

- GET  /             the control page (single static HTML file)
- WS   /ws           client -> {"type":"set","data":{param:value}} or
                     {"type":"source","data":"<spec>"};
                     server -> {"type":"tick","stats":{...},"params":{...}}
                     every ~0.5s (keeps multiple clients in sync)
- GET  /preview.mjpg throttled MJPEG preview (~12 fps, 640px wide)

The PIN: other devices give it once (the WebSocket's "auth" message, and
?pin= on the plain GETs). The machine yewee runs on does not: a loopback
peer that also asked for a loopback host name, from a loopback page, is
let in without it and is the only client told the PIN, so the operator
can read it off the screen. The host-name and Origin checks keep a web
page open in that same browser from riding on the exemption. Five wrong
PINs from one address refuse it for a minute, so four digits cannot be
walked through.

NOTE: no `from __future__ import annotations` here — the WebSocket type
hint must be a real class (FastAPI resolves it for dependency injection,
and the import lives inside create_app).
"""
import asyncio
import hmac
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from . import app_version, reporting
from .params import LiveParams
from .pipeline import Pipeline

STATIC_DIR = Path(__file__).parent / "static"

_LOOPBACK_PEERS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}
MAX_WRONG_PINS = 5          # per address, before it is refused for a while
LOCKOUT_SECONDS = 60


def _hostname(netloc: str) -> str:
    try:
        return (urlsplit("//" + netloc).hostname or "").lower()
    except ValueError:
        return ""


def is_local(conn) -> bool:
    """True for the panel opened on the yewee machine itself: the peer is
    loopback, the Host header names loopback (not a rebound DNS name), and
    any Origin is a loopback page (not another site's script)."""
    client = getattr(conn, "client", None)
    if client is None or client.host not in _LOOPBACK_PEERS:
        return False
    if _hostname(conn.headers.get("host", "")) not in _LOOPBACK_NAMES:
        return False
    origin = conn.headers.get("origin")
    if origin is not None:
        try:
            parts = urlsplit(origin)
        except ValueError:
            return False
        if (parts.hostname or "").lower() not in _LOOPBACK_NAMES:
            return False
    return True


class PinGate:
    """Checks a supplied PIN and throttles an address that keeps guessing."""

    def __init__(self, pin: str, clock=time.monotonic):
        self.pin = pin or ""
        self._clock = clock
        self._lock = threading.Lock()
        self._wrong: dict[str, int] = {}
        self._until: dict[str, float] = {}

    def locked_for(self, address: str) -> float:
        with self._lock:
            return max(0.0, self._until.get(address, 0.0) - self._clock())

    def check(self, conn, supplied) -> tuple[int, str] | None:
        """None when this connection may proceed, else (status, message).
        An empty or missing PIN is only "PIN required", never a strike: a
        phone asks for the preview before the person has typed anything."""
        if not self.pin or is_local(conn):
            return None
        address = conn.client.host if conn.client else "?"
        wait = self.locked_for(address)
        if wait > 0:
            return 429, f"Too many wrong PINs. Try again in {int(wait) + 1} s."
        supplied = "" if supplied is None else str(supplied)
        if not supplied:
            return 401, "PIN required"
        if hmac.compare_digest(supplied.encode("utf-8"), self.pin.encode("utf-8")):
            with self._lock:
                self._wrong.pop(address, None)
            return None
        with self._lock:
            n = self._wrong.get(address, 0) + 1
            if n >= MAX_WRONG_PINS:
                self._wrong.pop(address, None)
                self._until[address] = self._clock() + LOCKOUT_SECONDS
                # the address, never what was typed
                print(f"[yewee] panel: {MAX_WRONG_PINS} wrong PINs from {address}; "
                      f"refusing it for {LOCKOUT_SECONDS} s", flush=True)
            else:
                self._wrong[address] = n
        return 401, "Wrong PIN"


def create_app(pipeline: Pipeline, params: LiveParams, on_params_change=None,
               pin: str = "", phone_url: str | None = None):
    """pin: what other devices must give ("" = none). phone_url: the address
    phones use, shown with the PIN on the machine's own panel."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    from .capture import _camera_names, camera_authorization, probe_cameras

    app = FastAPI(title="yewee")
    index_html = (STATIC_DIR / "index.html").read_text()

    def _licence_for_feedback() -> str:
        """The shop key a feedback form may offer to include: only a bought
        (not trial) letissier.ie licence, which the studio can resolve."""
        lic = pipeline.licence()
        if lic.get("state") == "licensed" and lic.get("source") == "shop":
            return str(lic.get("key") or "")
        return ""

    gate = PinGate(pin)
    app.state.pin_gate = gate

    def _refused(request: Request):
        denied = gate.check(request, request.query_params.get("pin"))
        if denied is None:
            return None
        return PlainTextResponse(denied[1], status_code=denied[0])

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
        refused = _refused(request)
        if refused is not None:
            return refused
        current = pipeline.source_spec
        in_use = int(current) if current.isdigit() else None
        if current.lower().startswith("cam:"):
            try:
                from .capture import resolve_camera
                in_use = resolve_camera(current[4:])
            except Exception:
                in_use = None
        try:
            cameras = probe_cameras(
                backend=params.snapshot().get("cap_backend", "any"),
                skip=in_use)
        except Exception:
            cameras = []
        # Report the current source as the same spec the picker offers, so
        # a camera opened by bare index still highlights its cam:<name> row.
        if in_use is not None:
            for c in cameras:
                if c["index"] == in_use:
                    current = c["spec"]
                    break
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

    @app.get("/notices")
    def third_party_notices():
        """The licences of everything yewee ships with. Open to anyone who
        can reach the panel, PIN or not: it holds nothing about this show."""
        from . import notices
        return PlainTextResponse(notices.text())

    @app.get("/logs")
    def logs(request: Request):
        refused = _refused(request)
        if refused is not None:
            return refused
        from .paths import log_path as _log_path
        log_path = _log_path()
        try:
            lines = log_path.read_text(errors="replace").splitlines()[-200:]
            return PlainTextResponse("\n".join(lines) or "log is empty")
        except OSError:
            return PlainTextResponse("no log file yet")

    @app.get("/preview.mjpg")
    def preview(request: Request):
        refused = _refused(request)
        if refused is not None:
            return refused
        boundary = b"--frame"

        def gen():
            last = -1
            # At zero viewers the pipeline skips the preview's render work
            # entirely, not just the JPEG encode.
            pipeline.add_preview_client(+1)
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
                pipeline.add_preview_client(-1)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        local = is_local(sock)
        if pin and not local:
            address = sock.client.host if sock.client else "?"
            wait = gate.locked_for(address)
            if wait > 0:
                await sock.send_text(json.dumps(
                    {"type": "auth_locked", "seconds": int(wait) + 1}))
                await sock.close(code=4002)
                return
            await sock.send_text(json.dumps({"type": "auth_required"}))
            try:
                msg = json.loads(await asyncio.wait_for(sock.receive_text(), timeout=60))
                supplied = msg.get("data") if msg.get("type") == "auth" else None
            except (asyncio.TimeoutError, ValueError, AttributeError, WebSocketDisconnect):
                await sock.close(code=4001)
                return
            denied = gate.check(sock, "" if supplied is None else str(supplied))
            if denied is not None:
                await sock.close(code=4002 if denied[0] == 429 else 4001)
                return
        # Only the machine's own panel is told the PIN and the phone address.
        phones = {"pin": pin, "url": phone_url} if local else None
        last_tick = 0.0
        try:
            while not pipeline.stopped:
                try:
                    msg = await asyncio.wait_for(sock.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    msg = None
                kind = None
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
                    elif kind == "licence":
                        d = data.get("data") or {}
                        act = d.get("action")
                        try:
                            from .licensing import activate, deactivate
                            # both may wait on letissier.ie, so they run off
                            # the event loop that keeps the panel ticking
                            if act == "activate":
                                ok, note = await asyncio.to_thread(
                                    activate, str(d.get("key", "")))
                            elif act == "deactivate":
                                note = await asyncio.to_thread(deactivate)
                                ok = True
                            else:
                                ok, note = False, "Unknown licence action."
                        except Exception as exc:
                            ok, note = False, f"Licence error: {exc}"
                        pipeline.refresh_licence()
                        await sock.send_text(json.dumps(
                            {"type": "licence_result", "ok": ok, "message": note}))
                    elif kind == "reports":
                        d = data.get("data") or {}
                        if "auto_send" in d:
                            reporting.set_auto_send(bool(d["auto_send"]))
                    elif kind == "crash_prompt":
                        d = data.get("data") or {}
                        try:
                            note = await asyncio.to_thread(
                                reporting.answer_prompt, bool(d.get("send")),
                                bool(d.get("always")), str(d.get("note") or ""))
                            ok = True
                        except Exception as exc:
                            ok, note = False, f"Couldn't do that: {exc}"
                        await sock.send_text(json.dumps(
                            {"type": "crash_prompt_result", "ok": ok, "message": note}))
                    elif kind == "feedback":
                        d = data.get("data") or {}
                        # the key is looked up here, never taken from the
                        # browser, and only when the person ticked the box
                        key = _licence_for_feedback() if d.get("licence") else ""
                        try:
                            ok, note = await asyncio.to_thread(
                                reporting.submit_feedback, str(d.get("type") or ""),
                                str(d.get("message") or ""), str(d.get("email") or ""),
                                key, bool(d.get("public")))
                        except Exception as exc:
                            ok, note = False, f"Couldn't send that: {exc}"
                        await sock.send_text(json.dumps(
                            {"type": "feedback_result", "ok": ok, "message": note}))
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
                # Ticks keep their own cadence rather than answering every
                # inbound message: a slider drag arrives as a burst of
                # `set` messages, and replying to each with a full stats
                # frame turned one drag into a flood of them. A control
                # action still gets an immediate tick, because the panel
                # is waiting to see the state actually change.
                now = time.monotonic()
                if kind == "control" or now - last_tick >= 0.25:
                    last_tick = now
                    await sock.send_text(json.dumps({
                        "type": "tick",
                        "stats": pipeline.get_stats(),
                        "params": params.snapshot(),
                        "version": app_version(),
                        "reports": reporting.panel_state(),
                        "licence_offer": bool(_licence_for_feedback()),
                        **({"phones": phones} if phones else {}),
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

    thread = threading.Thread(target=run, daemon=True, name="yewee-web")
    thread.start()
    return server
