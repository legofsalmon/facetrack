"""OSC data output: the track table as coordinates instead of pixels.

yewee's product is where the faces are, and every other output ships that
as video — at 1080p the overlay feed carries about a kilobyte of
coordinates as 8.3 MB of pixels per frame. This sends the numbers
themselves over OSC/UDP, so Resolume, TouchDesigner, Notch and MadMapper
can draw at their own output resolution with nothing to key and no second
NDI feed to route. It is additive: the pixel feeds stay for the chains
that need matted video.

The OSC 1.0 encoder here is hand-rolled on purpose. It is thirty lines of
struct packing, and an extra dependency inside a signed, licensed
installer costs more than that.

Addresses — see the README for the full table:

    /yewee/faces            int    faces tracked right now
    /yewee/crowd            float  faces / slots, clamped to 1
    /yewee/center/x, /y     float  centre of everybody
    /yewee/fps              float  the loop's current rate
    /yewee/face/N/active    int    1 while slot N holds somebody
    /yewee/face/N/id        int    the track number yewee draws on screen
    /yewee/face/N/x, /y     float  centre of the face box
    /yewee/face/N/w, /h     float  its size
    /yewee/face/N/size      float  the larger of the two, as one "how close"
    /yewee/face/N/expression  str  FER+ label, "" when expressions are off
    /yewee/face/N/confidence  float  how sure that label is

Faces live in fixed SLOTS, not in a list: slot 1 keeps the same person for
as long as that person is tracked, which is what lets a mapping in a VJ
app hold still. Every slot is sent on every update, occupied or not, so a
receiver that starts late settles on the first packet instead of waiting
for somebody to move.
"""
from __future__ import annotations

import socket
import struct
import time

_IMMEDIATELY = struct.pack(">Q", 1)  # OSC time tag meaning "now"
_INT32 = (-2 ** 31, 2 ** 31 - 1)


def _osc_string(text: str) -> bytes:
    """OSC string: NUL-terminated, then padded to a multiple of four."""
    raw = text.encode("utf-8", "replace") + b"\0"
    return raw + b"\0" * (-len(raw) % 4)


def encode_message(address: str, *args) -> bytes:
    """One OSC message. Supports int (i), float (f) and string (s) args —
    everything this feed carries."""
    tags = ""
    body = b""
    for a in args:
        if isinstance(a, bool) or isinstance(a, int):
            tags += "i"
            body += struct.pack(">i", max(_INT32[0], min(_INT32[1], int(a))))
        elif isinstance(a, float):
            tags += "f"
            body += struct.pack(">f", a)
        else:
            tags += "s"
            body += _osc_string(str(a))
    return _osc_string(address) + _osc_string("," + tags) + body


def encode_bundle(messages) -> bytes:
    """An OSC bundle timestamped "now" — one datagram for many messages."""
    parts = [_osc_string("#bundle"), _IMMEDIATELY]
    for m in messages:
        parts.append(struct.pack(">i", len(m)))
        parts.append(m)
    return b"".join(parts)


BUNDLE_HEADER = 16  # "#bundle\0" plus the 8-byte time tag


class OSCOutput:
    """Sends the current track table to one OSC/UDP endpoint.

    Never raises from send(): UDP is fire-and-forget and a data feed must
    not be able to take the show down. A send that fails leaves its reason
    in `error` for the panel.
    """

    PREFIX = "/yewee"
    #: Keep each datagram inside a 1500-byte MTU, headers included, so a
    #: busy stage with 32 slots never depends on IP fragmentation.
    MAX_DATAGRAM = 1200

    def __init__(self, host: str = "127.0.0.1", port: int = 7000,
                 slots: int = 8, units: str = "normalised", rate: float = 30.0):
        self.host = (str(host).strip() or "127.0.0.1")
        self.port = int(port)
        self.slots = max(1, int(slots))
        self.units = units if units in ("normalised", "pixels") else "normalised"
        self.rate = float(rate)
        self.error = ""
        self.packets = 0
        # Resolve once, at creation: a bad host should fail where the panel
        # can report it, not silently every frame.
        family, _, _, _, self._addr = socket.getaddrinfo(
            self.host, self.port, type=socket.SOCK_DGRAM)[0]
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            # lets an operator aim at a subnet broadcast to feed a rack of
            # machines from one instance; harmless for a unicast target
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        self._slot_of: dict[int, int] = {}  # track id -> slot number (1-based)
        self._next_at = 0.0

    @property
    def target(self) -> str:
        return f"{self.host}:{self.port}"

    # ---- rate ----

    def _due(self) -> bool:
        """True when the next update is owed. Fires a quarter-period early
        so that a loop running at exactly the requested rate sends every
        frame instead of alternate ones."""
        period = 1.0 / self.rate if self.rate > 0 else 0.0
        now = time.monotonic()
        if now + period * 0.25 < self._next_at:
            return False
        self._next_at = max(now, self._next_at) + period
        return True

    # ---- slots ----

    def _assign(self, tracks) -> dict:
        """Hold each track in the slot it first got. Returns {slot: track}."""
        live = {t.id: t for t in tracks}
        for tid, slot in list(self._slot_of.items()):
            if tid not in live or slot > self.slots:
                del self._slot_of[tid]  # gone, or the slot count shrank
        taken = set(self._slot_of.values())
        for tid in sorted(live):
            if tid in self._slot_of:
                continue
            free = next((s for s in range(1, self.slots + 1) if s not in taken), None)
            if free is None:
                break  # more faces than slots; the extras wait for one
            self._slot_of[tid] = free
            taken.add(free)
        return {slot: live[tid] for tid, slot in self._slot_of.items()}

    # ---- sending ----

    def send(self, tracks, size, fps: float = 0.0, force: bool = False) -> bool:
        """Publish the track table. `size` is (height, width) — frame.shape[:2].
        Returns whether an update was due; a datagram that then fails to
        leave the machine shows up in `error`, never as an exception."""
        if not (force or self._due()):
            return False
        h, w = int(size[0]), int(size[1])
        sx, sy = (1.0, 1.0) if self.units == "pixels" \
            else (1.0 / max(1, w), 1.0 / max(1, h))
        n = len(tracks)
        if n:
            px = sum(float(t.bbox[0]) + float(t.bbox[2]) * 0.5 for t in tracks) / n
            py = sum(float(t.bbox[1]) + float(t.bbox[3]) * 0.5 for t in tracks) / n
        else:
            px, py = w * 0.5, h * 0.5
        messages = [
            encode_message(f"{self.PREFIX}/faces", n),
            encode_message(f"{self.PREFIX}/crowd", min(1.0, n / self.slots)),
            encode_message(f"{self.PREFIX}/center/x", px * sx),
            encode_message(f"{self.PREFIX}/center/y", py * sy),
            encode_message(f"{self.PREFIX}/fps", float(fps)),
        ]
        occupied = self._assign(tracks)
        for slot in range(1, self.slots + 1):
            messages += self._slot_messages(slot, occupied.get(slot), sx, sy, w, h)
        self._flush(messages)
        return True

    def clear(self, size=None, force: bool = False) -> bool:
        """Publish an empty table — paused, no signal, test card. Sent at
        the same rate rather than once, so the feed stays a heartbeat and a
        receiver that connects mid-pause still gets zeroed."""
        return self.send([], size or (1080, 1920), 0.0, force=force)

    def _slot_messages(self, slot: int, track, sx: float, sy: float,
                       w: int, h: int) -> list:
        base = f"{self.PREFIX}/face/{slot}"
        if track is None:
            # Parked, not flung: an empty slot reports the middle of the
            # frame at zero size, so a graphic bound to it sits still while
            # `active` is 0 instead of flying to the corner.
            return [
                encode_message(f"{base}/active", 0),
                encode_message(f"{base}/id", 0),
                encode_message(f"{base}/x", w * 0.5 * sx),
                encode_message(f"{base}/y", h * 0.5 * sy),
                encode_message(f"{base}/w", 0.0),
                encode_message(f"{base}/h", 0.0),
                encode_message(f"{base}/size", 0.0),
                encode_message(f"{base}/expression", ""),
                encode_message(f"{base}/confidence", 0.0),
            ]
        x, y, bw, bh = (float(v) for v in track.bbox)
        label, conf = track.emotion or ("", 0.0)
        return [
            encode_message(f"{base}/active", 1),
            encode_message(f"{base}/id", int(track.id)),
            encode_message(f"{base}/x", (x + bw * 0.5) * sx),
            encode_message(f"{base}/y", (y + bh * 0.5) * sy),
            encode_message(f"{base}/w", bw * sx),
            encode_message(f"{base}/h", bh * sy),
            encode_message(f"{base}/size", max(bw * sx, bh * sy)),
            encode_message(f"{base}/expression", label),
            encode_message(f"{base}/confidence", float(conf)),
        ]

    def _flush(self, messages) -> None:
        """Pack into MTU-sized bundles and put them on the wire."""
        batch, size = [], BUNDLE_HEADER
        for m in messages:
            if batch and size + 4 + len(m) > self.MAX_DATAGRAM:
                self._sendto(encode_bundle(batch))
                batch, size = [], BUNDLE_HEADER
            batch.append(m)
            size += 4 + len(m)
        if batch:
            self._sendto(encode_bundle(batch))

    def _sendto(self, data: bytes) -> None:
        try:
            self._sock.sendto(data, self._addr)
            self.packets += 1
            self.error = ""
        except OSError as exc:
            self.error = str(exc)  # report it; never let it reach the loop

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
