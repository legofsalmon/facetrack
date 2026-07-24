"""Broadcast-style overlay rendering: corner-bracket boxes, ID/emotion
labels, and a stats bar. Pure OpenCV drawing, negligible cost."""
from __future__ import annotations

import cv2
import numpy as np

from .tracker import Track

# Distinct, bright BGR colors, assigned per track ID.
PALETTE = [
    (80, 220, 80), (60, 170, 255), (255, 160, 60), (200, 80, 255),
    (60, 255, 255), (255, 90, 150), (120, 255, 170), (255, 220, 90),
    (90, 130, 255), (230, 230, 230),
]


def _col(frame: np.ndarray, bgr: tuple) -> tuple:
    """Drawing color for a 3- or 4-channel target: on BGRA canvases the
    alpha rides along in the color, so transparency needs no post-pass."""
    return (*bgr, 255) if frame.shape[2] == 4 else bgr


def draw_tracks(frame: np.ndarray, tracks: list[Track], show_emotion: bool = True,
                show_ids: bool = True) -> None:
    H, W = frame.shape[:2]
    th = max(1, round(W / 1100))
    fscale = max(0.4, W / 2600)
    for t in tracks:
        x, y, w, h = t.bbox
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W - 1, x + w)), int(min(H - 1, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _col(frame, PALETTE[t.id % len(PALETTE)])
        L = max(4, int(min(x2 - x1, y2 - y1) * 0.28))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, th + 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, th + 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        parts = []
        if show_ids:
            parts.append(f"#{t.id}")
        if show_emotion and t.emotion is not None:
            parts.append(t.emotion[0])
        if parts:
            label = " ".join(parts)
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, th)
            ly = y1 - 6 if y1 - tht - 10 > 0 else y2 + tht + 8
            cv2.rectangle(frame, (x1, ly - tht - 4), (x1 + tw + 6, ly + 4), color, -1)
            cv2.putText(frame, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        fscale, _col(frame, (20, 20, 20)), th, cv2.LINE_AA)


def render_overlay_bgra(shape_hw: tuple[int, int], tracks: list[Track],
                        show_emotion: bool = True, show_ids: bool = True) -> np.ndarray:
    """Graphics-only frame on transparency for keying: BGRA where undrawn
    pixels are (0,0,0,0). Alpha is drawn directly with the graphics (no
    full-frame post-pass — that cost 12-28 ms). Drawn-on-black with alpha
    satisfies NDI's premultiplied convention and survives downscaling."""
    H, W = shape_hw
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    draw_tracks(canvas, tracks, show_emotion=show_emotion, show_ids=show_ids)
    return canvas


def render_faces_cutout(frame: np.ndarray, tracks: list[Track],
                        margin: float = 0.15) -> np.ndarray:
    """The picture only inside the detected face boxes; everything else is
    transparent (BGRA, alpha 0). `margin` grows each box by that fraction
    of its size on every side. Full alpha inside the boxes keeps NDI's
    premultiplied convention trivially satisfied."""
    H, W = frame.shape[:2]
    out = np.zeros((H, W, 4), dtype=np.uint8)
    for t in tracks:
        x, y, w, h = t.bbox
        mx, my = w * margin, h * margin
        x1, y1 = int(max(0, x - mx)), int(max(0, y - my))
        x2, y2 = int(min(W, x + w + mx)), int(min(H, y + h + my))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2, :3] = frame[y1:y2, x1:x2]
        out[y1:y2, x1:x2, 3] = 255
    return out


def draw_stats(frame: np.ndarray, lines: list[str]) -> None:
    W = frame.shape[1]
    fscale = max(0.45, W / 2400)
    th = max(1, round(W / 1400))
    pad = 8
    sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fscale, th)[0] for s in lines]
    bw = max(s[0] for s in sizes) + 2 * pad
    lh = max(s[1] for s in sizes) + 10
    bh = lh * len(lines) + pad
    sub = frame[0:bh, 0:bw]
    cv2.addWeighted(sub, 0.35, np.zeros_like(sub), 0.65, 0, dst=sub)
    y = pad + sizes[0][1]
    for s in lines:
        cv2.putText(frame, s, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fscale, (240, 240, 240), th, cv2.LINE_AA)
        y += lh
