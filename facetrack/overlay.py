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
                show_ids: bool = True, color: tuple | None = None) -> None:
    """color: BGR override for every box (brand colour); None = palette."""
    H, W = frame.shape[:2]
    th = max(1, round(W / 1100))
    fscale = max(0.4, W / 2600)
    for t in tracks:
        x, y, w, h = t.bbox
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W - 1, x + w)), int(min(H - 1, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        color_t = _col(frame, color or PALETTE[t.id % len(PALETTE)])
        L = max(4, int(min(x2 - x1, y2 - y1) * 0.28))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color_t, th + 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color_t, th + 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_t, 1, cv2.LINE_AA)

        parts = []
        if show_ids:
            parts.append(f"#{t.id}")
        if show_emotion and t.emotion is not None:
            parts.append(t.emotion[0])
        if parts:
            label = " ".join(parts)
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, th)
            ly = y1 - 6 if y1 - tht - 10 > 0 else y2 + tht + 8
            cv2.rectangle(frame, (x1, ly - tht - 4), (x1 + tw + 6, ly + 4), color_t, -1)
            cv2.putText(frame, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        fscale, _col(frame, (20, 20, 20)), th, cv2.LINE_AA)


def render_overlay_bgra(shape_hw: tuple[int, int], tracks: list[Track],
                        show_emotion: bool = True, show_ids: bool = True,
                        color: tuple | None = None) -> np.ndarray:
    """Graphics-only frame on transparency for keying: BGRA where undrawn
    pixels are (0,0,0,0). Alpha is drawn directly with the graphics (no
    full-frame post-pass — that cost 12-28 ms). Drawn-on-black with alpha
    satisfies NDI's premultiplied convention and survives downscaling."""
    H, W = shape_hw
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    draw_tracks(canvas, tracks, show_emotion=show_emotion, show_ids=show_ids,
                color=color)
    return canvas


def render_faces_cutout(frame: np.ndarray, tracks: list[Track],
                        margin: float = 0.15, shape: str = "rectangle",
                        feather: int = 0,
                        people_mask: np.ndarray | None = None) -> np.ndarray:
    """The picture only inside the cutout mask; transparent elsewhere.

    shape: 'rectangle' / 'oval' (per-face, margin-grown) or 'people'
    (pass the segmenter's full-frame mask via people_mask). feather
    softens the mask edge (Gaussian, px). Output is premultiplied BGRA —
    soft edges multiply the picture down so NDI/Syphon key cleanly.

    Face shapes do all their work inside each face's own region (faces
    are a small fraction of the frame; full-frame math costs 20+ ms at
    1080p). The people path is inherently full-frame but uses OpenCV's
    SIMD multiply."""
    H, W = frame.shape[:2]

    if shape == "people" and people_mask is not None:
        # The segmenter's mask is upscaled from 192px, which bakes a mushy
        # ~15px ramp into the edge — feathering on top of that was barely
        # visible. Re-harden at 50% first (the isoline of the upscaled
        # field is smooth), then feather by the requested amount; the
        # 3px minimum anti-aliases the re-hardened contour.
        _, alpha = cv2.threshold(people_mask, 127, 255, cv2.THRESH_BINARY)
        k = max(feather, 3) | 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        a3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        b, g, r = cv2.split(cv2.multiply(frame, a3, scale=1 / 255.0))
        return cv2.merge((b, g, r, alpha))

    out = np.zeros((H, W, 4), dtype=np.uint8)
    oval = shape == "oval"
    for t in tracks:
        x, y, w, h = t.bbox
        mx, my = w * margin, h * margin
        # region: the grown box plus room for the feather to fade out
        x1 = int(max(0, x - mx - feather))
        y1 = int(max(0, y - my - feather))
        x2 = int(min(W, x + w + mx + feather))
        y2 = int(min(H, y + h + my + feather))
        if x2 <= x1 or y2 <= y1:
            continue
        if not oval and feather == 0:
            out[y1:y2, x1:x2, :3] = frame[y1:y2, x1:x2]
            out[y1:y2, x1:x2, 3] = 255
            continue
        m = np.zeros((y2 - y1, x2 - x1), np.uint8)
        if oval:
            cv2.ellipse(m, (int(x + w / 2 - x1), int(y + h / 2 - y1)),
                        (max(1, int(w / 2 + mx)), max(1, int(h / 2 + my))),
                        0, 0, 360, 255, -1)
        else:
            gx1 = max(0, int(round(x - mx)) - x1)
            gy1 = max(0, int(round(y - my)) - y1)
            gx2 = min(x2 - x1, int(round(x + w + mx)) - x1)
            gy2 = min(y2 - y1, int(round(y + h + my)) - y1)
            m[gy1:gy2, gx1:gx2] = 255
        if feather > 0:
            k = feather | 1
            m = cv2.GaussianBlur(m, (k, k), 0)
        a16 = m.astype(np.uint16)[..., None]
        prem = ((frame[y1:y2, x1:x2].astype(np.uint16) * a16) // 255).astype(np.uint8)
        # overlapping faces: max, so cutouts never darken each other
        np.maximum(out[y1:y2, x1:x2, :3], prem, out=out[y1:y2, x1:x2, :3])
        np.maximum(out[y1:y2, x1:x2, 3], m, out=out[y1:y2, x1:x2, 3])
    return out


def render_test_card(w: int, h: int, lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Static test-card templates: (program card BGR, alpha-test BGRA).

    Program card: SMPTE-style 75% bars, a grayscale ramp, and identity
    text. Alpha card: corner brackets + crosshair + label on transparency,
    for verifying the keyed feeds. The pipeline stamps a moving block and
    clock on copies each tick, so motion proves the chain is live."""
    card = np.zeros((h, w, 3), dtype=np.uint8)
    bars = [(235, 235, 235), (0, 235, 235), (235, 235, 0), (0, 235, 0),
            (235, 0, 235), (0, 0, 235), (235, 0, 0)]  # BGR: white..blue
    bar_h = int(h * 0.58)
    for i, c in enumerate(bars):
        x1 = int(w * i / len(bars))
        x2 = int(w * (i + 1) / len(bars))
        card[0:bar_h, x1:x2] = c
    ramp_y2 = int(h * 0.72)
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (ramp_y2 - bar_h, 1))
    card[bar_h:ramp_y2] = ramp[:, :, None]
    fscale = max(0.5, w / 1600)
    th = max(1, round(w / 1000))
    y = int(h * 0.80)
    for i, line in enumerate(lines):
        cv2.putText(card, line, (int(w * 0.03), y + i * int(44 * fscale + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale * (1.5 if i == 0 else 1.0),
                    (235, 235, 235), th, cv2.LINE_AA)

    ovl = np.zeros((h, w, 4), dtype=np.uint8)
    white = (255, 255, 255, 255)
    L, m, t2 = int(min(w, h) * 0.09), int(min(w, h) * 0.04), max(2, th + 1)
    for cx, cy, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1),
                           (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(ovl, (cx, cy), (cx + dx * L, cy), white, t2, cv2.LINE_AA)
        cv2.line(ovl, (cx, cy), (cx, cy + dy * L), white, t2, cv2.LINE_AA)
    cv2.line(ovl, (w // 2 - L, h // 2), (w // 2 + L, h // 2), white, t2, cv2.LINE_AA)
    cv2.line(ovl, (w // 2, h // 2 - L), (w // 2, h // 2 + L), white, t2, cv2.LINE_AA)
    label = "facetrack ALPHA TEST"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, th)
    cv2.putText(ovl, label, ((w - tw) // 2, h // 2 - L - 12),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, white, th, cv2.LINE_AA)
    return card, ovl


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
