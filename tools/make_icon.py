"""Draws the yewee app icon and writes every format the builds need.

    python tools/make_icon.py

Produces assets/yewee.png (1024), yewee.icns (macOS) and yewee.ico
(Windows). Kept as code rather than a binary blob so the shape can be
adjusted and every size stays consistent — an icon that is hand-edited at
one size and scaled to the rest goes muddy at 16px.

Everything is drawn at 4x and downsampled, which is cheaper than fighting
PIL for antialiased curves.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

SIZE = 1024
SS = 4  # supersample factor

BG = (16, 20, 27, 255)       # near-black, matches the control panel
GREEN = (80, 220, 96, 255)   # tracking brackets
BLUE = (86, 168, 255, 255)   # the subject

# Fractions of the canvas, so the geometry survives any SIZE change.
CORNER = 0.205               # rounded-square radius
BRACKET_INSET = 0.122
BRACKET_ARM = 0.259          # how far each arm reaches from the corner
BRACKET_THICK = 0.054
BRACKET_ROUND = 0.012

HEAD_CY, HEAD_R = 0.378, 0.113
SHOULDER_TOP, SHOULDER_BOTTOM = 0.525, 0.742   # bottom is the cut-off
SHOULDER_HALF_W = 0.247
SHOULDER_DEPTH = 0.42        # ellipse height; only its top arc is shown


def _px(f: float) -> int:
    return round(f * SIZE * SS)


def draw() -> Image.Image:
    n = SIZE * SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=_px(CORNER), fill=BG)

    # Four corner brackets, drawn as two overlapping bars per corner.
    inset, arm, thick = _px(BRACKET_INSET), _px(BRACKET_ARM), _px(BRACKET_THICK)
    r = _px(BRACKET_ROUND)
    far = n - inset
    for sx in (1, -1):
        for sy in (1, -1):
            x = inset if sx == 1 else far
            y = inset if sy == 1 else far
            def bar(dx: int, dy: int) -> None:
                x0, x1 = sorted((x, x + sx * dx))
                y0, y1 = sorted((y, y + sy * dy))
                d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=GREEN)

            bar(arm, thick)     # horizontal
            bar(thick, arm)     # vertical

    # The subject: head and shoulders, as a solid silhouette — the same
    # thing the people-mask output produces. Drawn on its own layer so the
    # shoulders can be cut off flat at the bottom.
    figure = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    fd = ImageDraw.Draw(figure)

    cx, cy, hr = n // 2, _px(HEAD_CY), _px(HEAD_R)
    fd.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=BLUE)

    half, top = _px(SHOULDER_HALF_W), _px(SHOULDER_TOP)
    fd.ellipse([cx - half, top, cx + half, top + _px(SHOULDER_DEPTH)], fill=BLUE)

    # Flat cut across the shoulders, so it reads as a bust rather than an egg.
    cut = Image.new("L", (n, n), 0)
    ImageDraw.Draw(cut).rectangle([0, 0, n, _px(SHOULDER_BOTTOM)], fill=255)
    figure.putalpha(Image.composite(figure.getchannel("A"),
                                    Image.new("L", (n, n), 0), cut))

    img.alpha_composite(figure)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def write_icns(png: Path, out: Path) -> bool:
    """macOS only, and only if iconutil is around."""
    if sys.platform != "darwin" or not shutil.which("iconutil"):
        return False
    src = Image.open(png)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "yewee.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = base * scale
                suffix = "" if scale == 1 else "@2x"
                src.resize((px, px), Image.LANCZOS).save(
                    iconset / f"icon_{base}x{base}{suffix}.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                       check=True)
    return True


def main() -> int:
    icon = draw()
    png = ASSETS / "yewee.png"
    icon.save(png)
    print(f"  wrote {png.relative_to(ROOT)}")

    ico = ASSETS / "yewee.ico"
    icon.save(ico, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print(f"  wrote {ico.relative_to(ROOT)}")

    icns = ASSETS / "yewee.icns"
    if write_icns(png, icns):
        print(f"  wrote {icns.relative_to(ROOT)}")
    else:
        print("  skipped .icns (needs macOS iconutil)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
