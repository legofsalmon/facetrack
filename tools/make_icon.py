"""Draws the yewee app icon and writes every format the builds need.

    python tools/make_icon.py

Produces assets/yewee.png (1024), yewee.icns (macOS) and yewee.ico
(Windows). Kept as code rather than a binary blob so the shape can be
adjusted and every size stays consistent — an icon that is hand-edited at
one size and scaled to the rest goes muddy at 16px.

The icon is the mask output itself: a white person on black, filling the
tile and leaving by the bottom edge. That is a frame of what yewee produces
when the people mask is exported, which beats any symbol standing in for
it, and pure black and white stays legible down to 16px where a detailed
mark turns to mush.

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

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)

# Fractions of the canvas, so the geometry survives any SIZE change.
CORNER = 0.205               # rounded-square radius

HEAD_CY, HEAD_R = 0.310, 0.163
SHOULDER_TOP = 0.575
SHOULDER_HALF_W = 0.420      # stops short of the sides, so the tile keeps its
SHOULDER_DEPTH = 0.860       # corners; the shoulders leave by the bottom edge


def _px(f: float) -> int:
    return round(f * SIZE * SS)


def draw() -> Image.Image:
    n = SIZE * SS
    img = Image.new("RGBA", (n, n), BLACK)
    d = ImageDraw.Draw(img)

    cx, cy, hr = n // 2, _px(HEAD_CY), _px(HEAD_R)
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=WHITE)

    half, top = _px(SHOULDER_HALF_W), _px(SHOULDER_TOP)
    d.ellipse([cx - half, top, cx + half, top + _px(SHOULDER_DEPTH)], fill=WHITE)

    # The tile does all the cropping: shoulders run out through the sides and
    # the bottom rather than stopping inside the frame.
    tile = Image.new("L", (n, n), 0)
    ImageDraw.Draw(tile).rounded_rectangle([0, 0, n - 1, n - 1],
                                           radius=_px(CORNER), fill=255)
    img.putalpha(tile)
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
