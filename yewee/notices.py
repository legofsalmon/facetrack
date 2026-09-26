"""THIRD-PARTY-NOTICES.txt: what yewee is built from, under whose terms.

build/build.py writes the file from the environment it packages (so it
lists exactly the versions that ship) and the spec bundles it next to the
executable. The panel serves it at /notices; a source run generates it on
the fly from the environment it is running in.

The licences of most dependencies require their copyright notice and
licence text to travel with any binary copy; LGPL-2.1 (FFmpeg inside the
OpenCV wheel) and MPL-2.0 (certifi) also require saying where the source
is. The NDI SDK licence asks for NDI's copyright notice and a clear
trademark notation (sections 3f and 3g).
"""
from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

#: Present in the build environment, never in what ships.
BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "macholib",
              "pefile", "pywin32-ctypes", "pip", "setuptools", "wheel", "packaging"}

HEADER = """\
Yewee — third-party notices
===========================

Yewee is made by Colm Hewson / LeTissier Creative Studios. It is built on
the open-source software and models listed below, each under its own
licence, reproduced in full further down. Where a licence asks for the
source of a component, it is available from the project linked beside it
at the version listed; for anything else, ask info@letissier.ie.

NDI® is a registered trademark of Vizrt NDI AB. Yewee sends and receives
NDI using the NDI runtime, Copyright (C) 2023-2024 Vizrt NDI AB, all rights
reserved, bundled by cyndilib and used under the NDI SDK licence
(https://ndi.link/ndisdk_license). Yewee is not a product of, and is not
endorsed by, Vizrt NDI AB. More about NDI: https://ndi.video/

Models (weights shipped in the models folder)
---------------------------------------------
  YuNet face detector (face_detection_yunet_2023mar.onnx)
      MIT — https://github.com/opencv/opencv_zoo
  CenterFace (centerface_dynamic.onnx; input dimensions made dynamic)
      MIT — https://github.com/Star-Clouds/CenterFace
  FER+ expression model (emotion-ferplus-8.onnx)
      MIT — https://github.com/onnx/models
  PP-HumanSeg (human_segmentation_pphumanseg_2023mar.onnx)
      Apache-2.0 — https://github.com/opencv/opencv_zoo
  MODNet portrait matting (modnet_portrait.onnx)
      Apache-2.0 — https://github.com/ZHKKKe/MODNet
  (RobustVideoMatting, GPL-3.0, is in internal builds only and never ships.)

Components that ask for their source to be offered
--------------------------------------------------
  FFmpeg (LGPL-2.1), inside opencv-python: the wheel's own licence file
      below names the build; source at https://github.com/opencv/opencv-python
      and https://ffmpeg.org. It is a separate library that can be replaced.
  certifi (MPL-2.0): unmodified; source at https://github.com/certifi/python-certifi
"""

_LICENCE_NAMES = ("license", "licence", "copying", "notice", "authors")


def _licence_label(dist) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or [])
                   if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    text = (meta.get("License") or "").strip().splitlines()
    return text[0][:80] if text else "see licence text"


def _licence_files(dist) -> list[tuple[str, str]]:
    out = []
    for f in dist.files or []:
        name = f.name.lower()
        parts = [p.lower() for p in f.parts]
        in_meta = any(p.endswith((".dist-info", ".egg-info")) for p in parts)
        wanted = any(name.startswith(n) or n in name for n in _LICENCE_NAMES)
        # the dist-info licence files, plus notices shipped inside the package
        # itself (cyndilib keeps the NDI runtime's there)
        if wanted and (in_meta or name.endswith((".txt", ".md", ".rst", ""))) \
                and not name.endswith((".py", ".pyc", ".so", ".dll", ".dylib", ".pyd")):
            try:
                text = Path(f.locate()).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            out.append(("/".join(f.parts[-2:]), text))
    seen, unique = set(), []
    for label, text in out:                    # the same file under two names
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append((label, text))
    return unique


def _python_licence() -> str:
    for base in (Path(sys.base_prefix), Path(sys.base_prefix) / "lib" /
                 f"python{sys.version_info.major}.{sys.version_info.minor}"):
        for name in ("LICENSE.txt", "LICENSE"):
            path = base / name
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
    return ("Python Software Foundation License Version 2 — "
            "https://docs.python.org/3/license.html\n")


def generate() -> str:
    dists = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if name and name.lower() not in BUILD_ONLY:
            dists.setdefault(name.lower(), dist)
    lines = [HEADER, "", "Software", "--------"]
    ordered = sorted(dists.values(), key=lambda d: d.metadata["Name"].lower())
    for dist in ordered:
        home = dist.metadata.get("Home-page") or ""
        if not home:
            for url in dist.metadata.get_all("Project-URL") or []:
                home = url.split(",", 1)[-1].strip()
                break
        lines.append(f"  {dist.metadata['Name']} {dist.version} — {_licence_label(dist)}"
                     + (f" — {home}" if home else ""))
    lines.append(f"  Python {sys.version.split()[0]} — PSF-2.0 — https://www.python.org")
    lines += ["", "", "Licence texts", "============="]
    for dist in ordered:
        files = _licence_files(dist)
        lines += ["", "-" * 72, f"{dist.metadata['Name']} {dist.version}", "-" * 72]
        if not files:
            lines.append(f"Licence: {_licence_label(dist)} (no licence file in the "
                         "distribution; see the project's home page)")
        for label, text in files:
            lines += [f"[{label}]", text.rstrip(), ""]
    lines += ["", "-" * 72, f"Python {sys.version.split()[0]}", "-" * 72,
              _python_licence().rstrip(), ""]
    return "\n".join(lines) + "\n"


def bundled_path() -> Path | None:
    """The notices file a packaged build carries, if this is one."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        path = Path(base) / "THIRD-PARTY-NOTICES.txt"
        if path.is_file():
            return path
    return None


def text() -> str:
    path = bundled_path()
    if path is not None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return generate()


if __name__ == "__main__":
    # Bytes, as UTF-8, whatever the console or pipe encoding: on Windows a
    # pipe defaults to cp1252, which garbles the dashes for a UTF-8 reader
    # and cannot encode every character in the licence texts at all.
    sys.stdout.flush()
    sys.stdout.buffer.write(generate().encode("utf-8"))
    sys.stdout.buffer.flush()
