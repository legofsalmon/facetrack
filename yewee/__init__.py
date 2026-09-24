"""yewee — real-time face detection, tracking and NDI output.

Detection only: finds and follows faces (boxes, stable IDs, optional
expression estimate). No identity recognition is performed.
"""

# The product's own version, and the one place it is declared. Versions
# were reset to 1.0.0 on 2026-09-24: the v1.0-v1.4 tags on GitHub predate
# the reset, so a higher number there is older, not newer. build/build.py
# stamps this into a packaged build and refuses a distribution build whose
# --version (the git tag in CI) disagrees with it.
__version__ = "1.0.0"


def app_version() -> str:
    """The version this copy reports: what the packaging step stamped, or
    __version__ when running from source."""
    try:
        from ._buildinfo import VERSION        # type: ignore
    except ImportError:
        return __version__
    return str(VERSION or __version__)
