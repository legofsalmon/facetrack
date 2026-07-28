"""Which build of facetrack this is.

Repo and self-built copies are INTERNAL: everything is available,
including components that may not be redistributed in a paid product.
Right now that means the RVM matting model — GPL-3.0, which is fine to
run in-house but incompatible with shipping a closed-source product,
since GPL would require handing every buyer the source with the right to
redistribute it freely.

The installer build sets DISTRIBUTION = True (see build/README.md), which
hides those components and leaves only permissively licensed ones. Keep
this module free of imports so the packaging step can rewrite it safely.
"""
from __future__ import annotations

import os

# Overwritten to True by the packaging step; the env var is for testing
# the distribution build without repacking.
DISTRIBUTION = os.environ.get("FACETRACK_DISTRIBUTION", "") == "1"
