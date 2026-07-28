#!/usr/bin/env python3
"""facetrack licence keys — vendor tool. Keep the private key secret.

    # once: make your signing keypair
    python tools/issue_key.py keygen

    # a normal one-off purchase (perpetual, any machine)
    python tools/issue_key.py issue --name "Jane Smith"

    # a reviewer / colleague key that lapses
    python tools/issue_key.py issue --name "Sam (review)" --edition review --days 90

    # tied to one machine (the customer reads the ID off the panel)
    python tools/issue_key.py issue --name "Big Venue" --machine a1b2c3d4e5f6a7b8

    # check a key the way the app will
    python tools/issue_key.py check FT1.xxx.yyy

The private key is read from FACETRACK_SECRET (hex) or --secret-file.
Never commit it; anyone holding it can mint licences.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facetrack import _ed25519 as ed          # noqa: E402
from facetrack.licensing import decode_key, encode_key  # noqa: E402


def _secret(args) -> bytes:
    raw = ""
    if args.secret_file:
        raw = Path(args.secret_file).read_text().strip()
    else:
        raw = os.environ.get("FACETRACK_SECRET", "").strip()
    if not raw:
        sys.exit("No signing key. Set FACETRACK_SECRET=<hex> or pass "
                 "--secret-file, or run `keygen` first.")
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        sys.exit("FACETRACK_SECRET must be 64 hex characters.")
    if len(key) != 32:
        sys.exit("Signing key must be 32 bytes (64 hex characters).")
    return key


def cmd_keygen(args) -> int:
    secret = secrets.token_bytes(32)
    public = ed.public_key(secret)
    print("\nPRIVATE key — store securely, never commit, never ship:\n")
    print(f"  {secret.hex()}\n")
    print("PUBLIC key — put this in the build (FACETRACK_PUBKEY):\n")
    print(f"  {public.hex()}\n")
    print("Suggested use:")
    print("  export FACETRACK_SECRET=<private>     # when issuing keys")
    print("  export FACETRACK_PUBKEY=<public>      # when building/running\n")
    return 0


def cmd_issue(args) -> int:
    secret = _secret(args)
    payload = {
        "v": 1,
        "p": "facetrack",
        "e": args.edition,
        "n": args.name,
        "i": date.today().isoformat(),
        "k": secrets.token_hex(6),
    }
    if args.days:
        payload["x"] = (date.today() + timedelta(days=args.days)).isoformat()
    if args.machine:
        payload["m"] = args.machine
    key = encode_key(payload, secret)
    if args.quiet:
        print(key)
        return 0
    print("\n" + json.dumps(payload, indent=2, sort_keys=True))
    print("\nLicence key — send this to the customer:\n")
    print(key + "\n")
    print("They paste it into the control panel (Licence card), or drop it")
    print("in a file called licence.key in the facetrack data folder.\n")
    return 0


def cmd_check(args) -> int:
    pub = args.public or os.environ.get("FACETRACK_PUBKEY", "")
    if not pub:
        sys.exit("Need the public key: --public <hex> or FACETRACK_PUBKEY.")
    payload = decode_key(args.key, public_key_hex=pub)
    if payload is None:
        print("INVALID — signature does not match this public key.")
        return 1
    print("VALID\n" + json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("keygen", help="create a signing keypair").set_defaults(fn=cmd_keygen)

    p = sub.add_parser("issue", help="issue a licence key")
    p.add_argument("--name", required=True, help="who it's for (shown in the panel)")
    p.add_argument("--edition", default="pro", help="pro | review | ... (default: pro)")
    p.add_argument("--days", type=int, default=0, help="expire after N days (0 = perpetual)")
    p.add_argument("--machine", default="", help="bind to a machine ID from the panel")
    p.add_argument("--secret-file", default="", help="file holding the private key hex")
    p.add_argument("--quiet", action="store_true", help="print only the key")
    p.set_defaults(fn=cmd_issue)

    p = sub.add_parser("check", help="verify a key")
    p.add_argument("key")
    p.add_argument("--public", default="", help="public key hex (or FACETRACK_PUBKEY)")
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
