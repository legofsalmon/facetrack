"""Ed25519 signing and verification, following the RFC 8032 reference.

Deliberately dependency-free: the app only ever *verifies* a licence
signature, once, at startup — so speed is irrelevant, and avoiding a
compiled crypto dependency keeps the installer small and the PyInstaller
build simple. `tools/issue_key.py` uses the signing half.

Correctness is pinned by the official RFC 8032 test vectors in the smoke
tests. Points are held in extended homogeneous coordinates (X:Y:Z:T) so
scalar multiplication needs no modular inversion per step.
"""
from __future__ import annotations

import hashlib

P = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, P - 2, P) % P
_SQRT_M1 = pow(2, (P - 1) // 4, P)

Point = tuple  # (X, Y, Z, T)


def _add(p1: Point, p2: Point) -> Point:
    a = (p1[1] - p1[0]) * (p2[1] - p2[0]) % P
    b = (p1[1] + p1[0]) * (p2[1] + p2[0]) % P
    c = 2 * p1[3] * p2[3] * _D % P
    dd = 2 * p1[2] * p2[2] % P
    e, f, g, h = b - a, dd - c, dd + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def _mul(s: int, p1: Point) -> Point:
    q: Point = (0, 1, 1, 0)          # neutral element
    while s > 0:
        if s & 1:
            q = _add(q, p1)
        p1 = _add(p1, p1)
        s >>= 1
    return q


def _equal(p1: Point, p2: Point) -> bool:
    if (p1[0] * p2[2] - p2[0] * p1[2]) % P != 0:
        return False
    return (p1[1] * p2[2] - p2[1] * p1[2]) % P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, P - 2, P) % P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * _SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if (x & 1) != sign:
        x = P - x
    return x


_G_Y = 4 * pow(5, P - 2, P) % P
_G_X = _recover_x(_G_Y, 0)
G: Point = (_G_X, _G_Y, 1, _G_X * _G_Y % P)


def _compress(p1: Point) -> bytes:
    zinv = pow(p1[2], P - 2, P)
    x = p1[0] * zinv % P
    y = p1[1] * zinv % P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes) -> Point | None:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % P)


def _sha512_modq(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % L


def _expand(secret: bytes) -> tuple[int, bytes]:
    h = hashlib.sha512(secret).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(secret: bytes) -> bytes:
    """32-byte public key for a 32-byte secret seed."""
    a, _ = _expand(secret)
    return _compress(_mul(a, G))


def sign(secret: bytes, msg: bytes) -> bytes:
    """64-byte signature. Vendor side only — the app never signs."""
    a, prefix = _expand(secret)
    pub = _compress(_mul(a, G))
    r = _sha512_modq(prefix + msg)
    big_r = _mul(r, G)
    rs = _compress(big_r)
    k = _sha512_modq(rs + pub + msg)
    return rs + int.to_bytes((r + k * a) % L, 32, "little")


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    """True if `sig` is a valid Ed25519 signature over `msg`."""
    if len(sig) != 64 or len(pub) != 32:
        return False
    a = _decompress(pub)
    if a is None:
        return False
    big_r = _decompress(sig[:32])
    if big_r is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= L:
        return False
    k = _sha512_modq(sig[:32] + pub + msg)
    return _equal(_mul(s, G), _add(big_r, _mul(k, a)))
