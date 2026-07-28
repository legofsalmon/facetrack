# Distributing facetrack commercially

Working notes for turning facetrack into a sold product. Phases 0 and 1
are done; 2-4 are planned but not built.

*Not legal advice — the licence positions below quote the upstream terms
directly so a solicitor can check them quickly.*

---

## Phase 0 — make it shippable ✅ done

### Models

Every model in a packaged build is now permissively licensed (MIT or
Apache-2.0) — see `models/README.md` for the table. Two changes made
that true:

- **SCRFD removed.** InsightFace states: *"The training data containing
  the annotation (and the models trained with these data) are available
  for non-commercial research purposes only."* Their code is MIT but the
  weights are not, so it could never ship in a paid product.
- **CenterFace added** (MIT) as the GPU detector in its place. The
  upstream ONNX has a fixed 10x3x32x32 input, so the shipped copy is
  patched to dynamic dimensions — same weights, recorded in
  `models/README.md` for attribution.

Measured on CPU, three images (small sample — validate on the show
machine with real footage):

| Image | YuNet 640 | YuNet 1280 | CenterFace 640 | CenterFace 1280 | actual |
|---|---|---|---|---|---|
| synthetic, 8 small faces | 8 | 8 | 7 | 8 | 8 |
| street photo, hard | 2 | 2 | 2 | **3** | ~5 |
| two faces, close | 2 | 3 (false +) | 2 | 2 | 2 |
| cost | 5.8 ms | 14.3 ms | 18.6 ms | 73 ms | |

CenterFace is not dramatically more accurate than YuNet on CPU and is
much slower there — **YuNet stays the default**. CenterFace earns its
place on the GPU: it is fully convolutional, so *Search detail* 960/1280
is cheap on CUDA, and that is where it resolved a face YuNet missed.

### Build variants

`facetrack/edition.py` carries a single `DISTRIBUTION` flag.

| | Internal build (repo) | Distribution build |
|---|---|---|
| `DISTRIBUTION` | `False` | `True` (set by packaging) |
| Silhouette models | PP-HumanSeg, MODNet, **RVM** | PP-HumanSeg, MODNet |
| RVM model file | present | excluded from the bundle |

**RVM is GPL-3.0.** Shipping it would put the entire product under
GPL-3.0 with a source-code requirement for every buyer, who could then
redistribute freely — incompatible with paid licensing. It stays fully
available for in-house shows.

The panel builds its *Silhouette model* list from what the running build
can actually load, so the distributed app simply never offers RVM.
`create_people_model()` refuses it outright as a second line of defence,
and a smoke test asserts both.

To sanity-check a distribution build without repacking:

```bash
FACETRACK_DISTRIBUTION=1 .venv/bin/python main.py
```

### Still to confirm before taking money

- **NDI SDK terms** — the NDI runtime ships inside the `cyndilib` wheel.
  NewTek/Vizrt's SDK licence governs redistribution and has attribution
  and branding requirements. Read it and comply.
- **MODNet ONNX provenance** — the weights come from a HuggingFace
  conversion (`Xenova/modnet`). MODNet's own repo is unambiguous that
  code *and models* are Apache-2.0; confirm the conversion carries the
  same terms, or re-export from the original weights.
- **EULA + privacy policy** — needed before selling. facetrack processes
  video locally and stores nothing, which makes the privacy policy short,
  but activation (if it phones home) does transmit a licence key and a
  machine fingerprint.

---

## Phase 1 — licensing ✅ done

**Ed25519 signed keys, verified offline.** You hold the private key; the
app embeds only the public key, so activation never needs a server — the
same key works online or air-gapped, which covers both activation paths
and makes reviewer keys trivial.

Crypto is `facetrack/_ed25519.py`, a dependency-free implementation of
the RFC 8032 reference (verification takes ~4 ms and runs once at
startup, so a compiled crypto library would only add installer weight).
The official RFC test vectors are asserted in the smoke tests.

### Issuing keys — the admin app

Double-click **`Licence Admin.command`** (Mac) or **`Licence Admin.bat`**
(Windows), or run `python tools/admin.py`. It opens a local page at
`127.0.0.1:8091` that:

- generates your signing keypair on first run (with a back-it-up warning)
  and shows the **public key** to build with;
- issues keys from a form — licensee, your own reference (order number
  or email), purchase vs reviewer, optional expiry, optional machine
  lock — and hands back a copyable key;
- keeps a **ledger** of everything issued, so you can look up who has
  what and re-copy a key a customer has lost;
- checks any key against your signing key.

**Never distribute it.** It holds the private key that mints licences.
It binds to localhost only, lives outside the `facetrack` package so
packaging can't sweep it in, and the signing key is written `chmod 600`
into a vendor folder separate from the app's own settings.

The same operations are available from the terminal if you prefer:

```bash
python tools/issue_key.py keygen                 # once — store the private key safely
export FACETRACK_SECRET=<private hex>

python tools/issue_key.py issue --name "Jane Smith"                      # perpetual
python tools/issue_key.py issue --name "Sam" --edition review --days 90  # reviewer
python tools/issue_key.py issue --name "Venue" --machine <id from panel> # node-locked
python tools/issue_key.py check FT1.…                                    # verify one
```

A key is ~206 characters: `FT1.<payload>.<signature>`, payload being
compact JSON (product, edition, name, issued, optional expiry, optional
machine binding, key id). Expiry and machine binding are optional, so a
normal purchase gets a perpetual key that works on any machine.

### In the app

`facetrack/licensing.py` exposes `status()`, `activate()` and
`deactivate()`. States:

| State | When | Behaviour |
|---|---|---|
| `unrestricted` | no public key compiled in (repo / internal builds) | no gating at all |
| `licensed` | a valid, unexpired, machine-matching key is stored | normal |
| `trial` | no key yet, within 72 hours of first run | normal, panel counts down |
| `expired` | no key, trial used up | feeds stay up but carry a TRIAL ENDED slate; panel stays usable so a key can be entered |

The panel gains a **Licence** card (hidden entirely in unrestricted
builds) showing the state, this machine's ID, and a paste-a-key field.
Activation takes effect within ~20 seconds without a restart.

The trial clock is anchored in the user data directory *and* a second
platform location, and takes the earliest first-run either knows about,
so deleting one file doesn't restart it. A rolled-back clock doesn't
hand back time either. It is still local state, so a determined user can
clear it — see the note above about not over-investing here.

### Building a licensed app

The packaging step sets the public key and the distribution flag:

```bash
export FACETRACK_PUBKEY=<public hex>
export FACETRACK_DISTRIBUTION=1
```

Both are read at import time, so a build can bake them into
`facetrack/licensing.py` and `facetrack/edition.py` instead of relying
on the environment.

### Not built yet

- **Online activation proper.** Today "online" and "offline" are the
  same operation: paste the key you were emailed. Server-backed
  activation, seat counting and revocation are Phase 4; the key format
  already carries a key id (`k`) for it.
- **Purchase → key delivery** is Phase 3 (the payment provider's webhook
  calls `issue_key.py`; the admin's ledger is the record until then).
- **Revocation** shows in the ledger but cannot be enforced without the
  Phase 4 server — a key already issued keeps working offline.

## Phase 2 — installers (planned)

GitHub Actions on `macos-latest` and `windows-latest`, certificates in
repository secrets, artefacts attached to a Release.

- **macOS**: PyInstaller → `.app` → Developer ID signing → notarisation →
  `.dmg`. Apple Developer Program, $99/yr. Without notarisation Gatekeeper
  blocks it.
- **Windows**: PyInstaller → Inno Setup → code signing (~$200–500/yr).
  Without a signature SmartScreen warns users off.
- Expect roughly 250–400 MB per installer once RVM is excluded.

## Phase 3 — selling (planned)

Use a **Merchant of Record** (Lemon Squeezy or Paddle) rather than raw
Stripe: they handle global VAT/sales-tax registration and remittance,
which is the part that quietly sinks solo-developer products. Their
purchase webhook calls your key issuer.

Source stays private; binaries need a **public** download point, since
private-repo release assets require a GitHub login. A public
releases-only repository or a small website both work.

## Phase 4 — online activation (optional)

Only needed for revocation and seat limits, which require a server and a
database. The Phase 1 key format doesn't change — the online step just
fetches or validates a key that the app can still verify offline.
