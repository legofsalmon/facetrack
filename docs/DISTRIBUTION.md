# Distributing yewee commercially

Working notes for turning yewee into a sold product. Phases 0–2 are
done: v1.4 is released with public downloads on the site and licensing
active. Phase 3 (checkout) is next; keys go by email meanwhile.

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

`yewee/edition.py` carries a single `DISTRIBUTION` flag.

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
YEWEE_DISTRIBUTION=1 .venv/bin/python main.py
```

### Still to confirm before taking money

- **NDI SDK terms** — the NDI runtime ships inside the `cyndilib` wheel.
  NewTek/Vizrt's SDK licence governs redistribution and has attribution
  and branding requirements. Read it and comply.
- **MODNet ONNX provenance** — the weights come from a HuggingFace
  conversion (`Xenova/modnet`). MODNet's own repo is unambiguous that
  code *and models* are Apache-2.0; confirm the conversion carries the
  same terms, or re-export from the original weights.
- **EULA + privacy policy** — needed before selling. yewee processes
  video locally and stores nothing, which makes the privacy policy short,
  but activation (if it phones home) does transmit a licence key and a
  machine fingerprint.

---

## Phase 1 — licensing ✅ done

**Ed25519 signed keys, verified offline.** You hold the private key; the
app embeds only the public key, so activation never needs a server — the
same key works online or air-gapped, which covers both activation paths
and makes reviewer keys trivial.

Crypto is `yewee/_ed25519.py`, a dependency-free implementation of
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
It binds to localhost only, lives outside the `yewee` package so
packaging can't sweep it in, and the signing key is written `chmod 600`
into a vendor folder separate from the app's own settings.

The same operations are available from the terminal if you prefer:

```bash
python tools/issue_key.py keygen                 # once — store the private key safely
export YEWEE_SECRET=<private hex>

python tools/issue_key.py issue --name "Jane Smith"                      # perpetual
python tools/issue_key.py issue --name "Sam" --edition review --days 90  # reviewer
python tools/issue_key.py issue --name "Venue" --machine <id from panel> # node-locked
python tools/issue_key.py check YW1.…                                    # verify one
```

A key is ~206 characters: `YW1.<payload>.<signature>`, payload being
compact JSON (product, edition, name, issued, optional expiry, optional
machine binding, key id). Expiry and machine binding are optional, so a
normal purchase gets a perpetual key that works on any machine.

### In the app

`yewee/licensing.py` exposes `status()`, `activate()` and
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
export YEWEE_PUBKEY=<public hex>
export YEWEE_DISTRIBUTION=1
```

Both are read at import time, so a build can bake them into
`yewee/licensing.py` and `yewee/edition.py` instead of relying
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

## Phase 2 — installers ✅ done (two credentials outstanding)

**Shipping since v1.4:** `build/build.py` + `build/yewee.spec` produce a
standalone bundle; CI builds both platforms on every tag (always
`--distribution` — an internal fallback once shipped RVM by accident,
so it no longer exists). The vendor public key lives in `build.py` and
is baked into every build; the private key stays in the Licence Admin's
data directory — back it up, losing it strands every install.

- **macOS**: signed (Developer ID, hardened runtime, camera entitlement)
  by `build/sign_macos.sh`, which also repairs the unsealed
  `Syphon.framework` PyInstaller leaves and builds the `.dmg`.
  **Outstanding: notarisation** — needs `notarytool store-credentials`
  run once by the account holder, then `sign_macos.sh --notarize`.
  Until then, other Macs need right-click → Open on first launch.
- **Windows**: Inno Setup installer built in CI (per-user, no admin,
  uninstall keeps licence + settings). **Outstanding: a code-signing
  certificate** (~£200–400/yr) — until then SmartScreen warns.

Signing exposed and fixed two packaged-only bugs worth remembering: the
app wrote logs inside its own bundle (breaks the signature; see
`yewee/paths.py`), and the camera-permission prompt never appeared
(needs the main run loop, a registered PyObjC block signature, and a
non-`LSBackgroundOnly` app — `build.py` fails the build if that
regresses).

Commands and details in `build/README.md`.

## Phase 3 — selling (planned)

**Decided: Lemon Squeezy**, as Merchant of Record — it handles global
VAT/sales-tax registration and remittance, which is the part that quietly
sinks solo-developer products, and gives customers a hosted portal for
receipts and re-downloads.

**Key delivery starts manual.** yewee uses its own signed keys (so
activation works offline), not Lemon Squeezy's licence feature. On a
sale, issue a key in the Licence Admin against the order number and reply
with it — a minute of work, and the signing key never leaves your
machine. Automating it via their `order_created` webhook means putting
that private key in a cloud function, where a breach lets anyone mint
licences; worth it only once the volume justifies the risk.

**The landing page** lives in its own repository,
[`legofsalmon/facetrack-site`](https://github.com/legofsalmon/facetrack-site),
deployed on Vercel. It is kept separate deliberately: Vercel then never
needs read access to this product source, and site deploys don't drag
~200 MB of models through a build. Checkout and account links point at
Lemon Squeezy.

**Downloads are live** (since v1.4): the site's Download section links
straight to this repo's GitHub release assets, which are public because
the repo itself is public — a deliberate decision (July 2026), revisit
if source-runs bypassing the licence starts to matter commercially. If
the repo ever goes private, move the installers to a public
releases-only repo (the `crewbox-dist` pattern) and repoint the two
links on the site.

**Until checkout opens**, licence keys go out by email — the site says
so under the price line.

## Phase 4 — online activation (optional)

Only needed for revocation and seat limits, which require a server and a
database. The Phase 1 key format doesn't change — the online step just
fetches or validates a key that the app can still verify offline.
