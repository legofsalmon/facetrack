# Distributing facetrack commercially

Working notes for turning facetrack into a sold product. Phase 0 is
done; the rest is planned but not built.

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

## Phase 1 — licensing (planned)

**Ed25519 signed keys, offline-first.** You hold the private key; the app
embeds only the public key. A key encodes product, edition, issue date,
optional expiry and optional machine binding, then is signed. The app
verifies the signature locally, so **both online and offline activation
work with no server at all**.

- *Free keys for reviewers* — issue a signed key, optionally time-limited.
  A small `issue-key` CLI for you; nothing else needed.
- *72-hour trial* — record first run in the app-support directory **and**
  the OS keychain/registry so casual deletion doesn't reset it.
- *Reality check* — facetrack is Python; a determined user can edit the
  check out. Freezing raises the bar a little, compiling just the licence
  module raises it more, nothing makes it airtight. Cap the effort: the
  goal is keeping honest people honest.

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
