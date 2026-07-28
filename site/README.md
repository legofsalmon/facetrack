# facetrack landing page

A single static page — no build step, no backend, no dependencies. That's
deliberate: with checkout and the customer portal handled by a Merchant of
Record, the site never touches card data, passwords or personal records, so
there is nothing to run, patch or breach.

```
site/
  index.html     the page
  img/           screenshots and icon
  README.md      this file
```

## Before it goes live: real footage

`img/overlay.jpg` and `img/testcard.png` are generated from facetrack's own
rendering code, so they are yours to publish. What the page still needs is a
**real crowd shot from your own camera** — a landing page for a face-tracking
product has to show it tracking real faces.

Do not reuse stock or press photos for this. Shoot it at one of your own
events (or with colleagues who have agreed), export a frame with the overlay
on, and drop it in as the hero image.

## Placeholders to fill in

Search `index.html` for these and replace:

| Placeholder | Becomes |
|---|---|
| `PRICE` | e.g. `£149` |
| `BUY_LINK` | your checkout URL (Lemon Squeezy / Paddle) |
| `ACCOUNT_LINK` | the provider's hosted customer portal URL |
| `DOWNLOAD_MAC` / `DOWNLOAD_WIN` | installer URLs (see below) |
| `VERSION` | e.g. `v1.3` |
| `EMAIL` | your contact address |
| `COPYRIGHT_YEAR` | e.g. `2026` |

Still to write: `privacy.html` and `terms.html` (the footer links to them).
The privacy one is short and honest — no video stored, no identification,
activation is local — but get the terms checked before taking money.

## Where the installers live

The app repo is private, so GitHub Release assets there need a login. Two
options:

1. **A public releases-only repo** — push the built installers there and
   link to those release assets. Free, versioned, no bandwidth bill.
2. **Object storage** (Cloudflare R2, S3) behind the same subdomain.

Option 1 is simpler and is what the download links should point at first.

## Deploying on Vercel

The site is plain static files — no build step, no framework. `vercel.json`
here sets clean URLs (so `/privacy` works), long cache headers on images,
and a few sensible security headers.

### Link the project (once)

1. Vercel → **Add New… → Project** → import `legofsalmon/facetrack`.
2. **Root Directory: `site`** — this is the important one. Without it Vercel
   looks at the repo root and finds a Python app.
3. Framework Preset: **Other**. Leave build and output commands empty.
4. Deploy. You get a `…vercel.app` URL straight away.

### Don't rebuild on every app commit

Most commits to this repo don't touch the site. Under
Settings → Git → **Ignored Build Step**, use:

```bash
git diff --quiet HEAD^ HEAD -- .
```

Vercel skips the build when that exits 0 (no changes inside `site/`) and
builds when it exits 1.

### The subdomain

Settings → **Domains** → add `facetrack.yourdomain.com`, then create the
CNAME Vercel shows you at your DNS host (or, if the domain's nameservers are
already on Vercel, it wires itself up). TLS is automatic.

**Hold this step until the page is finished** — see the checklist below.
Until then the `…vercel.app` URL is fine for review and for sending to
colleagues.

### A note on repo access

Linking this repo gives Vercel read access to the whole private
product source, and it clones ~200 MB (the models) on each build. Nothing
secret lives in the repo — your signing key stays on your machine — so this
is a normal trade to accept. If you'd rather Vercel never saw the product
source, put `site/` in its own small repo and point Vercel at that instead.

## Before you attach the subdomain

- [ ] Real crowd shot from your own camera (see above)
- [ ] `PRICE`, `BUY_LINK`, `DOWNLOAD_MAC`, `DOWNLOAD_WIN`, `VERSION`,
      `EMAIL`, `COPYRIGHT_YEAR` filled in
- [ ] `privacy.html` and `terms.html` written (footer links to them)
- [ ] Installers actually exist to download (Phase 2)

A live page with a broken Buy button and no price does more harm than no
page at all.

## Selling through Lemon Squeezy

1. Create the product in Lemon Squeezy, then copy its **checkout URL** into
   `BUY_LINK`.
2. Customers reach receipts and re-downloads through Lemon Squeezy's own
   customer portal — the footer already links to it. Confirm the exact URL
   in your dashboard, as stores can have their own.
3. **Key delivery.** Lemon Squeezy has its own licence-key feature, but
   facetrack uses its own signed keys so activation works offline — so
   issue keys yourself:
   - **At launch, do it manually.** A sale emails you; open the Licence
     Admin, issue a key against the order number, and reply with it. A
     minute per sale, and your signing key never leaves your machine.
   - **Automate later** with a serverless function on their `order_created`
     webhook. Worth knowing the trade: that function needs your private
     signing key, so a breach there means anyone can mint licences. Only
     worth it once volume makes manual issuing annoying.

## Local preview

```bash
python -m http.server -d site 8000    # then open http://localhost:8000
```
