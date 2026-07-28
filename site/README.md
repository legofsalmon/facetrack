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

## Deploying to a subdomain

Any static host works. Cloudflare Pages and Netlify both build from a
*private* repo, give free TLS, and take a custom subdomain — GitHub Pages
does not serve from private repos on the free plan.

**Cloudflare Pages**

1. Pages → Create → connect this repo
2. Build command: *(none)* · Output directory: `site`
3. Custom domains → add e.g. `facetrack.yourdomain.com`
4. Cloudflare adds the CNAME automatically if the domain is on Cloudflare;
   otherwise create `facetrack` → `<project>.pages.dev` at your DNS host

**Netlify** — same shape: publish directory `site`, then Domain settings →
add subdomain → create the CNAME it shows you.

Both redeploy on every push to `main`.

## Local preview

```bash
python -m http.server -d site 8000    # then open http://localhost:8000
```
