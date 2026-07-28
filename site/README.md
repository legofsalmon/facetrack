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

## Deploying to your own host

The site is plain static files — no PHP, no Node, no database. Upload the
**contents of `site/`** to the subdomain's document root:

```bash
# example, adjust to your host
rsync -av --delete site/ user@yourhost:/var/www/facetrack.yourdomain.com/
```

Or drag the files in over SFTP / cPanel File Manager. Then at your DNS
host, point the subdomain at the same server (an A record to its IP, or a
CNAME if your host gives you a hostname), and enable TLS for it — most
control panels offer one-click Let's Encrypt.

Check afterwards that `img/` came across and the page loads over **https**.

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
