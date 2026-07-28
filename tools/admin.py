#!/usr/bin/env python3
"""yewee licence admin — a small local web app for issuing keys.

    python tools/admin.py            # opens http://127.0.0.1:8091

NEVER SHIP THIS. It holds your private signing key; anyone with it can
mint licences. It binds to 127.0.0.1 only and is excluded from packaged
builds (see docs/DISTRIBUTION.md).

State lives in a vendor folder separate from the app's own settings:

    <user data>/yewee-vendor/signing.key   private key, chmod 600
    <user data>/yewee-vendor/issued.json   ledger of everything issued
"""
# NOTE: no `from __future__ import annotations` — FastAPI resolves the
# `Request` type hint for dependency injection, and a stringified
# annotation makes it treat the parameter as a query field instead.

import json
import os
import secrets
import sys
import threading
import webbrowser
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yewee import _ed25519 as ed                      # noqa: E402
from yewee.licensing import decode_key, encode_key    # noqa: E402


def vendor_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / "yewee-vendor"
    d.mkdir(parents=True, exist_ok=True)
    return d


KEY_PATH = vendor_dir() / "signing.key"
LEDGER_PATH = vendor_dir() / "issued.json"


def load_secret() -> bytes | None:
    try:
        return bytes.fromhex(KEY_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def save_secret(secret: bytes) -> None:
    KEY_PATH.write_text(secret.hex())
    try:
        KEY_PATH.chmod(0o600)
    except OSError:
        pass


def ledger() -> list:
    try:
        return json.loads(LEDGER_PATH.read_text())
    except (OSError, ValueError):
        return []


def record(entry: dict) -> None:
    rows = ledger()
    rows.insert(0, entry)
    LEDGER_PATH.write_text(json.dumps(rows, indent=2))


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<title>yewee licence admin</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1014;--panel:#161a21;--panel2:#1d222c;--line:#262c38;--text:#e8ebf0;
--dim:#8b94a3;--accent:#4da3ff;--good:#3ddc84;--bad:#ff5d5d;--warn:#ffc24d}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,sans-serif;
padding:18px;max-width:920px;margin:0 auto}
h1{font-size:18px;font-weight:650;margin-bottom:4px}h1 span{color:var(--accent)}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:14px;overflow:hidden}
.card h2{font-size:11.5px;font-weight:650;letter-spacing:1.1px;text-transform:uppercase;
color:var(--dim);padding:11px 14px 9px;border-bottom:1px solid var(--line)}
.row{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.lbl{flex:0 0 150px;display:flex;flex-direction:column}
.lbl label{font-size:13px}.hint{color:var(--dim);font-size:11px;line-height:1.3}
input,select{background:var(--panel2);color:var(--text);border:1px solid var(--line);
border-radius:7px;padding:7px 9px;font-size:13px;outline:none;flex:1;font-family:inherit}
input:focus,select:focus{border-color:var(--accent)}
button{background:var(--accent);color:#08111e;border:0;border-radius:7px;padding:8px 15px;
font-size:13px;font-weight:650;cursor:pointer}button:hover{filter:brightness(1.12)}
button.ghost{background:var(--panel2);color:var(--dim);border:1px solid var(--line);font-weight:500}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
.keyout{background:#0b1220;border:1px solid var(--accent);border-radius:8px;padding:11px;margin:0 14px 12px}
.warn{background:#2a1d0a;border:1px solid var(--warn);color:#ffd98a;border-radius:8px;
padding:11px 13px;margin:0 14px 12px;font-size:12.5px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:8px 14px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
td.k{color:var(--dim)}
.pill{font-size:10.5px;padding:2px 7px;border-radius:5px;background:var(--panel2);color:var(--dim)}
.pill.rev{background:#3a1414;color:var(--bad)}
.msg{padding:0 14px 12px;font-size:12.5px}
</style></head><body>
<h1>face<span>track</span> licence admin</h1>
<div class=sub>Local only — this machine, never exposed. Your private signing key lives here.</div>
<div id=app>loading…</div>

<script>
const $=(s)=>document.querySelector(s);
let state={};
async function api(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},
    body:body?JSON.stringify(body):undefined});
  return r.json();
}
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function load(){ state=await api('/api/state'); render(); }

function render(){
  if(!state.has_key){ $('#app').innerHTML=`
    <div class=card><h2>Set up signing</h2>
      <div class=warn><b>Do this once.</b> It creates the private key that signs every licence.
      Back it up somewhere safe — if you lose it you cannot issue keys that existing
      customers' copies will accept. If it leaks, anyone can mint licences.</div>
      <div class=row><button onclick=genkey()>Generate signing key</button></div>
    </div>`; return; }

  $('#app').innerHTML=`
  <div class=card><h2>Your public key</h2>
    <div class=hint style="padding:11px 14px 0">Build the app with this so it accepts your keys:
      <span class=mono>YEWEE_PUBKEY=…</span></div>
    <div class="keyout mono" style="margin-top:10px">${esc(state.public_key)}</div>
    <div class=row><button class=ghost onclick="copy('${esc(state.public_key)}')">Copy public key</button>
      <span class=hint>private key: <span class=mono>${esc(state.key_path)}</span></span></div>
  </div>

  <div class=card><h2>Issue a licence</h2>
    <div class=row><div class=lbl><label for=n>Licensee</label>
      <span class=hint>shown in their panel</span></div>
      <input id=n placeholder="Jane Smith"></div>
    <div class=row><div class=lbl><label for=note>Reference</label>
      <span class=hint>order number or email — for your records only</span></div>
      <input id=note placeholder="order #1234"></div>
    <div class=row><div class=lbl><label for=ed>Type</label></div>
      <select id=ed onchange="document.getElementById('days').value=this.value==='review'?'90':'0'">
        <option value=pro>Purchase — perpetual</option>
        <option value=review>Reviewer / colleague</option>
      </select></div>
    <div class=row><div class=lbl><label for=days>Expires after</label>
      <span class=hint>0 = never</span></div>
      <input id=days type=number value=0 min=0></div>
    <div class=row><div class=lbl><label for=mach>Machine ID</label>
      <span class=hint>optional — locks it to one machine</span></div>
      <input id=mach placeholder="(leave empty for any machine)"></div>
    <div class=row><button onclick=issue()>Issue key</button></div>
    <div id=out></div>
  </div>

  <div class=card><h2>Check a key</h2>
    <div class=row><input id=chk placeholder="YW1.…"><button class=ghost onclick=check()>Check</button></div>
    <div id=chkout class=msg></div>
  </div>

  <div class=card><h2>Issued (${state.ledger.length})</h2>
    ${state.ledger.length?`<table><tr><th>Date</th><th>Licensee</th><th>Type</th>
      <th>Expires</th><th>Reference</th><th></th></tr>
      ${state.ledger.map(r=>`<tr>
        <td class=k>${esc(r.issued)}</td><td>${esc(r.name)}</td>
        <td><span class="pill ${r.revoked?'rev':''}">${r.revoked?'revoked':esc(r.edition)}</span></td>
        <td class=k>${esc(r.expires||'never')}</td><td class=k>${esc(r.note||'')}</td>
        <td><button class=ghost style="padding:3px 9px;font-size:11px"
            onclick="showKey('${esc(r.id)}')">key</button></td></tr>`).join('')}
      </table>`:'<div class=msg style="color:var(--dim)">Nothing issued yet.</div>'}
  </div>`;
}

async function genkey(){ await api('/api/keygen',{}); load(); }
async function issue(){
  const body={name:$('#n').value.trim(),note:$('#note').value.trim(),
    edition:$('#ed').value,days:parseInt($('#days').value||'0',10),machine:$('#mach').value.trim()};
  if(!body.name){ alert('Who is it for?'); return; }
  const r=await api('/api/issue',body);
  if(!r.ok){ $('#out').innerHTML=`<div class=msg style="color:var(--bad)">${esc(r.error)}</div>`; return; }
  $('#out').innerHTML=`<div class="keyout mono" id=newkey>${esc(r.key)}</div>
    <div class=row><button onclick="copy(document.getElementById('newkey').textContent)">Copy key</button>
    <span class=hint>send this to ${esc(body.name)} — they paste it into the Licence card</span></div>`;
  state=await api('/api/state');
}
async function check(){
  const r=await api('/api/check',{key:$('#chk').value.trim()});
  $('#chkout').innerHTML=r.ok
    ?`<span style="color:var(--good)">Valid</span> — ${esc(r.payload.n)} · ${esc(r.payload.e)}
      · expires ${esc(r.payload.x||'never')}${r.payload.m?' · machine '+esc(r.payload.m):''}`
    :`<span style="color:var(--bad)">Not valid for this signing key.</span>`;
}
async function showKey(id){
  const r=await api('/api/key',{id});
  if(r.ok) prompt('Licence key (copy it):', r.key);
}
function copy(t){ navigator.clipboard.writeText(t); }
load();
</script></body></html>"""


def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="yewee licence admin")

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.get("/api/state")
    def state():
        secret = load_secret()
        return JSONResponse({
            "has_key": secret is not None,
            "public_key": ed.public_key(secret).hex() if secret else "",
            "key_path": str(KEY_PATH),
            "ledger": [{k: v for k, v in row.items() if k != "key"} for row in ledger()],
        })

    @app.post("/api/keygen")
    def keygen():
        if load_secret() is not None:
            return JSONResponse({"ok": False, "error": "A signing key already exists."})
        save_secret(secrets.token_bytes(32))
        return JSONResponse({"ok": True})

    @app.post("/api/issue")
    async def issue(request: Request):
        body = await request.json()
        secret = load_secret()
        if secret is None:
            return JSONResponse({"ok": False, "error": "No signing key yet."})
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"ok": False, "error": "Licensee name is required."})
        days = int(body.get("days") or 0)
        payload = {"v": 1, "p": "yewee", "e": body.get("edition") or "pro",
                   "n": name, "i": date.today().isoformat(),
                   "k": secrets.token_hex(6)}
        if days > 0:
            payload["x"] = (date.today() + timedelta(days=days)).isoformat()
        machine = (body.get("machine") or "").strip()
        if machine:
            payload["m"] = machine
        key = encode_key(payload, secret)
        record({"id": payload["k"], "issued": payload["i"], "name": name,
                "edition": payload["e"], "expires": payload.get("x", ""),
                "machine": machine, "note": (body.get("note") or "").strip(),
                "revoked": False, "key": key})
        return JSONResponse({"ok": True, "key": key})

    @app.post("/api/check")
    async def check(request: Request):
        body = await request.json()
        secret = load_secret()
        pub = ed.public_key(secret).hex() if secret else ""
        payload = decode_key(body.get("key", ""), public_key_hex=pub) if pub else None
        return JSONResponse({"ok": payload is not None, "payload": payload or {}})

    @app.post("/api/key")
    async def get_key(request: Request):
        body = await request.json()
        for row in ledger():
            if row.get("id") == body.get("id"):
                return JSONResponse({"ok": True, "key": row.get("key", "")})
        return JSONResponse({"ok": False})

    return app


def main() -> int:
    import uvicorn
    port = int(os.environ.get("YEWEE_ADMIN_PORT", "8091"))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  yewee licence admin -> {url}")
    print(f"  vendor folder: {vendor_dir()}")
    print("  (local only — never expose this; it can mint licences)\n")
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
