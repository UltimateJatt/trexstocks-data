"""Send results to Cloudflare KV in one bulk upload.

Needs three GitHub Secrets: CF_API_TOKEN, CF_ACCOUNT_ID, CF_KV_NAMESPACE_ID.
Every payload is also saved to data/out/ so you can inspect it in the run logs.
"""
import json
import os

import requests

from . import config
from .util import clean, log

API = "https://api.cloudflare.com/client/v4/accounts/{a}/storage/kv/namespaces/{n}"


def _base():
    a, n = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_KV_NAMESPACE_ID")
    t = os.environ.get("CF_API_TOKEN")
    if not (a and n and t):
        return None, None
    return API.format(a=a, n=n), {"Authorization": f"Bearer {t}"}


def get(key):
    base, h = _base()
    if not base:
        return None
    r = requests.get(f"{base}/values/{key}", headers=h, timeout=30)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def put(payloads: dict, dry=False):
    out_dir = config.ROOT / "data" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    body = []
    for k, v in payloads.items():
        text = json.dumps(clean(v), separators=(",", ":"))
        (out_dir / f"{k}.json").write_text(text)
        body.append({"key": k, "value": text})
    sizes = ", ".join(f"{b['key']} {len(b['value']) // 1024}KB" for b in body)
    base, h = _base()
    if dry or not base:
        log(f"DRY RUN (not sent to Cloudflare): {sizes}")
        return
    r = requests.put(f"{base}/bulk", headers={**h, "Content-Type": "application/json"},
                     data=json.dumps(body), timeout=60)
    if r.status_code != 200 or not r.json().get("success"):
        raise RuntimeError(f"Cloudflare KV upload failed: {r.status_code} {r.text[:300]}")
    log(f"published to Cloudflare KV: {sizes}")
