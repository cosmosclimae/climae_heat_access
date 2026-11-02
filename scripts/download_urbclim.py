#!/usr/bin/env python3
"""Download UrbClim daily files (VITO PROVIDE, 100 m, 2008-2017) for the French cities.

Adapted from VITO's download_data.py: finds each city folder in the server
listing (case-insensitive), streams every .nc file to
<out>/<city_lowercase>/, skips files already complete, retries on failure.

Usage (WSL):
  python scripts/download_urbclim.py --out /path/to/inputs/urbclim
  python scripts/download_urbclim.py --out ... --cities Paris Lyon
"""
import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

BASE = "https://provide.marvin.vito.be/ftp/compressed_daily/"
CITIES = ["Bordeaux", "Lille", "Lyon", "Marseille", "Montpellier",
          "Nantes", "Nice", "Paris", "Strasbourg", "Toulouse"]


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def listing(url, session):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return re.findall(r'href="([^"?#]+)"', r.text)


def remote_size(url, session):
    try:
        h = session.head(url, timeout=60, allow_redirects=True)
        return int(h.headers.get("Content-Length", -1))
    except Exception:
        return -1


def fetch(url, dest, session, retries=4):
    size = remote_size(url, session)
    if dest.exists() and size > 0 and dest.stat().st_size == size:
        print(f"  ok (already present) {dest.name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if size > 0 and tmp.stat().st_size != size:
                raise IOError(f"taille {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
            return True
        except Exception as e:
            print(f"  attempt {k}/{retries} failed for {dest.name}: {e}")
            time.sleep(5 * k)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cities", nargs="*", default=CITIES)
    ap.add_argument("--debug", action="store_true", help="print the links of the root page")
    a = ap.parse_args()
    s = requests.Session()
    root = listing(BASE, s)
    if a.debug:
        print("links found in the listing:", root[:60])
    # liens relatifs ou absolus : on garde le dernier segment de chaque dossier
    folders = {}
    for x in root:
        seg = x.rstrip("/").rsplit("/", 1)[-1]
        if seg and seg not in ("..", ".") and not seg.endswith(".nc"):
            folders.setdefault(norm(seg), seg)
    failed = []
    for city in a.cities:
        key = folders.get(norm(city)) or next((v for k, v in folders.items() if norm(city) in k), None)
        cands = [key] if key else [city, city.lower(), city.upper()]
        url, files = None, []
        for c in cands:
            try:
                u = BASE + c + "/"
                files = sorted({f.rsplit("/", 1)[-1] for f in listing(u, s) if f.endswith(".nc")})
                if files:
                    url = u
                    break
            except requests.HTTPError:
                continue
        if not url:
            print(f"[{city}] folder not found on the server (tried: {', '.join(cands)})")
            failed.append(city)
            continue
        dest = Path(a.out) / norm(city)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{city}] {len(files)} fichiers -> {dest}")
        for fn in files:
            name = fn.rsplit("/", 1)[-1]
            if not fetch(url + name, dest / name, s):
                failed.append(f"{city}/{name}")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
