#!/usr/bin/env python3
"""Download Copernicus HRL Tree Cover Density 10 m tiles for France + build a VRT.

Source: Copernicus Data Space Ecosystem (CDSE), collection CLMS,
products CLMS_HRLVLCC_TCD_S{year}_R10m_E{xx}N{yy}_03035_V01_R00 (GeoTIFF,
EPSG:3035, 100 km EEA tiles, reference years 2018-2024).

Credentials: a free CDSE account, passed through environment variables
    export CDSE_USER=...   CDSE_PASSWORD=...
Usage:
    python scripts/tcd_download.py --year 2024 --departements <departements.geojson> \
        --outdir /path/to/inputs/tcd_2024 \
        --vrt /path/to/inputs/tcd_2024_fr.vrt
    python scripts/tcd_download.py ... --list-only     (tiles + catalogue check, no download)
Then set `canopy_raster` in config/config.yaml to the VRT.
Already downloaded tiles are skipped (resumable).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def france_tiles(geojson):
    """EEA 100-km tiles (lower-left naming E{x/1e5}N{y/1e5}) intersecting the départements."""
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.read_file(geojson).to_crs(3035)
    union = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.unary_union
    x0, y0, x1, y1 = union.bounds
    tiles = []
    for ex in range(int(x0 // 1e5), int(x1 // 1e5) + 1):
        for ny in range(int(y0 // 1e5), int(y1 // 1e5) + 1):
            if box(ex * 1e5, ny * 1e5, (ex + 1) * 1e5, (ny + 1) * 1e5).intersects(union):
                tiles.append(f"E{ex:02d}N{ny:02d}")
    return tiles


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def find_product(name):
    q = urllib.parse.quote(f"Collection/Name eq 'CLMS' and Name eq '{name}'", safe="")
    res = get_json(f"{CATALOGUE}?$filter={q}&$select=Id,Name,ContentLength")
    return res["value"][0] if res.get("value") else None


def token():
    user, pwd = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASSWORD")
    if not user or not pwd:
        sys.exit("CDSE_USER / CDSE_PASSWORD missing (free account at dataspace.copernicus.eu)")
    data = urllib.parse.urlencode({"client_id": "cdse-public", "grant_type": "password",
                                   "username": user, "password": pwd}).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=60) as r:
        return json.load(r)["access_token"]


def download(pid, dest, tok):
    req = urllib.request.Request(DOWNLOAD.format(id=pid), headers={"Authorization": f"Bearer {tok}"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--departements", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vrt", required=True)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tiles = france_tiles(a.departements)
    print(f"{len(tiles)} EEA 100-km tiles intersect metropolitan France")
    found, missing = [], []
    for t in tiles:
        name = f"CLMS_HRLVLCC_TCD_S{a.year}_R10m_{t}_03035_V01_R00"
        p = find_product(name)
        (found if p else missing).append((t, p))
    print(f"CDSE catalogue: {len(found)} found, {len(missing)} missing {[t for t, _ in missing]}")
    if a.list_only:
        return
    tok, t0 = token(), time.time()
    files = []
    for t, p in found:
        dest = out / f"{p['Name']}.tif"
        files.append(dest)
        if dest.exists() and dest.stat().st_size == p.get("ContentLength", dest.stat().st_size):
            continue
        if time.time() - t0 > 500:              # token valid ~10 min
            tok, t0 = token(), time.time()
        print(f"downloading {p['Name']} ({p.get('ContentLength', 0) / 1e6:.0f} MB)")
        download(p["Id"], dest, tok)
    # zipped products, if any: extract the GeoTIFF
    tifs = []
    for f in files:
        with open(f, "rb") as fh:
            magic = fh.read(4)
        if magic[:2] == b"PK":
            import zipfile
            with zipfile.ZipFile(f) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
                z.extract(inner, out / "unzipped")
                tifs.append(out / "unzipped" / inner)
        else:
            tifs.append(f)
    if shutil.which("gdalbuildvrt"):
        subprocess.run(["gdalbuildvrt", a.vrt, *map(str, tifs)], check=True)
    else:
        from osgeo import gdal
        gdal.BuildVRT(a.vrt, [str(t) for t in tifs])
    print(f"VRT written: {a.vrt} ({len(tifs)} tiles) -> set canopy_raster in config.yaml")


if __name__ == "__main__":
    main()
