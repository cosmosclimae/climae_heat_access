#!/usr/bin/env python3
"""Elevation on each UrbClim grid (Copernicus DEM GLO-30, public AWS bucket).

For every city folder in inputs/urbclim/<city>/, reads the 100 m grid of one
WBGT file (EPSG:3035), downloads the 1x1 degree Copernicus DEM tiles that
cover it (cached in <urbclim_dir>/_dem/), averages the 30 m DEM onto the 100 m
UrbClim pixels and writes <city>/elevation.nc (variable `z`, metres).
Used by uhi_offsets.py to pick rural reference pixels at the same elevation.

Usage (WSL, network needed):
  python scripts/dem_urbclim.py --urbclim-dir /path/to/inputs/urbclim
"""
import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
import xarray as xr
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


def tile_name(lat, lon):
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def fetch(name, cache):
    p = cache / f"{name}.tif"
    if p.exists() and p.stat().st_size > 0:
        return p
    r = requests.get(URL.format(n=name), stream=True, timeout=120)
    if r.status_code == 404:          # sea-only tile
        return None
    r.raise_for_status()
    tmp = p.with_suffix(".part")
    with open(tmp, "wb") as f:
        for c in r.iter_content(1 << 20):
            f.write(c)
    tmp.replace(p)
    print(f"  DEM {name} ({p.stat().st_size / 1e6:.0f} Mo)")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urbclim-dir", required=True)
    a = ap.parse_args()
    root = Path(a.urbclim_dir)
    cache = root / "_dem"
    cache.mkdir(exist_ok=True)
    inv = Transformer.from_crs(3035, 4326, always_xy=True)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = next(iter(sorted(cdir.glob("WBGT_year_daily_max_*.nc"))), None)
        if f is None:
            continue
        g = xr.open_dataset(f)
        x, y = g.x.values, g.y.values
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        X0, X1 = x.min() - abs(dx) / 2, x.max() + abs(dx) / 2
        Y0, Y1 = y.min() - abs(dy) / 2, y.max() + abs(dy) / 2
        lo, la = inv.transform([X0, X1, X0, X1], [Y0, Y0, Y1, Y1])
        tiles = [fetch(tile_name(i, j), cache)
                 for i in range(math.floor(min(la)), math.floor(max(la)) + 1)
                 for j in range(math.floor(min(lo)), math.floor(max(lo)) + 1)]
        tiles = [t for t in tiles if t is not None]
        srcs = [rasterio.open(t) for t in tiles]
        mosaic, transform = merge(srcs)
        crs = srcs[0].crs
        for s in srcs:
            s.close()
        # destination grid: UrbClim pixel centres, rows in the file's y order
        dst_transform = rasterio.transform.Affine(abs(dx), 0, X0, 0, -abs(dy), Y1)
        tmp = np.full((len(y), len(x)), np.nan, dtype=np.float32)
        reproject(mosaic[0].astype(np.float32), tmp, src_transform=transform, src_crs=crs,
                  dst_transform=dst_transform, dst_crs="EPSG:3035", resampling=Resampling.average,
                  src_nodata=-32767, dst_nodata=np.nan)
        dst = tmp if dy < 0 else tmp[::-1]          # tmp rows go north -> south
        xr.Dataset({"z": (("y", "x"), dst)}, coords={"y": y, "x": x},
                   attrs={"crs": "EPSG:3035", "source": "Copernicus DEM GLO-30, mean on UrbClim 100 m grid"}
                   ).to_netcdf(cdir / "elevation.nc")
        print(f"{cdir.name}: z {np.nanmin(dst):.0f}-{np.nanmax(dst):.0f} m, NaN {int(np.isnan(dst).sum())}")


if __name__ == "__main__":
    main()
