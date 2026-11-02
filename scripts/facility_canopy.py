#!/usr/bin/env python3
"""Tree-canopy shade fraction around each outdoor facility.

f_canopy = mean tree cover density (0-100 %, Copernicus HRL TCD 10 m) inside a
disc of radius r around the Data ES point, sampled on a 10-m lattice, / 100.
If no raster is configured, f_canopy = 0 everywhere (flag canopy_available=False):
the DAG still runs and the canopy modifier is simply inactive.
Caveat: the Data ES point is often the installation centroid, not the field.
"""
import argparse

import numpy as np
import pandas as pd


def disc_offsets(r, step=10.0):
    g = np.arange(-r, r + step / 2, step)
    xx, yy = np.meshgrid(g, g)
    m = xx ** 2 + yy ** 2 <= r ** 2
    return xx[m], yy[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--raster", default="")
    ap.add_argument("--buffers", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fac = pd.read_parquet(a.facilities, columns=["fid", "layer", "lon", "lat"])
    fac = fac[fac.layer == "outdoor"].copy()
    out = pd.DataFrame({"fid": fac.fid.to_numpy()})
    if not a.raster:
        for r in a.buffers:
            out[f"canopy_r{int(r)}"] = 0.0
        out["canopy_available"] = False
        out.to_parquet(a.out, index=False)
        print("no canopy raster: canopy fractions set to 0")
        return
    import rasterio
    from pyproj import Transformer
    with rasterio.open(a.raster) as src:
        tr = Transformer.from_crs(4326, src.crs, always_xy=True)
        x, y = tr.transform(fac.lon.to_numpy(), fac.lat.to_numpy())
        nod = src.nodata
        for r in a.buffers:
            dx, dy = disc_offsets(r)
            vals = np.zeros(len(fac))
            cnt = np.zeros(len(fac))
            for ox, oy in zip(dx, dy):
                v = np.array([s[0] for s in src.sample(zip(x + ox, y + oy))], dtype=float)
                ok = np.isfinite(v) & (v >= 0) & (v <= 100)
                if nod is not None:
                    ok &= v != nod
                vals[ok] += v[ok]
                cnt[ok] += 1
            out[f"canopy_r{int(r)}"] = np.where(cnt > 0, vals / np.maximum(cnt, 1) / 100.0, 0.0)
    out["canopy_available"] = True
    out.to_parquet(a.out, index=False)
    print(out.describe())


if __name__ == "__main__":
    main()
