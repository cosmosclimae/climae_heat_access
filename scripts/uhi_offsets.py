#!/usr/bin/env python3
"""Urban heat island offsets for outdoor facilities from UrbClim (VITO, 100 m, daily).

For each city folder in inputs/urbclim/<city>/ and one year:
  dmax_i,d  = WBGTmax_UrbClim(pixel_i, d)  - mean over RURAL pixels of the domain (d)
  dmean_i,d = WBGTmean_UrbClim(pixel_i, d) - mean over RURAL pixels of the domain (d)
Rural pixels = land (inside the departement polygons) and farther than
uhi.rural_min_dist_m from any inhabited Filosofi carreau centroid. ERA5-Land
has no urban tile, so its WBGT is a non-urban value: the offset is the urban
excess UrbClim adds on top of it.
It does not depend on how UrbClim formulates WBGT (shade/sun, radiation scheme):
only the urban-minus-surroundings difference is transferred.
Facilities outside every domain get no offset. Months = climate.months.

Output parquet: fid, city, ldate (YYYYMMDD int), dmax, dmean  (float32)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer
from shapely import contains_xy
from shapely.ops import unary_union


def land_mask(xx, yy, land3035):
    return contains_xy(land3035, xx, yy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--urbclim-dir", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--departements", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    months = cfg["climate"]["months"]
    fac = pd.read_parquet(a.facilities, columns=["fid", "layer", "lon", "lat"])
    fac = fac[fac.layer == "outdoor"]
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    fx, fy = tr.transform(fac.lon.to_numpy(), fac.lat.to_numpy())
    land = unary_union(gpd.read_file(a.departements).to_crs(3035).geometry.values)
    from scipy.spatial import cKDTree
    pp = pd.read_parquet(a.population, columns=["x3035", "y3035", "ind"])
    ptree = cKDTree(np.c_[pp.x3035[pp.ind > 0], pp.y3035[pp.ind > 0]])
    ucfg = cfg.get("uhi", {})
    rmin = ucfg.get("rural_min_dist_m", 400)
    taken = np.zeros(len(fac), bool)
    parts, log = [], []
    for cdir in sorted(p for p in Path(a.urbclim_dir).iterdir() if p.is_dir() and not p.name.startswith("_")):
        fmax = cdir / f"WBGT_year_daily_max_{a.year}.nc"
        fmean = cdir / f"WBGT_year_daily_mean_{a.year}.nc"
        if not (fmax.exists() and fmean.exists()):
            log.append(f"{cdir.name}: missing {a.year}")
            continue
        dmx = xr.open_dataset(fmax)
        crs = str(dmx.attrs.get("crs", "EPSG:3035"))
        if "3035" not in crs:
            raise SystemExit(f"{fmax}: unexpected CRS {crs} (reproject facilities accordingly)")
        vmax = dmx[list(dmx.data_vars)[0]]
        dmn = xr.open_dataset(fmean)
        vmean = dmn[list(dmn.data_vars)[0]]
        t = pd.to_datetime(vmax.time.values)
        keep = np.isin(t.month, months)
        vmax, vmean = vmax.isel(time=np.where(keep)[0]), vmean.isel(time=np.where(keep)[0])
        x, y = vmax.x.values, vmax.y.values
        dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
        XX, YY = np.meshgrid(x, y)
        lm = land_mask(XX, YY, land)
        dist, _ = ptree.query(np.c_[XX.ravel(), YY.ravel()])
        lm = lm & (dist.reshape(XX.shape) > rmin)
        if lm.sum() < 200:
            log.append(f"{cdir.name}: only {int(lm.sum())} rural pixels, skipped")
            continue
        Amax = vmax.values.astype(np.float32)                # time x y x
        Amean = vmean.values.astype(np.float32)
        zf = cdir / "elevation.nc"
        Z = xr.open_dataset(zf).z.values if zf.exists() else None
        inside = (~taken) & (fx >= x.min() - dx / 2) & (fx <= x.max() + dx / 2) & \
                 (fy >= y.min() - dy / 2) & (fy <= y.max() + dy / 2)
        idx = np.where(inside)[0]
        ix = np.clip(np.round((fx[idx] - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
        iy = np.clip(np.round((fy[idx] - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
        if Z is None:
            # whole-domain rural mean (no DEM): biased where the rural ring is hilly
            ref_max = np.nanmean(np.where(lm[None], Amax, np.nan), axis=(1, 2))[:, None]
            ref_mean = np.nanmean(np.where(lm[None], Amean, np.nan), axis=(1, 2))[:, None]
            how = "domain"
        else:
            # rural pixels at the same elevation as each facility (+/- band, widened
            # until min_pixels rural pixels are found): removes the altitude effect
            band = ucfg.get("elev_band_m", 50)
            nmin = ucfg.get("min_rural_pixels", 100)
            zr = Z[lm]
            ok = np.isfinite(zr)
            zr = zr[ok]
            Rmax = Amax[:, lm][:, ok]                         # time x rural px
            Rmean = Amean[:, lm][:, ok]
            o = np.argsort(zr)
            zr, Rmax, Rmean = zr[o], Rmax[:, o], Rmean[:, o]
            cmax = np.concatenate([np.zeros((Rmax.shape[0], 1)), np.nancumsum(Rmax, 1)], 1)
            cmean = np.concatenate([np.zeros((Rmean.shape[0], 1)), np.nancumsum(Rmean, 1)], 1)
            zi = Z[iy, ix]
            zi = np.where(np.isfinite(zi), zi, np.nanmedian(zr))
            lo = np.searchsorted(zr, zi - band)
            hi = np.searchsorted(zr, zi + band)
            w = band
            for _ in range(6):
                small = (hi - lo) < nmin
                if not small.any():
                    break
                w = w * 2
                lo = np.where(small, np.searchsorted(zr, zi - w), lo)
                hi = np.where(small, np.searchsorted(zr, zi + w), hi)
            n = np.maximum(hi - lo, 1)
            ref_max = (cmax[:, hi] - cmax[:, lo]) / n
            ref_mean = (cmean[:, hi] - cmean[:, lo]) / n
            how = f"elevation-matched (+/-{band} m)"
        dmax = Amax[:, iy, ix] - ref_max                      # time x facilities
        dmean = Amean[:, iy, ix] - ref_mean
        ld = np.array([int(s.strftime("%Y%m%d")) for s in pd.to_datetime(vmax.time.values)])
        nT, nF = dmax.shape
        parts.append(pd.DataFrame({"fid": np.tile(fac.fid.to_numpy()[idx], nT).astype(np.int32),
                                   "city": cdir.name, "ldate": np.repeat(ld, nF).astype(np.int32),
                                   "dmax": dmax.ravel().astype(np.float32),
                                   "dmean": dmean.ravel().astype(np.float32)}))
        taken[idx] = True
        jja = np.isin(pd.to_datetime(vmax.time.values).month, [6, 7, 8])
        log.append(f"{cdir.name} [{how}]: {nF} facilities, {int(lm.sum())} rural px, JJA mean dmax {np.nanmean(dmax[jja]):+.2f} "
                   f"(p90 {np.nanpercentile(np.nanmean(dmax[jja], 0), 90):+.2f}) dmean {np.nanmean(dmean[jja]):+.2f}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["fid", "city", "ldate", "dmax", "dmean"])
    df = df.dropna(subset=["dmax", "dmean"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{a.year}:", *log, sep="\n  ")


if __name__ == "__main__":
    main()
