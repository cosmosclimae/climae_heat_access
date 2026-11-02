#!/usr/bin/env python3
"""Hourly Liljegren WBGT (sun per surface class + OSHA shade) for one month.

Only ERA5-Land cells hosting outdoor facilities, only intervals whose LOCAL
start hour (Europe/Paris, DST handled) is in climate.local_start_hours.

ERA5-Land conventions handled here:
  * ssrd at step t = accumulation since 00 UTC -> energy of (t-1h, t] is
    val(t) - val(t-1), except at 01 UTC where val(t) is already one hour
    (same rule as climae_era5land_dashboard deaccum_generic);
  * the interval (t-1h, t] uses instantaneous T, Td, sp, wind at t-1 and t,
    averaged at mid-interval (climae.io policy v2);
  * solar geometry at mid-interval via 60-s quadrature (policy v2), BEFORE any
    local-time relabelling; the local label is the interval START hour.

Output netCDF (int16, 0.01 degC; elevation 0.1 deg):
  wbgt_shade(k, t), wbgt_sun_<class>(k, t), elev_mid(k, t)
  coords: k (cells, wbgt_cells.parquet order), t (interval start UTC),
          local_date (YYYYMMDD int), local_hour (int)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import liljegren_np as L                       # noqa: E402
from common import open_era5, var              # noqa: E402


def deaccum(acc, hours_utc):
    out = np.empty_like(acc)
    out[0] = np.nan
    diff = acc[1:] - acc[:-1]
    out[1:] = np.where((hours_utc[1:] == 1)[:, None], acc[1:], np.maximum(diff, 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=40, help="intervals per vectorised batch")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(a.config).resolve().parents[1]
    pol = L.Policy.load(root / cfg["climate"]["policy"])
    sens = L.Sensors.load(root / cfg["climate"]["sensors"])
    surfaces = {k: tuple(v) for k, v in cfg["facilities"]["surface_classes"].items()}
    hours_keep = set(cfg["climate"]["local_start_hours"])
    tz = cfg["climate"]["timezone"]

    cells = pd.read_parquet(a.cells).sort_values("k")
    ds, tn, la, lo = open_era5(a.era5)
    il = xr.DataArray(cells.ilat.to_numpy(), dims="k")
    jl = xr.DataArray(cells.ilon.to_numpy(), dims="k")

    def get(key, scale=1.0):
        return (ds[var(ds, key)].isel({la: il, lo: jl}).values.astype(np.float64)) * scale   # (nt, k)

    t = pd.DatetimeIndex(ds[tn].values)
    if t.tz is not None:
        t = t.tz_convert(None)
    hours_utc = t.hour.to_numpy()
    t2m, d2m, psf = get("t2m"), get("d2m"), get("sp")
    ws = np.hypot(get("u10"), get("v10"))
    rh = L.rh_from_dew(t2m, d2m)
    energy = deaccum(get("ssrd"), hours_utc)                      # J/m2 over (t-1h, t]

    start = t - pd.Timedelta(hours=1)
    loc = start.tz_localize("UTC").tz_convert(tz)
    keep = np.where(np.isin(loc.hour, list(hours_keep)) & (np.arange(len(t)) >= 1))[0]
    nk, nt = len(cells), len(keep)
    lat = cells.lat.to_numpy()[:, None]
    lon = cells.lon.to_numpy()[:, None]

    names = ["wbgt_shade"] + [f"wbgt_sun_{s}" for s in surfaces]
    res = {n: np.full((nk, nt), np.nan, dtype=np.float32) for n in names}
    elev = np.full((nk, nt), np.nan, dtype=np.float32)
    flag_or = np.zeros((nk, nt), dtype=np.int16)
    for c0 in range(0, nt, a.chunk):
        sel = keep[c0:c0 + a.chunk]
        idx1 = sel                    # end of interval
        idx0 = sel - 1                # start of interval
        st = start[sel].to_numpy().astype("datetime64[ns]")[None, :]
        o = L.wbgt_interval(
            energy_j_m2=energy[idx1].T, start_utc=st, lat=lat, lon=lon,
            t0_k=t2m[idx0].T, t1_k=t2m[idx1].T, rh0=rh[idx0].T, rh1=rh[idx1].T,
            p0_pa=psf[idx0].T, p1_pa=psf[idx1].T, ws0=ws[idx0].T, ws1=ws[idx1].T,
            pol=pol, sens=sens, surfaces=surfaces)
        sl = slice(c0, c0 + len(sel))
        for n in names:
            res[n][:, sl] = o[n]
        elev[:, sl] = o["elev_mid"]
        flag_or[:, sl] = o["flags"]

    enc = {n: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4}
           for n in names}
    enc["elev_mid"] = {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -32768, "zlib": True, "complevel": 4}
    enc["flags"] = {"zlib": True, "complevel": 4}
    lk = loc[keep]
    out = xr.Dataset(
        {**{n: (("k", "t"), res[n]) for n in names}, "elev_mid": (("k", "t"), elev), "flags": (("k", "t"), flag_or)},
        coords={"k": cells.k.to_numpy(), "t": start[keep].to_numpy(),
                "local_date": ("t", (lk.year * 10000 + lk.month * 100 + lk.day).to_numpy().astype(np.int32)),
                "local_hour": ("t", lk.hour.to_numpy().astype(np.int16))},
        attrs={"source": str(a.era5), "method": "Liljegren (climae.io v2 policy) vectorised port",
               "surfaces": yaml.safe_dump(surfaces),
               "flag_counts": str({int(b): int(((flag_or & b) > 0).sum()) for b in [1, 2, 4, 8, 16, 32, 64]}),
               "nan_shade": int(np.isnan(res["wbgt_shade"]).sum())})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.out, encoding=enc)
    print(f"{a.era5}: {nk} cells x {nt} intervals ; flags {out.attrs['flag_counts']}")


if __name__ == "__main__":
    main()
