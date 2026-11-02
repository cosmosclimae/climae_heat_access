"""Shared helpers: ERA5-Land file access, grids, catchment matrices."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

TIME_NAMES = ["valid_time", "time"]
LAT_NAMES = ["latitude", "lat"]
LON_NAMES = ["longitude", "lon"]
VAR_ALIASES = {
    "t2m": ["t2m", "2t", "VAR_2T"], "d2m": ["d2m", "2d", "VAR_2D"],
    "sp": ["sp", "VAR_SP"], "u10": ["u10", "10u", "VAR_10U"], "v10": ["v10", "10v", "VAR_10V"],
    "ssrd": ["ssrd", "VAR_SSRD"],
}


def find(ds, cands):
    for c in cands:
        if c in ds.variables or c in ds.dims:
            return c
    return None


def open_era5(path):
    ds = xr.open_dataset(path)
    tn, la, lo = find(ds, TIME_NAMES), find(ds, LAT_NAMES), find(ds, LON_NAMES)
    if tn is None or la is None or lo is None:
        raise ValueError(f"{path}: dimensions temps/lat/lon introuvables")
    return ds, tn, la, lo


def var(ds, key):
    name = find(ds, VAR_ALIASES[key])
    if name is None:
        raise ValueError(f"variable {key} absente ({VAR_ALIASES[key]})")
    return name


def gaussian_w(tau, tau0, w0):
    beta = tau0 / math.sqrt(math.log(1.0 / w0))
    w = np.exp(-(tau / beta) ** 2)
    return np.where(tau <= tau0, w, 0.0)


def build_W(od, nJ, nI, tau0, w0, fid_to_col, jcol="jid", fcol="fid", tcol="tau_min"):
    """Sparse weights J x I (columns = position in the facility subset)."""
    sub = od[od[tcol] <= tau0]
    col = fid_to_col[sub[fcol].to_numpy()]
    ok = col >= 0
    rows = sub[jcol].to_numpy()[ok]
    w = gaussian_w(sub[tcol].to_numpy()[ok], tau0, w0)
    return sp.csr_matrix((w, (rows, col[ok])), shape=(nJ, nI))


def fid_map(fids, n_total):
    m = np.full(n_total, -1, dtype=np.int64)
    m[np.asarray(fids)] = np.arange(len(fids))
    return m


def local_hour_index(start_hours_cfg):
    return {h: k for k, h in enumerate(start_hours_cfg)}


def load_pop(path, groups):
    pop = pd.read_parquet(path)
    return pop, {g: pop[g].to_numpy(dtype=np.float64) for g in groups}


# --------------------------------------------------------------------------
# facility-level thermal availability
# --------------------------------------------------------------------------
def outdoor_table(fac_path, canopy_path, fcells_path, wcells_path, access="all"):
    """Outdoor facilities (ordered by fid) with cell index k, canopy, attributes."""
    fac = pd.read_parquet(fac_path)
    o = fac[fac.layer == "outdoor"]
    if access == "public":
        o = o[o.access_public]
    o = o.sort_values("fid").reset_index(drop=True)
    can = pd.read_parquet(canopy_path)
    fc = pd.read_parquet(fcells_path)[["fid", "cell"]]
    wc = pd.read_parquet(wcells_path)[["cell", "k"]]
    o = o.merge(can, on="fid", how="left").merge(fc, on="fid", how="left").merge(wc, on="cell", how="left")
    for c in can.columns:
        if c.startswith("canopy_r"):
            o[c] = o[c].fillna(0.0)
    return o


def scenario_arrays(o, cfg, scen, canopy_col=None):
    """Shade fraction f_i, surface class name per facility, lighting switch."""
    mod = cfg["modifiers"][scen]
    if canopy_col is None:
        r = mod.get("canopy_buffer", cfg["facilities"]["canopy_buffer_m"][0])
        canopy_col = f"canopy_r{int(r)}"
    f = np.zeros(len(o))
    if mod["canopy"]:
        f = o[canopy_col].to_numpy(dtype=float).clip(0, 1)
    if mod["cover"]:
        f = np.where(o.covered.to_numpy(), 1.0, f)
    cls = o.surface_class.to_numpy() if mod["surface"] else np.full(len(o), "reference", dtype=object)
    return f, cls, bool(mod["lighting"])


def facility_wbgt(ds_arrays, k, f, cls, t_sel):
    """WBGT_i(t) = f*shade + (1-f)*sun[class]  for facilities (k index) and time slice."""
    shade = ds_arrays["wbgt_shade"][:, t_sel][k]
    out = np.empty_like(shade)
    for c in np.unique(cls):
        m = cls == c
        sun = ds_arrays[f"wbgt_sun_{c}"][:, t_sel][k[m]]
        out[m] = f[m, None] * shade[m] + (1 - f[m, None]) * sun
    return out


def availability(w, q, light_ok, band=None):
    """Binary (band None) or graded thermal availability. NaN WBGT (cold or
    invalid physics, e.g. ice wick) is treated as available."""
    w = np.where(np.isfinite(w), w, -99.0)
    if band is None:
        a = (w < q)
    else:
        a = np.where(w < q - band, 1.0, np.where(w < q, 0.5, 0.0))
    return a * light_ok
