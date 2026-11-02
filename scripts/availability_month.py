#!/usr/bin/env python3
"""Facility thermal availability and temporary access deserts for one month.

For every modifier scenario s and threshold q:
  counts[s,q,i,h]  number of days the outdoor facility i is available at local hour h
  graded[q,i,h]    same with graded availability (main scenario only)
  deserts[s,q]     sparse (jid, h, n_days) : no available outdoor facility within the
                   main tau0 (main access scenario), for carreaux with A0 > 0.
Deserts use daily-hourly values (not climatology), as in the plan (Z_{j,d,h}).
Facility order = outdoor facilities, ALL access types, ordered by fid (the desert
matrix B uses the public subset; mapping through fid).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import outdoor_table, scenario_arrays, facility_wbgt, availability   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wbgt", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--canopy", required=True)
    ap.add_argument("--facility-cells", required=True)
    ap.add_argument("--wbgt-cells", required=True)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--catchment-meta", required=True)
    ap.add_argument("--offsets", default=None,
                    help="optional UHI offsets parquet (fid, ldate, dmax, dmean) added to facility WBGT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    hours = cfg["climate"]["local_start_hours"]
    hidx = {h: n for n, h in enumerate(hours)}
    qs = cfg["climate"]["thresholds"]
    scens = list(k for k in cfg["modifiers"] if k != "main")
    main_s = cfg["modifiers"]["main"]
    band = cfg["climate"]["graded_band"]
    min_el = cfg["climate"]["lighting_min_elevation_deg"]

    o = outdoor_table(a.facilities, a.canopy, a.facility_cells, a.wbgt_cells, access="all")
    k = o.k.to_numpy().astype(np.int64)
    lit = o.lit.to_numpy()
    nI, nH = len(o), len(hours)
    B = sp.load_npz(a.catchment).tocsr()
    meta = pd.read_parquet(a.catchment_meta)
    pos = pd.Series(np.arange(nI), index=o.fid.to_numpy())
    bcols = pos.loc[meta.fid.to_numpy()].to_numpy()          # B column -> row in o
    rows_ok = B.getnnz(1) > 0

    ds = xr.open_dataset(a.wbgt)
    arr = {v: ds[v].values.astype(np.float32) for v in ds.data_vars if v.startswith("wbgt_")}
    elev = ds["elev_mid"].values.astype(np.float32)
    ldate = ds["local_date"].values
    lhour = ds["local_hour"].values

    sc = {s: scenario_arrays(o, cfg, s) for s in scens}
    off = None
    if a.offsets:
        of = pd.read_parquet(a.offsets)
        of = of[of.fid.isin(pos.index)]
        of["row"] = pos.loc[of.fid.to_numpy()].to_numpy()
        off = {d: g for d, g in of.groupby("ldate")}
        peak_h = set(cfg.get("uhi", {}).get("peak_hours", [12, 13, 14, 15, 16, 17]))
    counts = np.zeros((len(scens), len(qs), nI, nH), dtype=np.uint8)
    graded = np.zeros((len(qs), nI, nH), dtype=np.float32)
    ndays = np.zeros(nH, dtype=np.uint8)
    des = {(s, q): np.zeros((B.shape[0], nH), dtype=np.uint8) for s in scens for q in qs}
    nan_count = 0
    for d in np.unique(ldate):
        t_sel = np.where(ldate == d)[0]
        hcol = np.array([hidx[h] for h in lhour[t_sel]])
        ndays[hcol] += 1
        el = elev[:, t_sel][k]
        for si, s in enumerate(scens):
            f, cls, light_on = sc[s]
            w = facility_wbgt(arr, k, f, cls, t_sel)
            if off is not None and int(d) in off:
                g = off[int(d)]
                pk = np.array([h in peak_h for h in lhour[t_sel]])
                w[g.row.to_numpy()] += np.where(pk[None, :], g.dmax.to_numpy()[:, None],
                                                g.dmean.to_numpy()[:, None]).astype(w.dtype)
            if si == 0:
                nan_count += int(np.isnan(w).sum())
            light_ok = (lit[:, None] | (el >= min_el)) if light_on else np.ones_like(w, dtype=bool)
            for qi, q in enumerate(qs):
                av = availability(w, q, light_ok)
                counts[si, qi][:, hcol] += av.astype(np.uint8)
                if s == main_s:
                    graded[qi][:, hcol] += availability(w, q, light_ok, band=band)
                n_av = B @ av[bcols].astype(np.float32)               # J x nh_day
                z = (n_av == 0) & rows_ok[:, None]
                des[(s, q)][:, hcol] += z.astype(np.uint8)
    out = {"fid": o.fid.to_numpy(), "hours": np.array(hours), "ndays": ndays, "scens": np.array(scens),
           "qs": np.array(qs), "counts": counts, "graded": graded, "nan_wbgt": np.array(nan_count)}
    for (s, q), m in des.items():
        j, h = np.nonzero(m)
        out[f"des_{s}_q{q}_j"] = j.astype(np.int32)
        out[f"des_{s}_q{q}_h"] = h.astype(np.int8)
        out[f"des_{s}_q{q}_n"] = m[j, h]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"{a.wbgt}: {nI} facilities, {len(np.unique(ldate))} days, NaN WBGT {nan_count}")


if __name__ == "__main__":
    main()
