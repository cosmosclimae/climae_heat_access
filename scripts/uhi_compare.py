#!/usr/bin/env python3
"""UHI sensitivity: same window (2008-2017), facility WBGT without vs with UrbClim offsets.

Per city (carreaux inside the UrbClim domain):
  facilities and JJA offsets (mean, p90 of per-facility mean dmax),
  peak loss L_peak (mean, share served > loss_threshold), base vs UHI,
  heat-only desert hours per served person, base vs UHI.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

DIM = {5: 31, 6: 30, 7: 31, 8: 31, 9: 30}


def dh(path):
    d = pd.read_parquet(path, columns=["jid", "month", "n_desert", "n_days"])
    h = d.n_desert.to_numpy(float) / d.n_days.to_numpy(float) * d.month.map(DIM).to_numpy(float)
    return pd.Series(h).groupby(d.jid.to_numpy()).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--access-base", required=True)
    ap.add_argument("--access-uhi", required=True)
    ap.add_argument("--deserts-base", required=True)
    ap.add_argument("--deserts-uhi", required=True)
    ap.add_argument("--offsets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    q = cfg["climate"]["main_threshold"]
    hs = cfg.get("analysis", {}).get("heat_scenario", "nolight")
    thr = cfg.get("analysis", {}).get("loss_threshold", 0.25)
    udir = Path(cfg["uhi"]["urbclim_dir"])
    y0 = cfg["uhi"]["years"][0]

    pop = pd.read_parquet(a.population, columns=["jid", "x3035", "y3035", "ind"])
    b = pd.read_parquet(a.access_base, columns=["jid", "Lpeak_ind", "has_outdoor"])
    u = pd.read_parquet(a.access_uhi, columns=["jid", "Lpeak_ind"]).rename(columns={"Lpeak_ind": "Lpeak_uhi"})
    df = pop.merge(b, on="jid").merge(u, on="jid")
    df["dh_base"] = df.jid.map(dh(Path(a.deserts_base) / f"deserts_{hs}_q{q}.parquet")).fillna(0)
    df["dh_uhi"] = df.jid.map(dh(Path(a.deserts_uhi) / f"deserts_{hs}_q{q}.parquet")).fillna(0)
    off = pd.concat([pd.read_parquet(p) for p in a.offsets], ignore_index=True)
    off["m"] = off.ldate // 100 % 100
    rows = []
    for cdir in sorted(p for p in udir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = next(iter(sorted(cdir.glob("WBGT_year_daily_max_*.nc"))), None)
        if f is None:
            continue
        g = xr.open_dataset(f)
        x0, x1, ya, yb = float(g.x.min()), float(g.x.max()), float(g.y.min()), float(g.y.max())
        m = df.x3035.between(x0, x1) & df.y3035.between(ya, yb) & df.has_outdoor
        s = df[m]
        w = s.ind.to_numpy(float)
        if w.sum() == 0:
            continue
        o = off[(off.city == cdir.name) & off.m.isin([6, 7, 8])].groupby("fid").dmax.mean()
        rows.append({
            "city": cdir.name, "served_population": w.sum(), "facilities": int(o.size),
            "dmax_JJA_mean": float(o.mean()) if o.size else np.nan,
            "dmax_JJA_p90": float(o.quantile(.9)) if o.size else np.nan,
            "L_peak_base": float(np.average(s.Lpeak_ind, weights=w)),
            "L_peak_uhi": float(np.average(s.Lpeak_uhi, weights=w)),
            f"share_L_peak_gt_{thr}_base": float(w[s.Lpeak_ind.to_numpy() > thr].sum() / w.sum()),
            f"share_L_peak_gt_{thr}_uhi": float(w[s.Lpeak_uhi.to_numpy() > thr].sum() / w.sum()),
            "desert_h_heat_base": float(np.average(s.dh_base, weights=w)),
            "desert_h_heat_uhi": float(np.average(s.dh_uhi, weights=w)),
        })
    r = pd.DataFrame(rows)
    if len(r):
        r["desert_h_change_pct"] = 100 * (r.desert_h_heat_uhi / r.desert_h_heat_base - 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.out, index=False)
    print(r.round(3).to_string(index=False))
    print(f"(window {cfg['uhi']['years']}, q={q}, heat-only deserts '{hs}'; first year read: {y0})")


if __name__ == "__main__":
    main()
