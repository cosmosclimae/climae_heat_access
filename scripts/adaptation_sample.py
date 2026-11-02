#!/usr/bin/env python3
"""Draw the hours used to evaluate shading strategies (Monte Carlo sample).

Stratified by local hour: the same number of intervals per local hour, drawn
uniformly over the adaptation period and months. Writes the WBGT fields of
the sampled intervals for all facility cells (compact npz) + the weight of one
sampled interval in population-hours per season.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wbgt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=None, help="override config seed (independent evaluation samples)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"] if a.seed is None else a.seed)
    hours = cfg["climate"]["local_start_hours"]
    index = []                                          # (file, t, local_hour)
    for fi, p in enumerate(a.wbgt):
        lh = xr.open_dataset(p)["local_hour"].values
        index += [(fi, t, int(h)) for t, h in enumerate(lh)]
    index = np.array(index)
    per_h = max(1, a.n // len(hours))
    pick = []
    for h in hours:
        cand = np.where(index[:, 2] == h)[0]
        pick.append(rng.choice(cand, size=min(per_h, len(cand)), replace=False))
    pick = index[np.sort(np.concatenate(pick))]
    names, store = None, {}
    for fi in np.unique(pick[:, 0]):
        ds = xr.open_dataset(a.wbgt[fi])
        ts = pick[pick[:, 0] == fi, 1]
        if names is None:
            names = [v for v in ds.data_vars if v.startswith("wbgt_")] + ["elev_mid"]
        for v in names:
            store.setdefault(v, []).append(ds[v].values[:, ts].astype(np.float32))
    out = {v: np.concatenate(store[v], axis=1) for v in names}
    years = sorted({re.search(r"(\d{4})-\d{2}", Path(p).name).group(1) for p in a.wbgt})
    total_intervals = len(index) / len(years)           # intervals per season
    out["hour"] = pick[:, 2]
    out["season_hours_per_sample"] = np.array(total_intervals / len(pick))
    np.savez_compressed(a.out, **out)
    print(f"sampled {len(pick)} intervals out of {len(index)} ; 1 sample = {out['season_hours_per_sample']:.2f} season-hours")


if __name__ == "__main__":
    main()
