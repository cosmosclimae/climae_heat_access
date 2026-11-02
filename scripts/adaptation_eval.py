#!/usr/bin/env python3
"""Out-of-sample evaluation of shading sets for one département.

The sets are chosen on the selection sample (adaptation_select.py). Here the
desert events are rebuilt on an INDEPENDENT sample of hours (other seed and/or
other years) with exactly the same rules as adaptation_dep.py, and we count the
heat-attributable desert person-hours that each fixed set removes.
Option --shade sets the post-intervention shade fraction (partial shading test).
Output: one row per (strategy, budget_pct) with avoided person-hours (total and
low-income) and the département totals of heat-attributable deserts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import outdoor_table, scenario_arrays, availability     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dep", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--canopy", required=True)
    ap.add_argument("--facility-cells", required=True)
    ap.add_argument("--wbgt-cells", required=True)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--catchment-meta", required=True)
    ap.add_argument("--shade", type=float, default=1.0,
                    help="shade fraction after intervention (1 = complete; partial < 1, never below current f_i)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    q = cfg["climate"]["main_threshold"]
    scen = cfg["modifiers"]["main"]
    min_el = cfg["climate"]["lighting_min_elevation_deg"]
    ad = cfg["adaptation"]

    pop = pd.read_parquet(a.population)
    jdep = pop.index[pop.dep == a.dep].to_numpy()
    B = sp.load_npz(a.catchment).tocsr()[jdep]
    meta = pd.read_parquet(a.catchment_meta)
    o = outdoor_table(a.facilities, a.canopy, a.facility_cells, a.wbgt_cells, access="public")
    o = o.set_index("fid").loc[meta.fid.to_numpy()].reset_index()
    used = np.unique(B.indices)
    f, cls, light_on = scenario_arrays(o, cfg, scen)

    z = np.load(a.sample)
    arr = {k: z[k] for k in z.files if k.startswith("wbgt_")}
    el = z["elev_mid"]
    k = o.k.to_numpy().astype(np.int64)
    T = el.shape[1]

    def wb(ix, fsh):
        shade = arr["wbgt_shade"][k[ix]]
        out = np.empty((len(ix), T), dtype=np.float32)
        for c in np.unique(cls[ix]):
            m = cls[ix] == c
            out[m] = fsh[m, None] * shade[m] + (1 - fsh[m, None]) * arr[f"wbgt_sun_{c}"][k[ix][m]]
        return out

    lit = o.lit.to_numpy()
    light_ok = (lit[used, None] | (el[k[used]] >= min_el)) if light_on else np.ones((len(used), T), bool)
    av_now = np.zeros((B.shape[1], T), dtype=np.float32)
    av_now[used] = availability(wb(used, f[used]), q, light_ok)
    av_sh = av_now.copy()
    av_sh[used] = availability(wb(used, np.maximum(f[used], a.shade)), q, light_ok)

    rows_ok = B.getnnz(1) > 0
    desert = (B @ av_now == 0) & rows_ok[:, None]
    ej, et = np.nonzero(desert)
    lo = np.zeros((B.shape[1], T), dtype=np.float32)
    lo[used] = light_ok
    heat_attr = np.asarray((B @ lo)[ej, et] > 0).ravel()
    ej, et = ej[heat_attr], et[heat_attr]

    scale = float(z["season_hours_per_sample"])
    pdep = pop.iloc[jdep]
    w_tot = pdep["ind"].to_numpy(float)[ej] * scale
    lowq = pdep.q_nivvie.isin(ad["equity_quintiles"]).to_numpy()
    w_low = (pdep["ind"].to_numpy(float) * lowq)[ej] * scale

    sel = pd.read_parquet(a.selected)
    col_of = pd.Series(np.arange(len(o)), index=o.fid.to_numpy())
    cand = (o.dep.astype(str).to_numpy() == a.dep) & (f < 1)
    recs = []
    for (strat, pct), g in sel.groupby(["strategy", "budget_pct"]):
        cols = col_of.reindex(g.fid.to_numpy()).dropna().astype(int).to_numpy()
        cols = cols[cand[cols]]
        fixed = np.zeros(len(ej), bool)
        if len(cols) and len(ej):
            for e0 in range(0, len(ej), 200_000):
                jj, tt = ej[e0:e0 + 200_000], et[e0:e0 + 200_000]
                sub = B[jj][:, cols].tocoo()
                ok = av_sh[cols[sub.col], tt[sub.row]] > 0
                fixed[e0 + np.unique(sub.row[ok])] = True
        recs.append({"strategy": strat, "budget_pct": pct, "n_selected_dep": len(cols),
                     "avoided_total": float(w_tot[fixed].sum()), "avoided_low_income": float(w_low[fixed].sum())})
    res = pd.DataFrame(recs)
    res["dep"] = a.dep
    res["heat_desert_total"] = float(w_tot.sum())
    res["heat_desert_low_income"] = float(w_low.sum())
    res.to_parquet(a.out, index=False)
    print(f"dep {a.dep}: heat-attributable events {len(ej)}, desert pop-hours {w_tot.sum():.0f}")


if __name__ == "__main__":
    main()
