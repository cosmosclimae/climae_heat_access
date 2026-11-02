#!/usr/bin/env python3
"""Shading priorities for one département: a coverage problem on deserts.

Event e = (carreau j, sampled hour t) with no thermally available outdoor
facility in j's catchment (main tau0, public access). Shading candidate i
(outdoor, public, not covered, in the département) fixes e if i is in j's
catchment and its SHADE availability at t is 1 (f_i = 1, lighting unchanged).
Objective = population-hours of desert removed (weights P_j x season-hours per
sample). It is monotone submodular -> greedy with (1 - 1/e) guarantee.

Strategies (all evaluated on the same event set):
  coverage   : greedy on total population
  equity_<g> : greedy on group g weights (age groups, low living-standard quintiles)
  heat       : candidates ranked by mean sun WBGT over sampled hours
  population : candidates ranked by D_i (E2SFCA demand, total population)
For each pick we store the marginal gain for every objective, so national
curves can be merged exactly across départements (sum of independent parts).
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


def sequence_gains(order, Mc, Mr, Wt):
    """Marginal gains (all objectives) along a fixed candidate order."""
    covered = np.zeros(Mr.shape[0], dtype=bool)
    out = []
    for c in order:
        ev = Mc.indices[Mc.indptr[c]:Mc.indptr[c + 1]]
        new = ev[~covered[ev]]
        covered[new] = True
        out.append(Wt[new].sum(0))
    return np.array(out).reshape(len(order), Wt.shape[1])


def greedy(Mc, Mr, w, Wt, max_picks):
    # gains[c] = sum of w over events of column c
    col_of = np.repeat(np.arange(Mc.shape[1]), np.diff(Mc.indptr))
    gains = np.bincount(col_of, weights=w[Mc.indices], minlength=Mc.shape[1])
    covered = np.zeros(Mr.shape[0], dtype=bool)
    order, marg = [], []
    for _ in range(max_picks):
        c = int(np.argmax(gains))
        if gains[c] <= 0:
            break
        ev = Mc.indices[Mc.indptr[c]:Mc.indptr[c + 1]]
        new = ev[~covered[ev]]
        covered[new] = True
        marg.append(Wt[new].sum(0))
        order.append(c)
        # decrement gains of every candidate sharing the newly covered events
        if len(new):
            sub = Mr[new]
            gains -= np.bincount(sub.indices, weights=np.repeat(w[new], np.diff(sub.indptr)),
                                 minlength=len(gains))
        gains[c] = 0.0
    return order, np.array(marg).reshape(len(order), Wt.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dep", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--canopy", required=True)
    ap.add_argument("--facility-cells", required=True)
    ap.add_argument("--wbgt-cells", required=True)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--catchment-meta", required=True)
    ap.add_argument("--fac-access", required=True, help="facilities.parquet from the main accessibility run (D_i)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    q = cfg["climate"]["main_threshold"]
    scen = cfg["modifiers"]["main"]
    min_el = cfg["climate"]["lighting_min_elevation_deg"]
    ad = cfg["adaptation"]

    pop = pd.read_parquet(a.population)
    jdep = pop.index[pop.dep == a.dep].to_numpy()                       # jid == row index
    B = sp.load_npz(a.catchment).tocsr()[jdep]
    meta = pd.read_parquet(a.catchment_meta)
    o = outdoor_table(a.facilities, a.canopy, a.facility_cells, a.wbgt_cells, access="public")
    o = o.set_index("fid").loc[meta.fid.to_numpy()].reset_index()      # align with B columns
    used = np.unique(B.indices)
    f, cls, light_on = scenario_arrays(o, cfg, scen)

    z = np.load(a.sample)
    arr = {k: z[k] for k in z.files if k.startswith("wbgt_")}
    el = z["elev_mid"]
    k = o.k.to_numpy().astype(np.int64)
    T = el.shape[1]
    tall = np.arange(T)

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
    av_sh = np.zeros((B.shape[1], T), dtype=np.float32)
    av_sh[used] = availability(wb(used, np.ones(len(used))), q, light_ok)
    sun_mean = np.full(B.shape[1], np.nan)
    sun_mean[used] = np.nanmean(wb(used, np.zeros(len(used))), axis=1)

    cand = (o.dep.astype(str).to_numpy() == a.dep) & (f < 1)
    rows_ok = B.getnnz(1) > 0
    n_av = B @ av_now                                                   # Jdep x T
    desert = (n_av == 0) & rows_ok[:, None]
    ej, et = np.nonzero(desert)
    # heat-attributable deserts: at least one catchment facility is lit or in daylight,
    # so heat (not darkness) closes the last option. Shading can only act on these.
    lo = np.zeros((B.shape[1], T), dtype=np.float32)
    lo[used] = light_ok
    heat_attr = np.asarray((B @ lo)[ej, et] > 0).ravel()
    # incidence events x candidates : B[j, i] * av_sh[i, t] for candidate i
    Bc = B[:, np.where(cand)[0]].tocsr()
    cidx = np.where(cand)[0]
    rr, cc = [], []
    for e0 in range(0, len(ej), 200_000):
        jj, tt = ej[e0:e0 + 200_000], et[e0:e0 + 200_000]
        sub = Bc[jj].tocoo()
        keep = av_sh[cidx[sub.col], tt[sub.row]] > 0
        rr.append(sub.row[keep] + e0)
        cc.append(sub.col[keep])
    rr = np.concatenate(rr) if rr else np.array([], int)
    cc = np.concatenate(cc) if cc else np.array([], int)
    M = sp.csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(len(ej), len(cidx)))
    Mr, Mc = M.tocsr(), M.tocsc()

    scale = float(z["season_hours_per_sample"])
    pdep = pop.iloc[jdep]
    obj = {"total": pdep["ind"].to_numpy(float)}
    for g in ad["equity_groups"]:
        obj[g] = pdep[g].to_numpy(float)
    lowq = pdep.q_nivvie.isin(ad["equity_quintiles"]).to_numpy()
    obj["low_income"] = pdep["ind"].to_numpy(float) * lowq
    names = list(obj)
    Wt = np.stack([obj[n][ej] * scale for n in names], axis=1)          # events x objectives

    D = pd.read_parquet(a.fac_access).set_index("fid").D_ind
    Dc = D.reindex(o.fid.to_numpy()[cidx]).fillna(0).to_numpy()
    recs = []
    picks_cap = len(cidx)
    for strat, wname in [("coverage", "total")] + [(f"equity_{n}", n) for n in names if n != "total"]:
        order, marg = greedy(Mc, Mr, Wt[:, names.index(wname)], Wt, picks_cap)
        for r, (c, g) in enumerate(zip(order, marg)):
            recs.append({"strategy": strat, "rank_dep": r, "fid": int(o.fid.iat[cidx[c]]),
                         "key": float(g[names.index(wname)]), **{f"gain_{n}": float(g[i]) for i, n in enumerate(names)}})
    for strat, key in [("heat", sun_mean[cidx]), ("population", Dc)]:
        order = np.argsort(-np.nan_to_num(key, nan=-1e9))
        marg = sequence_gains(order, Mc, Mr, Wt)
        for r, (c, g) in enumerate(zip(order, marg)):
            recs.append({"strategy": strat, "rank_dep": r, "fid": int(o.fid.iat[cidx[c]]), "key": float(key[c]),
                         **{f"gain_{n}": float(g[i]) for i, n in enumerate(names)}})
    res = pd.DataFrame(recs)
    res["dep"] = a.dep
    res["n_candidates_dep"] = len(cidx)
    res.to_parquet(a.out, index=False)
    tot_desert = Wt.sum(0)
    print(f"dep {a.dep}: carreaux {len(jdep)}, events {len(ej)}, candidates {len(cidx)}, "
          f"desert pop-hours/season {dict(zip(names, np.round(tot_desert)))}")
    tot_heat = Wt[heat_attr].sum(0)
    pd.DataFrame([dict(zip(names, tot_desert)) | {f"{n}_heat": v for n, v in zip(names, tot_heat)}
                  | {"dep": a.dep, "events": len(ej), "events_heat": int(heat_attr.sum()),
                     "candidates": len(cidx)}]).to_parquet(
        Path(a.out).with_name(f"desert_total_{a.dep}.parquet"), index=False)


if __name__ == "__main__":
    main()
