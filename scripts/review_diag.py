#!/usr/bin/env python3
"""Diagnostics requested in review (round 2).

1. Conservation: I_g = public outdoor facilities with positive demand of group g
   (at least one resident of g within tau0). Counts facilities with D_ind > 0 but
   D_g = 0, and the aggregate peak loss 1 - mean_{I_g}(abar_peak) per group
   (C_i = 1), which the E2SFCA identity makes equal across groups only if I_g is shared.
2. Département merge of the greedy: share of served cells (and residents) whose
   catchment contains a facility of another département.
3. Shapley decomposition of the Q5-Q1 desert-hour gap over three blocks of supply
   terms (log number of facilities; mean peak availability; composition), with
   region FE and with ERA5-Land cell FE (weighted least squares, within transform).
Outputs: <outdir>/diag_conservation.csv, diag_crossdep.csv, diag_shapley.csv
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def fe_gap(df, fe, cols, y="dh"):
    """Weighted within-FE OLS; returns the Q5 vs Q1 coefficient."""
    X = pd.get_dummies(df.q_nivvie.astype(int), prefix="q", drop_first=True).astype(float)
    if cols:
        X = pd.concat([X, df[cols].astype(float)], axis=1)
    w = df.ind.to_numpy(float)
    gi = pd.factorize(df[fe].astype(str).to_numpy())[0]
    ng = gi.max() + 1
    sw = np.bincount(gi, w, ng)

    def dm(v):
        return v - (np.bincount(gi, w * v, ng) / sw)[gi]
    Xd = np.column_stack([dm(X[c].to_numpy()) for c in X.columns])
    yd = dm(df[y].to_numpy(float))
    sq = np.sqrt(w)
    b, *_ = np.linalg.lstsq(Xd * sq[:, None], yd * sq, rcond=None)
    return float(b[list(X.columns).index("q_5")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--access", required=True)
    ap.add_argument("--od", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--fac-heat", required=True)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--catchment-meta", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = cfg["network"]["tau0_min"][0]
    groups = cfg["population"]["groups"]

    pop = pd.read_parquet(a.population, columns=["jid", "dep", "q_nivvie", *groups])
    fac = pd.read_parquet(a.facilities, columns=["fid", "dep", "layer", "access_public", "family", "surface_class"])
    pub = fac[(fac.layer == "outdoor") & fac.access_public]
    od = pd.read_parquet(a.od, columns=["jid", "fid", "layer", "tau_min"])
    od = od[(od.layer == "outdoor") & (od.tau_min <= t0) & od.fid.isin(pub.fid)]
    fh = pd.read_parquet(a.fac_heat, columns=["fid", "abar_peak"]).set_index("fid").abar_peak

    # 1. conservation sets
    odp = od.merge(pop[["jid", *groups]], on="jid")
    rows = []
    has = {g: set(odp.loc[odp[g] > 0, "fid"]) for g in groups}
    for g in groups:
        I = np.array(sorted(has[g]))
        rows.append({"group": g, "facilities_positive_demand": len(I),
                     "missing_vs_ind": len(has["ind"] - has[g]),
                     "aggregate_peak_loss": float(1 - fh.reindex(I).mean())})
    pd.DataFrame(rows).to_csv(out / "diag_conservation.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. cross-département catchments
    B = sp.load_npz(a.catchment).tocoo()
    meta = pd.read_parquet(a.catchment_meta)
    fdep = fac.set_index("fid").dep.astype(str).reindex(meta.fid.to_numpy()).to_numpy()
    pdep = pop.dep.astype(str).to_numpy()
    cross = np.zeros(len(pop), bool)
    m = fdep[B.col] != pdep[B.row]
    cross[np.unique(B.row[m])] = True
    served = np.bincount(B.row, minlength=len(pop)) > 0
    w = pop["ind"].to_numpy(float)
    cd = pd.DataFrame([{"served_cells": int(served.sum()), "cross_cells": int((cross & served).sum()),
                        "share_cells": float((cross & served).sum() / served.sum()),
                        "share_residents": float(w[cross & served].sum() / w[served].sum())}])
    cd.to_csv(out / "diag_crossdep.csv", index=False)
    print(cd.to_string(index=False))

    # 3. Shapley decomposition of the Q5-Q1 gap
    od2 = od.merge(fh.rename("abar_peak").reset_index(), on="fid").merge(
        pub[["fid", "family", "surface_class"]], on="fid")
    od2["tennis"] = od2.family.eq("Court de tennis")
    od2["citystade"] = od2.family.eq("Multisports/City-stades")
    od2["boules"] = od2.family.eq("Boulodrome")
    od2["syn_min"] = od2.surface_class.isin(["synthetic", "mineral"])
    sup = od2.groupby("jid").agg(n_fac=("fid", "size"), abar_peak_mean=("abar_peak", "mean"),
                                 tennis=("tennis", "mean"), citystade=("citystade", "mean"),
                                 boules=("boules", "mean"), syn_min=("syn_min", "mean"))
    met = pd.read_parquet(a.metrics, columns=["jid", "region", "era5_cell", "dh_heat_ref"])
    acc = pd.read_parquet(a.access, columns=["jid", "has_outdoor"])
    d = met.merge(pop[["jid", "q_nivvie", "ind"]], on="jid").merge(acc, on="jid").merge(
        sup, left_on="jid", right_index=True)
    d = d[d.has_outdoor & (d.ind > 0) & d.q_nivvie.between(1, 5)].copy()
    d["dh"] = d.dh_heat_ref.fillna(0.0)
    d["log_n"] = np.log(d.n_fac)
    blocks = {"redundancy (log n)": ["log_n"], "peak availability": ["abar_peak_mean"],
              "composition": ["tennis", "citystade", "boules", "syn_min"]}
    names = list(blocks)
    srows = []
    for fe in ["region", "era5_cell"]:
        val = {}
        for r in range(len(names) + 1):
            for S in itertools.combinations(names, r):
                val[S] = fe_gap(d, fe, [c for n in S for c in blocks[n]])
        base, full = val[()], val[tuple(names)]
        rec = {"fe": fe, "gap_no_supply": base, "gap_all_supply": full}
        n = len(names)
        for nm in names:
            phi = 0.0
            others = [x for x in names if x != nm]
            for r in range(n):
                for S in itertools.combinations(others, r):
                    Sw = tuple(x for x in names if x in S)
                    Sp = tuple(x for x in names if x in S or x == nm)
                    wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                    phi += wgt * (val[Sw] - val[Sp])          # reduction of the gap due to block nm
            rec[f"shapley_{nm}"] = phi
            rec[f"share_{nm}"] = phi / (base - full) if base != full else np.nan
        srows.append(rec)
    sh = pd.DataFrame(srows)
    sh.to_csv(out / "diag_shapley.csv", index=False)
    print(sh.round(2).T.to_string())


if __name__ == "__main__":
    main()
