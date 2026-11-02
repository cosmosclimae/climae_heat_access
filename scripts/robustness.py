#!/usr/bin/env python3
"""Robustness of the desert results beyond L_peak (reviewer points 3 and 5).

For each desert variant (scenario x threshold, plus facility subsets):
  desert hours per served resident and season, share above 100 h,
  income gaps with Q1 (weighted least squares) with region FE and ERA5-Land cell
  FE, 95% intervals from a cluster bootstrap on ERA5-Land cells.
Also the same gaps on native 200-m cells only (main variant).
Output CSV: variant, served_pop_M, desert_h, share_gt100, fe, Q2..Q5 (est, lo, hi).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DIM = {5: 31, 6: 30, 7: 31, 8: 31, 9: 30}


def desert_hours(des, nd=None):
    d = des.copy()
    if nd is not None:
        d = d.merge(nd, on=["month", "hour"], how="left")
    h = d.n_desert.to_numpy(float) / d.n_days.to_numpy(float) * d.month.map(DIM).to_numpy(float)
    return pd.Series(h).groupby(d.jid.to_numpy()).sum()


def gaps(df, fe, y, B, seed):
    X = pd.get_dummies(df.q_nivvie.astype(int), prefix="q", drop_first=True).astype(float)
    cols = list(X.columns)
    w = df.ind.to_numpy(float)
    gi = pd.factorize(df[fe].astype(str).to_numpy())[0]
    ng = gi.max() + 1
    sw = np.bincount(gi, w, ng)

    def dm(v):
        return v - (np.bincount(gi, w * v, ng) / sw)[gi]
    Xd = np.column_stack([dm(X[c].to_numpy()) for c in cols])
    yd = dm(df[y].to_numpy(float))
    cl = pd.factorize(df.era5_cell.to_numpy())[0]
    nc = cl.max() + 1
    kx = Xd.shape[1]
    XtX = np.zeros((nc, kx, kx))
    Xty = np.zeros((nc, kx))
    Xw = Xd * w[:, None]
    for j in range(kx):
        Xty[:, j] = np.bincount(cl, Xw[:, j] * yd, nc)
        for l in range(j, kx):
            v = np.bincount(cl, Xw[:, j] * Xd[:, l], nc)
            XtX[:, j, l] = v
            XtX[:, l, j] = v
    b = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc)
        bs.append(np.linalg.solve(np.tensordot(cnt, XtX, 1), cnt @ Xty))
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    return {c: (b[i], lo[i], hi[i]) for i, c in enumerate(cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True, help="results/analysis/carreaux_metrics.parquet")
    ap.add_argument("--population", required=True)
    ap.add_argument("--access", required=True, help="access/main/access.parquet")
    ap.add_argument("--deserts-dir", required=True, help="clim/deserts_ref")
    ap.add_argument("--subsets", nargs="*", default=[], help="name=deserts.parquet,ndays.parquet,served.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    B = cfg.get("bootstrap_n", 500)
    B = min(B, 200)
    seed = cfg.get("seed", 1)
    q0 = cfg["climate"]["main_threshold"]
    hs = cfg["analysis"].get("heat_scenario", "nolight")

    m = pd.read_parquet(a.metrics, columns=["jid", "region", "era5_cell"])
    p = pd.read_parquet(a.population, columns=["jid", "q_nivvie", "ind", "native200"])
    acc = pd.read_parquet(a.access, columns=["jid", "has_outdoor"])
    base = m.merge(p, on="jid").merge(acc, on="jid")
    base = base[(base.ind > 0) & base.q_nivvie.between(1, 5)]

    variants = []
    for q in cfg["climate"]["thresholds"]:
        variants.append((f"heat-only q{q}", Path(a.deserts_dir) / f"deserts_{hs}_q{q}.parquet", None, None))
    for s in ["full", "nosurf", "canopy50"]:
        variants.append((f"combined {s} q{q0}", Path(a.deserts_dir) / f"deserts_{s}_q{q0}.parquet", None, None))
    for spec in a.subsets:
        name, files = spec.split("=", 1)
        dfile, nfile, sfile = files.split(",")
        variants.append((f"heat-only q{q0}, {name} only", Path(dfile), Path(nfile), Path(sfile)))

    rows = []
    for name, dpath, npath, spath in variants:
        if not dpath.exists():
            print(f"skip {name}: {dpath} missing")
            continue
        des = pd.read_parquet(dpath)
        if npath is not None:
            nd = pd.read_parquet(npath).groupby(["month", "hour"], as_index=False).n_days.sum()
            des = des.groupby(["jid", "month", "hour"], as_index=False).n_desert.sum()
            dh = desert_hours(des, nd)
            served = set(pd.read_parquet(spath).jid)
            d = base[base.jid.isin(served)].copy()
        else:
            dh = desert_hours(des)
            d = base[base.has_outdoor].copy()
        d["dh"] = d.jid.map(dh).fillna(0.0)
        w = d.ind.to_numpy(float)
        rec = {"variant": name, "served_pop_M": w.sum() / 1e6,
               "desert_h": float(np.average(d.dh, weights=w)),
               "share_gt100": float(w[d.dh.to_numpy() > 100].sum() / w.sum())}
        for fe in ["region", "era5_cell"]:
            g = gaps(d, fe, "dh", B, seed)
            r = dict(rec, fe=fe)
            for c, (e, lo, hi) in g.items():
                r[f"{c}"], r[f"{c}_lo"], r[f"{c}_hi"] = e, lo, hi
            rows.append(r)
        if name == f"heat-only q{q0}":
            dn = d[d.native200]
            g = gaps(dn, "era5_cell", "dh", B, seed)
            r = {"variant": name + ", native 200-m cells", "served_pop_M": dn.ind.sum() / 1e6,
                 "desert_h": float(np.average(dn.dh, weights=dn.ind)),
                 "share_gt100": float(dn.ind[dn.dh > 100].sum() / dn.ind.sum()), "fe": "era5_cell"}
            for c, (e, lo, hi) in g.items():
                r[f"{c}"], r[f"{c}_lo"], r[f"{c}_hi"] = e, lo, hi
            rows.append(r)
        print(name, "done")
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
