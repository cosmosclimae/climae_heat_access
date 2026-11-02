#!/usr/bin/env python3
"""Sensitivity: heat-modulated 3SFCA (after Wan, Zou & Sternberg 2012) on metropolitan areas.

Selection weight with thermal availability (climatological, per month x hour):
  G_ji = abar_i W_ji / sum_r abar_r W_jr
  D*_i = sum_j G_ji W_ji P_j ;  R_i = C_i abar_i / D*_i ;  A_j = sum_i G_ji W_ji R_i
Baseline (abar = 1) gives A0; L = 1 - A_heat/A0 ; B = mean over (m, h).
Unlike E2SFCA, demand is redistributed towards available facilities, which
breaks the conservation property. Compared with the E2SFCA main run on the
same carreaux: Spearman of B, shares of population with B > 0.25, by age group.
"""
import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from pyproj import Transformer
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_W, fid_map      # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")

METROS = {"Paris": (2.349, 48.857), "Marseille": (5.370, 43.296), "Lyon": (4.835, 45.764),
          "Toulouse": (1.444, 43.605), "Montpellier": (3.877, 43.611), "Bordeaux": (-0.579, 44.838),
          "Lille": (3.057, 50.629), "Strasbourg": (7.752, 48.573)}


def three_step(W, P, C, abar):
    """W: J x I csr (gaussian weights), P: J, C: I, abar: I x T -> A: J x T."""
    out = np.zeros((W.shape[0], abar.shape[1]))
    Wc = W.tocsr()
    rows = np.repeat(np.arange(W.shape[0]), np.diff(Wc.indptr))
    cols, w = Wc.indices, Wc.data
    for t in range(abar.shape[1]):
        aw = abar[cols, t] * w
        den = np.bincount(rows, weights=aw, minlength=W.shape[0])
        G = np.divide(aw, den[rows], out=np.zeros_like(aw), where=den[rows] > 0)
        Dst = np.bincount(cols, weights=G * w * P[rows], minlength=W.shape[1])
        R = np.divide(C * abar[:, t], Dst, out=np.zeros_like(C), where=Dst > 0)
        out[:, t] = np.bincount(rows, weights=G * w * R[cols], minlength=W.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--od", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--clim", required=True)
    ap.add_argument("--access-main", required=True)
    ap.add_argument("--radius-km", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    tau0, w0 = cfg["network"]["tau0_min"][0], cfg["network"]["w_at_tau0"]
    q, scen = cfg["climate"]["main_threshold"], cfg["modifiers"]["main"]
    groups = cfg["population"]["groups"]
    pop = pd.read_parquet(a.population)
    fac = pd.read_parquet(a.facilities)
    o = fac[(fac.layer == "outdoor") & fac.access_public].sort_values("fid")
    z = np.load(a.clim)
    s_i, q_i = list(z["scens"]).index(scen), list(z["qs"]).index(q)
    pos = pd.Series(np.arange(len(z["fid"])), index=z["fid"])
    abar = np.nan_to_num(z["abar"][s_i, q_i][pos.loc[o.fid.to_numpy()].to_numpy()].reshape(len(o), -1), nan=1.0)
    od = pd.read_parquet(a.od)
    od = od[od.layer == "outdoor"]
    main = pd.read_parquet(a.access_main)
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    rows = []
    for city, (lo, la) in METROS.items():
        cx, cy = tr.transform(lo, la)
        sel = np.hypot(pop.x3035 - cx, pop.y3035 - cy) <= a.radius_km * 1000
        jids = pop.jid[sel].to_numpy()
        if len(jids) == 0:
            continue
        sub = od[od.jid.isin(jids)]
        fids = np.unique(sub.fid)
        fm = fid_map(o.fid.to_numpy(), int(fac.fid.max()) + 1)
        cols = fm[fids]
        fids, cols = fids[cols >= 0], cols[cols >= 0]
        jm = pd.Series(np.arange(len(jids)), index=jids)
        sub = sub[sub.fid.isin(fids)].assign(jid=lambda d: jm.loc[d.jid].to_numpy())
        W = build_W(sub, len(jids), len(fids), tau0, w0, fid_map(fids, int(fac.fid.max()) + 1))
        C = np.ones(len(fids))
        ab = abar[cols]
        for g in groups:
            P = pop.set_index("jid").loc[jids, g].to_numpy(float)
            A0 = three_step(W, P, C, np.ones((len(fids), 1)))[:, 0]
            Ah = three_step(W, P, C, ab)
            with np.errstate(invalid="ignore", divide="ignore"):
                L = np.where(A0[:, None] > 0, 1 - Ah / A0[:, None], np.nan)
            B3 = np.nanmean(L, axis=1)
            Be = main.set_index("jid").loc[jids, f"B_{g}"].to_numpy()
            ok = np.isfinite(B3) & np.isfinite(Be)
            rows.append({"city": city, "group": g, "carreaux": int(ok.sum()), "population": float(P[ok].sum()),
                         "spearman_B_3sfca_vs_e2sfca": float(spearmanr(B3[ok], Be[ok]).statistic) if ok.sum() > 2 else np.nan,
                         "share_B_gt_025_3sfca": float(P[ok][B3[ok] > .25].sum() / max(P[ok].sum(), 1)),
                         "share_B_gt_025_e2sfca": float(P[ok][Be[ok] > .25].sum() / max(P[ok].sum(), 1)),
                         "B_median_3sfca": float(np.nanmedian(B3[ok])) if ok.any() else np.nan,
                         "B_median_e2sfca": float(np.nanmedian(Be[ok])) if ok.any() else np.nan})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).query("group == 'ind'").round(3).to_string())


if __name__ == "__main__":
    main()
