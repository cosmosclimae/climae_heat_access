#!/usr/bin/env python3
"""Baseline and heat-constrained E2SFCA, losses, indoor fallback.

Per population group g (same Gaussian weights at both steps):
  D_ig = sum_k P_kg w_ki ;  A0_jg = sum_i C_i/D_ig w_ji
  Aheat_jg(m,h) = sum_i C_i abar_i(m,h)/D_ig w_ji
  L = 1 - Aheat/A0 (A0 > 0) ; B = mean_{m,h} L ; Fin = indoor E2SFCA.
Conservation (written in Methods): sum_j P_jg Aheat_jg = sum_i C_i abar_i for all g,
so equity is analysed on distributions, never on population-weighted means of A.

Outputs
  access.parquet   : per carreau, per group: A0, Aheat_mean, B, L_jul_17, Lpeak (peak window), Fin ; flags
  L_ind.npy        : float16 J x (M*H) loss for the total population (main runs)
  facilities.parquet : D_i (total population), C_i, abar_jul_17
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_W, fid_map      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--od", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--clim", required=True)
    ap.add_argument("--scen", required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--tau0", type=float, required=True)
    ap.add_argument("--access", required=True)
    ap.add_argument("--capacity", default="unit")
    ap.add_argument("--avail", default="binary", choices=["binary", "graded"])
    ap.add_argument("--save-L", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fac-out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    groups = cfg["population"]["groups"]
    w0 = cfg["network"]["w_at_tau0"]

    pop = pd.read_parquet(a.population)
    nJ = len(pop)
    fac = pd.read_parquet(a.facilities)
    nF = int(fac.fid.max()) + 1
    od = pd.read_parquet(a.od)
    z = np.load(a.clim)
    s_idx = list(z["scens"]).index(a.scen)
    q_idx = list(z["qs"]).index(a.q)
    abar_all = z["abar_graded"][q_idx] if a.avail == "graded" else z["abar"][s_idx, q_idx]   # I x M x H
    months, hours = list(z["months"]), list(z["hours"])
    M, H = len(months), len(hours)
    clim_pos = pd.Series(np.arange(len(z["fid"])), index=z["fid"])

    out_f = fac[fac.layer == "outdoor"]
    if a.access == "public":
        out_f = out_f[out_f.access_public]
    out_f = out_f.sort_values("fid")
    C = out_f["capacity_surface" if a.capacity == "surface" else "capacity_unit"].to_numpy(float)
    abar = abar_all[clim_pos.loc[out_f.fid.to_numpy()].to_numpy()].reshape(len(out_f), M * H)
    abar = np.nan_to_num(abar, nan=1.0)
    W = build_W(od[od.layer == "outdoor"], nJ, len(out_f), a.tau0, w0, fid_map(out_f.fid.to_numpy(), nF))
    in_f = fac[fac.layer == "indoor"]
    if a.access == "public":
        in_f = in_f[in_f.access_public]
    in_f = in_f.sort_values("fid")
    Cin = in_f["capacity_surface" if a.capacity == "surface" else "capacity_unit"].to_numpy(float)
    Win = build_W(od[od.layer == "indoor"], nJ, len(in_f), a.tau0, w0, fid_map(in_f.fid.to_numpy(), nF))
    W, Win = W.astype(np.float32), Win.astype(np.float32)
    abar = abar.astype(np.float32)
    WT, WinT = W.T.tocsr(), Win.T.tocsr()

    res = pd.DataFrame({"jid": pop.jid.to_numpy()})
    jul = months.index(7) * H + hours.index(17) if (7 in months and 17 in hours) else 0
    pw = cfg["climate"].get("peak_window", {"month": 7, "hours": [14, 15, 16]})
    peak = [months.index(pw["month"]) * H + hours.index(h) for h in pw["hours"]
            if pw["month"] in months and h in hours]
    D_tot = None
    cons = {}
    for g in groups:
        P = pop[g].to_numpy(float)
        D = WT @ P
        R0 = np.divide(C, D, out=np.zeros_like(C), where=D > 0)
        A0 = W @ R0
        Ah = W @ (R0.astype(np.float32)[:, None] * abar)                         # J x (M*H)
        with np.errstate(invalid="ignore", divide="ignore"):
            L = np.where(A0[:, None] > 0, 1.0 - Ah / A0[:, None], np.nan)
        Din = WinT @ P
        Rin = np.divide(Cin, Din, out=np.zeros_like(Cin), where=Din > 0)
        res[f"A0_{g}"] = A0
        res[f"Aheat_{g}"] = Ah.mean(axis=1)
        res[f"B_{g}"] = np.nanmean(L, axis=1) if L.size else np.nan
        res[f"Ljul17_{g}"] = L[:, jul]
        res[f"Lpeak_{g}"] = np.nanmean(L[:, peak], axis=1) if peak else np.nan
        res[f"Fin_{g}"] = Win @ Rin
        # conservation check: sum_j P_j Aheat_j == sum_i C_i abar_i (over served facilities)
        served = D > 0
        cons[g] = float(np.abs((P @ Ah).sum() - (C[served, None] * abar[served]).sum()) /
                        max((C[served, None] * abar[served]).sum(), 1e-9))
        if g == "ind":
            D_tot = D
            if a.save_L:
                np.save(Path(a.out).with_name("L_ind.npy"), L.astype(np.float16))
    res["has_outdoor"] = res["A0_ind"] > 0
    res.to_parquet(a.out, index=False)
    pd.DataFrame({"fid": out_f.fid.to_numpy(), "C": C, "D_ind": D_tot, "abar_mean": abar.mean(1),
                  "abar_jul17": abar[:, jul],
                  "abar_peak": abar[:, peak].mean(1) if peak else np.nan}).to_parquet(a.fac_out, index=False)
    print(f"{a.out}: J={nJ} I_out={len(out_f)} I_in={len(in_f)} ; conservation rel. error {cons}")


if __name__ == "__main__":
    main()
