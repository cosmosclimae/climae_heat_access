#!/usr/bin/env python3
"""National shading curves from département results (exact merge).

coverage / equity_*: greedy sequences per département have non-increasing
marginal gains; merging them by gain reproduces the national greedy.
heat / population: national ranking by the same key; each département's
prefix follows its own ranking, so gains add up.
Outputs: curve (strategy, k, cumulative gains), table at budgets (% of national
candidates), and figure.
"""
import argparse

import numpy as np
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--totals", nargs="+", required=True)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = pd.concat([pd.read_parquet(p) for p in a.inputs], ignore_index=True)
    tot = pd.concat([pd.read_parquet(p) for p in a.totals], ignore_index=True)
    gcols = [c for c in df.columns if c.startswith("gain_")]
    n_cand = int(df.drop_duplicates("dep").n_candidates_dep.sum())
    curves = []
    for strat, g in df.groupby("strategy"):
        g = g.sort_values(["key", "dep", "rank_dep"], ascending=[False, True, True])
        c = g[gcols].cumsum().reset_index(drop=True)
        c.insert(0, "k", np.arange(1, len(c) + 1))
        c.insert(0, "strategy", strat)
        curves.append(c)
    cur = pd.concat(curves, ignore_index=True)
    cur.to_parquet(a.curve, index=False)
    totals = tot[[c.replace("gain_", "") for c in gcols]].sum()
    rows = []
    for pct in cfg["adaptation"]["budgets_pct"]:
        k = max(1, int(round(n_cand * pct / 100)))
        for strat, c in cur.groupby("strategy"):
            r = c[c.k <= k].tail(1)
            if len(r) == 0:
                continue
            rec = {"budget_pct": pct, "k": k, "strategy": strat}
            for gc in gcols:
                name = gc.replace("gain_", "")
                rec[f"avoided_{name}"] = float(r[gc].iat[0])
                rec[f"avoided_share_{name}"] = float(r[gc].iat[0] / totals[name]) if totals[name] > 0 else np.nan
                hn = f"{name}_heat"
                if hn in tot:
                    th = float(tot[hn].sum())
                    rec[f"avoided_share_heat_{name}"] = float(r[gc].iat[0] / th) if th > 0 else np.nan
            rows.append(rec)
    pd.DataFrame(rows).to_csv(a.table, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    den = float(tot["total_heat"].sum()) if "total_heat" in tot else totals["total"]
    lab = {"coverage": "Max. coverage (greedy)", "heat": "Hottest facilities first",
           "population": "Population served first", "equity_low_income": "Low-income weighted (Q1-Q2)"}
    for i, strat in enumerate(["coverage", "equity_low_income", "population", "heat"]):
        c = cur[cur.strategy == strat]
        if len(c):
            ax.plot(c.k / n_cand * 100, c.gain_total / den * 100, label=lab[strat],
                    lw=2 if strat == "coverage" else 1.2, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i])
    ax.set_xlim(0, max(cfg["adaptation"]["budgets_pct"]) * 1.2)
    ax.set_xlabel("Facilities shaded (% of candidates)")
    ax.set_ylabel("Heat-attributable desert person-hours avoided (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=200)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
