#!/usr/bin/env python3
"""National shading sets per strategy and budget, from the département sequences.

Same ordering as adaptation_merge.py (marginal gain or ranking key, descending),
so the set of the first k facilities is exactly the set behind each point of the
in-sample curve. Output: strategy, budget_pct, k, fid (one row per selected facility).
"""
import argparse

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = pd.concat([pd.read_parquet(p, columns=["strategy", "rank_dep", "fid", "key", "dep", "n_candidates_dep"])
                    for p in a.inputs], ignore_index=True)
    n_cand = int(df.drop_duplicates("dep").n_candidates_dep.sum())
    rows = []
    for strat, g in df.groupby("strategy"):
        g = g.sort_values(["key", "dep", "rank_dep"], ascending=[False, True, True]).reset_index(drop=True)
        for pct in cfg["adaptation"]["budgets_pct"]:
            k = max(1, int(round(n_cand * pct / 100)))
            sel = g.head(k)
            rows.append(pd.DataFrame({"strategy": strat, "budget_pct": pct, "k": k, "fid": sel.fid.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(a.out, index=False)
    print(out.groupby(["strategy", "budget_pct"]).size().unstack())


if __name__ == "__main__":
    main()
