#!/usr/bin/env python3
"""National out-of-sample shading results vs in-sample table.

Inputs: per-département evaluation files for each evaluation sample (tag=path
list), and the in-sample strategies_budgets.csv. Output: one row per strategy and
budget with the share of heat-attributable desert person-hours removed in
sample and in each independent sample (total population and low-income).
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-sample", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="tag=file1,file2,... per evaluation sample")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ins = pd.read_csv(a.in_sample)
    out = ins[["strategy", "budget_pct", "k", "avoided_share_heat_total", "avoided_share_heat_low_income"]].rename(
        columns={"avoided_share_heat_total": "share_in_sample", "avoided_share_heat_low_income": "share_low_in_sample"})
    for spec in a.eval:
        tag, files = spec.split("=", 1)
        d = pd.concat([pd.read_parquet(p) for p in files.split(",")], ignore_index=True)
        tot = d.drop_duplicates("dep")[["heat_desert_total", "heat_desert_low_income"]].sum()
        g = d.groupby(["strategy", "budget_pct"])[["avoided_total", "avoided_low_income"]].sum().reset_index()
        g[f"share_{tag}"] = g.avoided_total / tot.heat_desert_total
        g[f"share_low_{tag}"] = g.avoided_low_income / tot.heat_desert_low_income
        out = out.merge(g[["strategy", "budget_pct", f"share_{tag}", f"share_low_{tag}"]],
                        on=["strategy", "budget_pct"], how="left")
        print(f"{tag}: heat-attributable desert person-hours/season {tot.heat_desert_total:.3e}")
    out = out.sort_values(["budget_pct", "share_in_sample"], ascending=[True, False])
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
