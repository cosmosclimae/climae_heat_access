#!/usr/bin/env python3
"""Step 0 go/no-go: how often is the WBGT threshold exceeded, where and when?

Exceedance frequency P(WBGT >= q) per region x month x local hour, for the sun
(reference ground and each surface class) and shade WBGT, on the cells hosting
outdoor facilities, weighted by the number of outdoor facilities per cell.
Writes a long CSV, a July heat-map figure and a short YAML verdict:
regions where July 14-17 h sun-reference exceedance of the main threshold > 5 %.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wbgt", nargs="+", required=True)
    ap.add_argument("--wbgt-cells", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--facility-cells", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--verdict", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    qs = cfg["climate"]["thresholds"]
    qmain = cfg["climate"]["main_threshold"]
    months = cfg["climate"]["months"]
    hours = cfg["climate"]["local_start_hours"]
    wc = pd.read_parquet(a.wbgt_cells).sort_values("k")
    fac = pd.read_parquet(a.facilities)
    fc = pd.read_parquet(a.facility_cells)
    o = fac[fac.layer == "outdoor"][["fid", "dep", "region"]].merge(fc[["fid", "cell"]], on="fid")
    # 'Région Nom' is sometimes empty in Data ES: fill it from the département
    dep2reg = (fac.dropna(subset=["region"]).groupby("dep").region
               .agg(lambda s: s.value_counts().index[0]))
    o["region"] = o["region"].fillna(o["dep"].map(dep2reg)).fillna("Inconnue")
    nfac = o.groupby("cell").size()
    region_of_cell = o.groupby("cell").region.agg(lambda s: s.value_counts().index[0])
    wc["n_fac"] = wc.cell.map(nfac).fillna(0).to_numpy()
    wc["region"] = wc.cell.map(region_of_cell).to_numpy()

    acc, tot = {}, {}
    for p in a.wbgt:
        mon = int(re.search(r"(\d{4})-(\d{2})", Path(p).name).group(2))
        ds = xr.open_dataset(p)
        lh = ds["local_hour"].values
        for v in [x for x in ds.data_vars if x.startswith("wbgt_")]:
            arr = ds[v].values
            for h in hours:
                sel = arr[:, lh == h]
                n = np.isfinite(sel).sum(1)
                tot[(v, mon, h)] = tot.get((v, mon, h), 0) + n
                for q in qs:
                    key = (v, q, mon, h)
                    acc[key] = acc.get(key, 0) + (sel >= q).sum(1)
    rows = []
    for (v, q, mon, h), cnt in acc.items():
        n = tot[(v, mon, h)]
        f = np.where(n > 0, cnt / np.maximum(n, 1), np.nan)
        df = pd.DataFrame({"region": wc.region, "w": wc.n_fac, "f": f})
        df = df[df.w > 0]
        g = df.groupby("region").apply(lambda d: np.average(d.f, weights=d.w), include_groups=False)
        for r, val in g.items():
            rows.append((v, q, mon, h, r, float(val)))
        rows.append((v, q, mon, h, "France", float(np.average(df.f, weights=df.w))))
    res = pd.DataFrame(rows, columns=["var", "q", "month", "hour", "region", "p_exceed"])
    res.to_csv(a.csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_month = 7 if (res.month == 7).any() else int(res.month.max())
    sub = res[(res["var"] == "wbgt_sun_reference") & (res.q == qmain) & (res.month == fig_month)]
    piv = sub.pivot(index="region", columns="hour", values="p_exceed")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(piv) + 1.5))
    im = ax.imshow(piv.to_numpy() * 100, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(len(piv.columns)), [f"{h}h" for h in piv.columns])
    ax.set_title(f"P(WBGT sun ≥ {qmain} °C), month {fig_month}, local hour (interval start)")
    fig.colorbar(im, ax=ax, label="% of hours")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=200)

    aft = res[(res["var"] == "wbgt_sun_reference") & (res.q == qmain) & (res.month == fig_month) & res.hour.between(14, 17)]
    by_r = aft.groupby("region").p_exceed.mean().sort_values(ascending=False)
    verdict = {"threshold": qmain, "metric": f"mean P(sun WBGT >= q), month {fig_month}, 14-17 h local",
               "by_region": {k: round(v, 4) for k, v in by_r.items()},
               "regions_above_5pct": [k for k, v in by_r.items() if v > 0.05 and k != "France"],
               "rule": "if few regions exceed 5 %, lower the central threshold to 26 degC or frame the paper on the south"}
    with open(a.verdict, "w", encoding="utf-8") as f:
        yaml.safe_dump(verdict, f, allow_unicode=True, sort_keys=False)
    print(verdict["regions_above_5pct"])


if __name__ == "__main__":
    main()
