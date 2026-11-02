#!/usr/bin/env python3
"""Heat-only temporary deserts restricted to a subset of facility families (one year).

Same rules as availability_month.py (daily-hourly values, main tau0 catchment,
heat-only scenario = analysis.heat_scenario, main threshold), but only facilities
of the families in analysis.desert_subsets[<name>] count as an option. Cells with
no facility of the subset in their catchment are not "served" for this subset.
Outputs (one year): <out> jid, month, hour, n_desert ; <out>.ndays.parquet month, hour, n_days ;
<out>.served.parquet jid of cells with at least one subset facility in reach.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import outdoor_table, scenario_arrays, facility_wbgt, availability   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--wbgt", nargs="+", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--canopy", required=True)
    ap.add_argument("--facility-cells", required=True)
    ap.add_argument("--wbgt-cells", required=True)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--catchment-meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    fams = cfg["analysis"]["desert_subsets"][a.subset]
    hours = cfg["climate"]["local_start_hours"]
    hidx = {h: n for n, h in enumerate(hours)}
    q = cfg["climate"]["main_threshold"]
    scen = cfg["analysis"].get("heat_scenario", "nolight")
    min_el = cfg["climate"]["lighting_min_elevation_deg"]

    o = outdoor_table(a.facilities, a.canopy, a.facility_cells, a.wbgt_cells, access="all")
    B = sp.load_npz(a.catchment).tocsc()
    meta = pd.read_parquet(a.catchment_meta)
    pos = pd.Series(np.arange(len(o)), index=o.fid.to_numpy())
    bcols_all = pos.loc[meta.fid.to_numpy()].to_numpy()
    keep = o.family.isin(fams).to_numpy()[bcols_all]
    B = B[:, np.where(keep)[0]].tocsr()
    rows = bcols_all[keep]                                   # B column -> row in o
    rows_ok = B.getnnz(1) > 0
    sub = o.iloc[rows].reset_index(drop=True)
    k = sub.k.to_numpy().astype(np.int64)
    lit = sub.lit.to_numpy()
    f, cls, light_on = scenario_arrays(sub, cfg, scen)

    parts, nds = [], []
    for path in a.wbgt:
        ds = xr.open_dataset(path)
        arr = {v: ds[v].values.astype(np.float32) for v in ds.data_vars if v.startswith("wbgt_")}
        elev = ds["elev_mid"].values.astype(np.float32)
        ldate, lhour = ds["local_date"].values, ds["local_hour"].values
        month = int(re.search(r"(\d{4})-(\d{2})", Path(path).name).group(2))
        des = np.zeros((B.shape[0], len(hours)), dtype=np.uint16)
        nd = np.zeros(len(hours), dtype=np.uint16)
        for d in np.unique(ldate):
            t_sel = np.where(ldate == d)[0]
            hcol = np.array([hidx[h] for h in lhour[t_sel]])
            nd[hcol] += 1
            w = facility_wbgt(arr, k, f, cls, t_sel)
            el = elev[:, t_sel][k]
            light_ok = (lit[:, None] | (el >= min_el)) if light_on else np.ones_like(w, dtype=bool)
            av = availability(w, q, light_ok).astype(np.float32)
            z = ((B @ av) == 0) & rows_ok[:, None]
            des[:, hcol] += z.astype(np.uint16)
        j, h = np.nonzero(des)
        parts.append(pd.DataFrame({"jid": j.astype(np.int32), "month": np.int8(month),
                                   "hour": np.array(hours)[h].astype(np.int8), "n_desert": des[j, h]}))
        nds.append(pd.DataFrame({"month": np.int8(month), "hour": np.array(hours).astype(np.int8), "n_days": nd}))
        print(f"{path}: {int(rows_ok.sum())} served cells, {len(j)} desert cell-hours")
    out = pd.concat(parts, ignore_index=True)
    served = pd.DataFrame({"jid": np.where(rows_ok)[0].astype(np.int32)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    served.to_parquet(Path(a.out).with_suffix(".served.parquet"), index=False)
    pd.concat(nds, ignore_index=True).to_parquet(Path(a.out).with_suffix(".ndays.parquet"), index=False)


if __name__ == "__main__":
    main()
