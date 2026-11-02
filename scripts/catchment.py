#!/usr/bin/env python3
"""Merge per-département OD tables and build the binary desert catchment.

B[j, i] = 1 if outdoor facility i (access scenario `main`) is within the main
tau0 of carreau j. Columns follow the outdoor subset ordered by fid.
Outputs: od.parquet (all pairs <= max tau0), catchment_main.npz (scipy sparse),
catchment_meta.parquet (outdoor fid order).
"""
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from common import fid_map


def outdoor_subset(fac, access):
    o = fac[fac.layer == "outdoor"]
    if access == "public":
        o = o[o.access_public]
    return o.sort_values("fid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--od", nargs="+", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--od-out", required=True)
    ap.add_argument("--b-out", required=True)
    ap.add_argument("--meta-out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    od = pd.concat([pd.read_parquet(p) for p in a.od], ignore_index=True)
    od = od.sort_values("tau_min").drop_duplicates(["jid", "fid"]).sort_values(["jid", "fid"])
    od.to_parquet(a.od_out, index=False)
    nJ = len(pd.read_parquet(a.population, columns=["jid"]))
    fac = pd.read_parquet(a.facilities)
    sub = outdoor_subset(fac, cfg["access"]["main"])
    fm = fid_map(sub.fid.to_numpy(), int(fac.fid.max()) + 1)
    tau0 = cfg["network"]["tau0_min"][0]
    o = od[(od.layer == "outdoor") & (od.tau_min <= tau0)]
    col = fm[o.fid.to_numpy()]
    ok = col >= 0
    B = sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (o.jid.to_numpy()[ok], col[ok])), shape=(nJ, len(sub)))
    sp.save_npz(a.b_out, B)
    pd.DataFrame({"fid": sub.fid.to_numpy(), "col": np.arange(len(sub))}).to_parquet(a.meta_out, index=False)
    print(f"OD pairs {len(od)} ; B {B.shape} nnz={B.nnz} ; carreaux with >=1 facility: {(B.getnnz(1) > 0).sum()}")


if __name__ == "__main__":
    main()
