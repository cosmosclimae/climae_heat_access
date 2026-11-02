#!/usr/bin/env python3
"""Sum monthly availability files over a period -> climatological availability.

abar[s,q,i,m,h] = available days / days   (per calendar month m, local hour h)
abar_graded[q,i,m,h]  (main scenario)
deserts_<s>_q<q>.parquet : jid, month, hour, n_desert, n_days  (F = n_desert/n_days)

Memory-bounded: availability counts in one pass; deserts accumulated one
(scenario, threshold) at a time in a dense uint16 array restricted to the
carreaux that ever appear in a desert (npz keys are decompressed lazily).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def month_of(p):
    return int(re.search(r"(\d{4})-(\d{2})", Path(p).name).group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--months", nargs="+", type=int, required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    M = {m: n for n, m in enumerate(a.months)}

    # ---- pass 1: availability counts --------------------------------------
    acc = accg = nd = None
    for p in a.inputs:
        mi = M[month_of(p)]
        with np.load(p, allow_pickle=False) as z:
            if acc is None:
                scens, qs, hours, fid = z["scens"], z["qs"], z["hours"], z["fid"]
                S, Q, I, H = z["counts"].shape
                acc = np.zeros((S, Q, I, len(M), H), dtype=np.float32)
                accg = np.zeros((Q, I, len(M), H), dtype=np.float32)
                nd = np.zeros((len(M), H), dtype=np.float32)
            acc[..., mi, :] += z["counts"]
            accg[:, :, mi, :] += z["graded"]
            nd[mi] += z["ndays"]
    nd_safe = np.where(nd > 0, nd, np.nan)
    np.savez_compressed(a.out_npz, abar=(acc / nd_safe[None, None, None]).astype(np.float32),
                        abar_graded=(accg / nd_safe[None, None]).astype(np.float32), fid=fid,
                        scens=scens, qs=qs, hours=hours, months=np.array(a.months), ndays=nd)
    del acc, accg

    # ---- pass 2: deserts, one (scenario, threshold) at a time --------------
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H = len(hours)
    for s in scens:
        for q in qs:
            key = f"des_{s}_q{q}"
            # rows that ever appear
            jall = []
            for p in a.inputs:
                with np.load(p, allow_pickle=False) as z:
                    jall.append(np.unique(z[key + "_j"]))
            jids = np.unique(np.concatenate(jall)) if jall else np.array([], dtype=np.int32)
            del jall
            row = pd.Series(np.arange(len(jids)), index=jids)
            dense = np.zeros((len(jids), len(M), H), dtype=np.uint16)
            for p in a.inputs:
                mi = M[month_of(p)]
                with np.load(p, allow_pickle=False) as z:
                    j, h, n = z[key + "_j"], z[key + "_h"], z[key + "_n"]
                if len(j):
                    np.add.at(dense, (row.loc[j].to_numpy(), np.full(len(j), mi), h.astype(np.int64)),
                              n.astype(np.uint16))
            r, m, h = np.nonzero(dense)
            df = pd.DataFrame({"jid": jids[r].astype(np.int32), "month": np.array(a.months)[m].astype(np.int8),
                               "hour": hours[h].astype(np.int8), "n_desert": dense[r, m, h],
                               "n_days": nd[m, h].astype(np.uint16)})
            df.to_parquet(out / f"deserts_{s}_q{q}.parquet", index=False)
            print(f"{key}: {len(jids)} carreaux, {len(df)} rows")
            del dense, df
    print(f"{len(a.inputs)} months done")


if __name__ == "__main__":
    main()
