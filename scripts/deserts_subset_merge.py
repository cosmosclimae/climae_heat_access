#!/usr/bin/env python3
"""Sum yearly subset deserts (deserts_subset.py) over a period, month by month.

Memory-bounded: for each month, a dense (cells x hours) uint16 counter is filled
year by year (only that month's rows are read), then written as sparse rows.
Outputs: <out> jid, month, hour, n_desert ; <ndays> month, hour, n_days ;
<served> jid of cells with at least one subset facility in reach.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ndays", required=True)
    ap.add_argument("--served", required=True)
    a = ap.parse_args()
    nd = pd.concat([pd.read_parquet(Path(p).with_suffix(".ndays.parquet")) for p in a.inputs], ignore_index=True)
    nd = nd.groupby(["month", "hour"], as_index=False).n_days.sum()
    nd.to_parquet(a.ndays, index=False)
    hours = np.sort(nd.hour.unique())
    hpos = np.full(int(hours.max()) + 1, -1)
    hpos[hours] = np.arange(len(hours))
    nj = 0
    for p in a.inputs:
        nj = max(nj, int(pq.read_table(p, columns=["jid"]).column("jid").to_numpy().max()) + 1)
    writer = None
    for m in sorted(nd.month.unique()):
        acc = np.zeros((nj, len(hours)), dtype=np.uint16)
        for p in a.inputs:
            t = pq.read_table(p, filters=[("month", "=", int(m))]).to_pandas()
            np.add.at(acc, (t.jid.to_numpy(), hpos[t.hour.to_numpy()]), t.n_desert.to_numpy().astype(np.uint16))
        j, h = np.nonzero(acc)
        df = pd.DataFrame({"jid": j.astype(np.int32), "month": np.int8(m), "hour": hours[h].astype(np.int8),
                           "n_desert": acc[j, h]})
        tab = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(a.out, tab.schema)
        writer.write_table(tab)
        del acc
        print(f"month {m}: {len(df)} cell-hours")
    writer.close()
    pd.read_parquet(Path(a.inputs[0]).with_suffix(".served.parquet")).to_parquet(a.served, index=False)
    print(f"{len(a.inputs)} years merged")


if __name__ == "__main__":
    main()
