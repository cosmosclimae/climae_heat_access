#!/usr/bin/env python3
"""QPV flags per facility, read straight from Data ES (fix without re-running facilities).

Data ES stores the QPV *code* in 'QPV' and 'QPV à 200 mètres' (empty when outside).
Output: fid, qpv (inside a QPV), qpv200 (inside or within 200 m).
"""
import argparse
import io
import zipfile

import pandas as pd

ID = "Numéro de l'équipement sportif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-es", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = [ID, "QPV", "QPV à 200 mètres"]
    if a.data_es.endswith(".zip"):
        with zipfile.ZipFile(a.data_es) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            raw = pd.read_csv(io.BytesIO(z.read(name)), sep=";", dtype=str, usecols=cols, encoding="utf-8-sig")
    else:
        raw = pd.read_csv(a.data_es, sep=";", dtype=str, usecols=cols, encoding="utf-8-sig")
    raw = raw.drop_duplicates(ID).set_index(ID)
    fac = pd.read_parquet(a.facilities, columns=["fid", "id"])
    q = raw.reindex(fac.id.to_numpy())
    fac["qpv"] = q["QPV"].fillna("").str.strip().ne("").to_numpy()
    fac["qpv200"] = (q["QPV à 200 mètres"].fillna("").str.strip().ne("").to_numpy()) | fac.qpv.to_numpy()
    fac[["fid", "qpv", "qpv200"]].to_parquet(a.out, index=False)
    print(f"QPV: {int(fac.qpv.sum())} facilities inside, {int(fac.qpv200.sum())} inside or within 200 m")


if __name__ == "__main__":
    main()
