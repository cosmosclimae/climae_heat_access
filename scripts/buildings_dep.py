#!/usr/bin/env python3
"""OSM building footprint per 200-m INSPIRE cell (EPSG:3035) for one département.

Used as dasymetric weight to split Filosofi 'niveau naturel' cells > 200 m.
Buildings carrying a clearly non-residential `building=*` tag are dropped.
Each footprint is assigned to the 200-m cell of its centroid.
Output: parquet (cell200 = 'N{n}E{e}' lower-left metres, area_m2).
"""
import argparse

import numpy as np
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    drop = set(cfg["population"]["non_residential_building_tags"])
    from pyrosm import OSM
    b = OSM(a.pbf).get_buildings()
    if b is None or len(b) == 0:
        pd.DataFrame({"cell200": [], "area_m2": []}).to_parquet(a.out, index=False)
        return
    tag = b["building"].astype(str).str.lower() if "building" in b else pd.Series("yes", index=b.index)
    b = b[~tag.isin(drop)]
    b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(3035)
    area = b.geometry.area.to_numpy()
    cen = b.geometry.centroid
    n = (np.floor(cen.y.to_numpy() / 200) * 200).astype(np.int64)
    e = (np.floor(cen.x.to_numpy() / 200) * 200).astype(np.int64)
    df = pd.DataFrame({"n": n, "e": e, "area_m2": area})
    g = df.groupby(["n", "e"], as_index=False)["area_m2"].sum()
    g["cell200"] = "N" + g.n.astype(str) + "E" + g.e.astype(str)
    g[["cell200", "n", "e", "area_m2"]].to_parquet(a.out, index=False)
    print(f"{a.pbf}: {len(b)} residential-ish buildings -> {len(g)} cells")


if __name__ == "__main__":
    main()
