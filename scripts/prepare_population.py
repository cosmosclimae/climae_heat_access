#!/usr/bin/env python3
"""Filosofi 2021 -> demand points at 200 m.

mode nivnaturel_dasymetric
    200-m cells are kept as is. Larger cells (1 km ... 64 km) are split into
    their 200-m sub-cells, counts shared in proportion to OSM residential
    building footprint (buildings_dep.py). A large cell with no building
    footprint keeps one point at its centroid (flag `fallback_centroid`).
    Ratios (living standard, poverty) are inherited from the parent cell.
mode native_200m
    reads a strict 200-m file (same column names).

Groups: ind, ind_6_10, ind_11_17, ind_18_64 (18_24+25_39+40_54+55_64),
ind_65p (65_79+80p). Living standard = ind_snv / ind (parent level);
national population-weighted quintiles are computed on parent cells.
"""
import argparse
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

AGE = {"ind_18_64": ["ind_18_24", "ind_25_39", "ind_40_54", "ind_55_64"], "ind_65p": ["ind_65_79", "ind_80p"]}
COUNTS = ["ind", "ind_6_10", "ind_11_17", "ind_18_64", "ind_65p", "men", "men_pauv"]
RX = re.compile(r"RES(\d+)mN(\d+)E(\d+)")


def wquantile_bins(x, w, q=5):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    out = np.empty(len(x), dtype=np.int8)
    out[o] = np.minimum((cw * q).astype(int), q - 1) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--buildings", nargs="*", default=[])
    ap.add_argument("--departements", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    mode = cfg["population"]["mode"]

    g = gpd.read_file(a.gpkg, ignore_geometry=True)
    for k, cols in AGE.items():
        g[k] = g[cols].sum(axis=1)
    m = g["idcar_nat"].str.extract(RX).astype(np.int64)
    g["res"], g["n0"], g["e0"] = m[0], m[1], m[2]
    g["nivvie"] = np.where(g.ind > 0, g.ind_snv / g.ind.where(g.ind > 0), np.nan)
    g["pauv_share"] = np.where(g.men > 0, g.men_pauv / g.men.where(g.men > 0), np.nan)
    ok = g.nivvie.notna() & (g.ind > 0)
    g["q_nivvie"] = 0
    g.loc[ok, "q_nivvie"] = wquantile_bins(g.loc[ok, "nivvie"].to_numpy(), g.loc[ok, "ind"].to_numpy())
    g["parent"] = np.arange(len(g))
    rep = {"mode": mode, "parent_cells": len(g), "population": float(g.ind.sum()),
           "population_by_cell_size": g.groupby("res").ind.sum().round(0).to_dict()}

    small = g[g.res == 200]
    pts = [pd.DataFrame({"parent": small.parent, "n": small.n0, "e": small.e0, "share": 1.0,
                         "native200": True, "fallback_centroid": False})]
    big = g[g.res > 200]
    if mode == "nivnaturel_dasymetric" and len(big):
        bld = pd.concat([pd.read_parquet(p, columns=["n", "e", "area_m2"]) for p in a.buildings]) if a.buildings else \
            pd.DataFrame({"n": [], "e": [], "area_m2": []})
        bld = bld.groupby(["n", "e"], as_index=False).area_m2.sum()
        # expand each big parent into its 200-m sub-cells (vectorised by size)
        sub = []
        for res, grp in big.groupby("res"):
            k = res // 200
            r, c = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            r, c = r.ravel(), c.ravel()
            sub.append(pd.DataFrame({
                "parent": np.repeat(grp.parent.to_numpy(), k * k),
                "n": np.repeat(grp.n0.to_numpy(), k * k) + np.tile(r * 200, len(grp)),
                "e": np.repeat(grp.e0.to_numpy(), k * k) + np.tile(c * 200, len(grp))}))
        sub = pd.concat(sub, ignore_index=True).merge(bld, on=["n", "e"], how="inner")
        tot = sub.groupby("parent").area_m2.transform("sum")
        sub["share"] = sub.area_m2 / tot
        sub["native200"] = False
        sub["fallback_centroid"] = False
        pts.append(sub.drop(columns="area_m2"))
        miss = big[~big.parent.isin(sub.parent.unique())]
    else:
        miss = big
    if len(miss):
        half = (miss.res // 2 // 200) * 200
        pts.append(pd.DataFrame({"parent": miss.parent, "n": miss.n0 + half, "e": miss.e0 + half, "share": 1.0,
                                 "native200": False, "fallback_centroid": True}))
    p = pd.concat(pts, ignore_index=True)
    p = p.merge(g[["parent", "idcar_nat", "res", "nivvie", "pauv_share", "q_nivvie"] + COUNTS], on="parent")
    for c in COUNTS:
        p[c] = p[c] * p["share"]
    p = p[p.ind > 0].reset_index(drop=True)
    p["x3035"], p["y3035"] = p.e + 100.0, p.n + 100.0
    lon, lat = Transformer.from_crs(3035, 4326, always_xy=True).transform(p.x3035.to_numpy(), p.y3035.to_numpy())
    p["lon"], p["lat"] = lon, lat
    dep = gpd.read_file(a.departements).to_crs(4326)[["code", "geometry"]]
    gp = gpd.GeoDataFrame(p[["lon", "lat"]], geometry=gpd.points_from_xy(p.lon, p.lat), crs=4326)
    j = gpd.sjoin(gp, dep, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    p["dep"] = j["code"].reindex(p.index).to_numpy()
    # points falling just outside polygons (coast): nearest département
    if p.dep.isna().any():
        na = p.dep.isna()
        gna = gp[na.to_numpy()].to_crs(2154)
        nn = gpd.sjoin_nearest(gna, dep.to_crs(2154), how="left")
        nn = nn[~nn.index.duplicated()]
        p.loc[na, "dep"] = nn["code"].to_numpy()
    p["jid"] = np.arange(len(p), dtype=np.int32)
    p["cell200"] = "N" + p.n.astype(str) + "E" + p.e.astype(str)
    cols = ["jid", "cell200", "idcar_nat", "res", "native200", "fallback_centroid", "x3035", "y3035", "lon", "lat",
            "dep", "nivvie", "pauv_share", "q_nivvie"] + COUNTS
    p[cols].to_parquet(a.out, index=False)
    rep.update({"points": len(p), "population_out": float(p.ind.sum()),
                "points_native200": int(p.native200.sum()),
                "population_fallback_centroid": float(p.loc[p.fallback_centroid, "ind"].sum()),
                "parents_fallback_centroid": int(len(miss))})
    with open(a.report, "w", encoding="utf-8") as f:
        yaml.safe_dump(rep, f, allow_unicode=True, sort_keys=False)
    print(rep)


if __name__ == "__main__":
    main()
