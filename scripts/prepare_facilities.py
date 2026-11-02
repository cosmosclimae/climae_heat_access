#!/usr/bin/env python3
"""Data ES -> facility table (outdoor supply + indoor fallback) with attributes x_i.

Unit = équipement (one row per 'Numéro de l'équipement sportif'); activities are
not exploded. Outputs one parquet with column `layer` in {outdoor, indoor}.

x_i attributes written here: covered, surface_class, lit, access_public,
seasonal, surface_m2. Canopy fraction is added by facility_canopy.py.
"""
import argparse
import io
import zipfile

import numpy as np
import pandas as pd
import yaml

C = {
    "id": "Numéro de l'équipement sportif", "inst": "Numéro de l'installation sportive",
    "name": "Nom de l'équipement sportif", "type": "Type d'équipement sportif",
    "family": "Famille d'équipement sportif", "nature": "Nature de l'équipement sportif",
    "soil": "Nature du sol", "light": "Eclairage de l'aire d'évolution",
    "free": "Equipement d'accès libre", "use": "Type d'utilisation",
    "seasonal": "Ouverture exclusivement saisonnière", "hs": "Installation hors-service",
    "surface": "Surface de l'aire d'évolution", "lon": "Longitude", "lat": "Latitude",
    "dep": "Département Code", "region": "Région Nom", "insee": "Commune INSEE", "qpv": "QPV", "qpv200": "QPV à 200 mètres",
    "density": "Densite Catégorie", "rnb": "RNB ID",
}


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "oui", "1", "yes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-es", required=True)
    ap.add_argument("--departements", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))["facilities"]

    if a.data_es.endswith(".zip"):
        with zipfile.ZipFile(a.data_es) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            raw = pd.read_csv(io.BytesIO(z.read(name)), sep=";", dtype=str, low_memory=False, encoding="utf-8-sig")
    else:
        raw = pd.read_csv(a.data_es, sep=";", dtype=str, low_memory=False, encoding="utf-8-sig")
    df = raw[[v for v in C.values() if v in raw.columns]].rename(columns={v: k for k, v in C.items()})
    rep = {"rows_raw": len(df)}

    df["lon"] = pd.to_numeric(df["lon"].str.replace(",", "."), errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"].str.replace(",", "."), errors="coerce")
    inside = df.lon.between(-5.6, 9.8) & df.lat.between(41.2, 51.3)   # metropolitan France incl. Corsica
    rep["dropped_outside_metropole_or_no_xy"] = int((~inside).sum())
    df = df[inside]
    hs = truthy(df["hs"].fillna(""))
    rep["dropped_hors_service"] = int(hs.sum())
    df = df[~hs]
    dup = df.duplicated("id")
    rep["dropped_duplicate_id"] = int(dup.sum())
    df = df[~dup].copy()

    out_m = df.family.isin(cfg["outdoor_families"]) & df.nature.isin(cfg["outdoor_natures"])
    in_m = df.family.isin(cfg["indoor_families"]) & df.nature.isin(cfg["indoor_natures"])
    df["layer"] = np.select([out_m, in_m], ["outdoor", "indoor"], default="")
    rep["outdoor"] = int(out_m.sum())
    rep["indoor"] = int(in_m.sum())
    rep["excluded_type_or_nature"] = int((df.layer == "").sum())
    df = df[df.layer != ""].copy()

    df["covered"] = df.nature.isin(cfg["covered_natures"])
    df["surface_class"] = df.soil.map(cfg["surface_map"]).fillna("reference")
    df["lit"] = truthy(df["light"].fillna(""))
    use = df["use"].fillna("")
    df["access_public"] = truthy(df["free"].fillna("")) | use.str.contains("Individuel", regex=False)
    df["seasonal"] = truthy(df["seasonal"].fillna(""))
    df["surface_m2"] = pd.to_numeric(df["surface"].str.replace(",", "."), errors="coerce")
    # QPV columns hold the QPV code (e.g. QN03107M), empty otherwise
    df["qpv"] = df["qpv"].fillna("").str.strip().ne("")
    df["qpv200"] = df.get("qpv200", pd.Series("", index=df.index)).fillna("").str.strip().ne("") | df["qpv"]
    # capacity: unit (main) or area normalised by the type median, winsorised
    med = df.groupby("type")["surface_m2"].transform("median")
    ratio = (df.surface_m2 / med).clip(lower=0.25, upper=4.0)
    df["capacity_unit"] = 1.0
    df["capacity_surface"] = ratio.fillna(1.0)

    # département from the coordinates (Data ES codes are not zero-padded and sometimes missing)
    import geopandas as gpd
    dep = gpd.read_file(a.departements).to_crs(2154)[["code", "geometry"]]
    pts = gpd.GeoDataFrame(index=df.index, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(2154)
    j = gpd.sjoin_nearest(pts, dep, how="left", max_distance=5000, distance_col="d_dep")
    j = j[~j.index.duplicated()]
    df["dep_declared"] = df["dep"]
    df["dep"] = j["code"].reindex(df.index)
    rep["dropped_no_departement_within_5km"] = int(df.dep.isna().sum())
    df = df[df.dep.notna()]

    df = df.sort_values(["layer", "id"]).reset_index(drop=True)
    df["fid"] = np.arange(len(df), dtype=np.int32)
    keep = ["fid", "id", "inst", "layer", "name", "type", "family", "nature", "soil", "surface_class",
            "covered", "lit", "access_public", "seasonal", "surface_m2", "capacity_unit", "capacity_surface",
            "lon", "lat", "dep", "dep_declared", "region", "insee", "qpv", "qpv200", "density", "rnb"]
    df[keep].to_parquet(a.out, index=False)

    o = df[df.layer == "outdoor"]
    rep["outdoor_covered_share"] = round(float(o.covered.mean()), 4)
    rep["outdoor_lit_share"] = round(float(o.lit.mean()), 4)
    rep["outdoor_public_share"] = round(float(o.access_public.mean()), 4)
    rep["outdoor_surface_class"] = o.surface_class.value_counts().to_dict()
    rep["outdoor_by_family"] = o.family.value_counts().to_dict()
    with open(a.report, "w", encoding="utf-8") as f:
        yaml.safe_dump(rep, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
