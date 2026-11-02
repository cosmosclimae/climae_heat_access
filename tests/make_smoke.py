#!/usr/bin/env python3
"""Synthetic end-to-end test on one département (no OSM, no real ERA5).

Builds in --work: synthetic ERA5-Land files (2019-2020, July-August, Hérault
box), facilities/population subsets of the département, a Euclidean x 1.3 OD
table, dummy OSM files with old timestamps, and override.yaml. Then:

  python tests/make_smoke.py --work /tmp/smoke --facilities <facilities.parquet> \
         --population <population.parquet> --geojson <departements.geojson> --gpkg <filosofi.gpkg>
  snakemake -j 4 --configfile config/config.yaml /tmp/smoke/override.yaml --rerun-triggers mtime

facilities.parquet / population.parquet = outputs of rules `facilities` and
`population` (or of the scripts run by hand). Climate values are synthetic:
only the plumbing is tested, not the numbers.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree


def era5_month(path, y, m, rng):
    lat = np.round(np.arange(43.95, 43.15, -0.1), 2)
    lon = np.round(np.arange(2.5, 4.25, 0.1), 2)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    sea = (LA < 43.45) & (LO > 3.3)
    t = pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month * 24, freq="h")
    hr = t.hour.to_numpy() + 0.0
    day = t.dayofyear.to_numpy()
    heat = rng.normal(0, 2.5, size=day.max() + 1)[day]
    T = 273.15 + 27 + heat[:, None, None] + 7 * np.sin((hr[:, None, None] - 9) / 24 * 2 * np.pi) \
        + (3.5 - LO[None]) * 0.8 + (43.6 - LA[None]) * 2
    Td = T - 12 + rng.normal(0, 1, (len(t), 1, 1))
    sp_ = np.full(T.shape, 100500.) - (LA[None] - 43.2) * 2000
    u = np.full(T.shape, 2.0) + rng.normal(0, .5, (len(t), 1, 1))
    v = np.full(T.shape, 1.0)
    lst = hr + 0.25
    cz = np.clip(np.sin(np.radians(43.6)) * np.sin(np.radians(20)) +
                 np.cos(np.radians(43.6)) * np.cos(np.radians(20)) * np.cos((lst - 12) / 12 * np.pi), 0, None)
    cloud = rng.uniform(0.6, 1.0, size=day.max() + 1)[day]
    g = (900 * cz * cloud)[:, None, None] * np.ones_like(T)
    acc = np.zeros_like(g)
    for i in range(len(t)):
        acc[i] = g[i] * 3600 if t[i].hour == 1 else (acc[i - 1] if i else 0) + g[i] * 3600
    ds = xr.Dataset({k: (("valid_time", "latitude", "longitude"), x.astype("float32"))
                     for k, x in dict(t2m=T, d2m=Td, sp=sp_, u10=u, v10=v, ssrd=acc).items()},
                    coords={"valid_time": t, "latitude": lat, "longitude": lon})
    for k in ds.data_vars:
        ds[k] = ds[k].where(~sea[None])
    ds.to_netcdf(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--dep", default="34")
    a = ap.parse_args()
    w = Path(a.work)
    for d in ["inputs/era5", "data", "od", "osm"]:
        (w / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for y in [2019, 2020]:
        for m in [7, 8]:
            era5_month(w / f"inputs/era5/era5land_full_{y}-{m:02d}.nc", y, m, rng)
    (w / "inputs/france.osm.pbf").write_text("dummy")
    old = time.time() - 3600 * 24 * 30
    os.utime(w / "inputs/france.osm.pbf", (old, old))
    f = pd.read_parquet(a.facilities)
    f = f[f.dep == a.dep].copy()
    f["fid"] = np.arange(len(f), dtype=np.int32)
    p = pd.read_parquet(a.population)
    p = p[p.dep == a.dep].reset_index(drop=True)
    p["jid"] = np.arange(len(p), dtype=np.int32)
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    fx, fy = tr.transform(f.lon.to_numpy(), f.lat.to_numpy())
    pairs = cKDTree(np.c_[fx, fy]).query_ball_point(np.c_[p.x3035, p.y3035], r=1500 / 1.3)
    rows = [(j, f.fid.to_numpy()[i], f.layer.to_numpy()[i],
             np.hypot(p.x3035[j] - fx[i], p.y3035[j] - fy[i]) * 1.3 / 1.25 / 60) for j, L in enumerate(pairs) for i in L]
    od = pd.DataFrame(rows, columns=["jid", "fid", "layer", "tau_min"]).astype({"tau_min": "float32"})
    (w / f"osm/dep_{a.dep}.osm.pbf").write_text("dummy")
    time.sleep(1)
    pd.DataFrame({"cell200": [], "n": [], "e": [], "area_m2": []}).to_parquet(w / f"osm/buildings_{a.dep}.parquet")
    time.sleep(1)
    f.to_parquet(w / "data/facilities.parquet", index=False)
    p.to_parquet(w / "data/population.parquet", index=False)
    (w / "data/population_report.yaml").write_text("smoke: true\n")
    (w / "data/facilities_report.yaml").write_text("smoke: true\n")
    time.sleep(1)
    od.to_parquet(w / f"od/od_{a.dep}.parquet", index=False)
    (w / f"od/od_{a.dep}.yaml").write_text("smoke: true\n")
    ov = {"work_root": str(w), "deps": [a.dep],
          "inputs": {"era5land_dir": str(w / "inputs/era5"), "era5land_pattern": "era5land_full_{year}-{month}.nc",
                     "data_es_zip": str(w / "inputs/france.osm.pbf"), "departements_geojson": a.geojson,
                     "filosofi_gpkg": a.gpkg, "filosofi_200m_gpkg": "", "osm_pbf": str(w / "inputs/france.osm.pbf"),
                     "canopy_raster": ""},
          "climate": {"periods": {"ref": [2019, 2020], "recent": [2020, 2020]}, "months": [7, 8]},
          "adaptation": {"period": "recent", "months": [7, 8], "budgets_pct": [1, 5, 10],
                         "equity_groups": ["ind_6_10", "ind_11_17", "ind_65p"], "equity_quintiles": [1, 2]},
          "bootstrap_n": 50}
    yaml.safe_dump(ov, open(w / "override.yaml", "w"), sort_keys=False)
    print(f"smoke set ready in {w}: {len(f)} facilities, {len(p)} carreaux, {len(od)} OD pairs")


if __name__ == "__main__":
    main()
