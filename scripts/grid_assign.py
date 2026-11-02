#!/usr/bin/env python3
"""ERA5-Land land grid + nearest valid land cell for facilities and carreaux.

Outputs
  cells.parquet          : every land cell (cell, ilat, ilon, lat, lon, dep)
  facility_cells.parquet : fid -> cell (outdoor + indoor)
  wbgt_cells.parquet     : subset of cells hosting outdoor facilities (order used
                           by wbgt_month.py: column `k`)
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import open_era5, var


def nearest(cells, lon, lat):
    c = np.cos(np.radians(46.5))
    tree = cKDTree(np.c_[cells.lon.to_numpy() * c, cells.lat.to_numpy()])
    d, k = tree.query(np.c_[lon * c, lat])
    return cells.cell.to_numpy()[k], d * 111.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="un fichier ERA5-Land mensuel")
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--departements", required=True)
    ap.add_argument("--cells-out", required=True)
    ap.add_argument("--fac-out", required=True)
    ap.add_argument("--wbgt-cells-out", required=True)
    a = ap.parse_args()

    ds, tn, la, lo = open_era5(a.template)
    t2m = ds[var(ds, "t2m")].isel({tn: 0}).values
    lats, lons = ds[la].values, ds[lo].values
    ii, jj = np.where(np.isfinite(t2m))
    cells = pd.DataFrame({"ilat": ii, "ilon": jj, "lat": lats[ii], "lon": lons[jj]})
    cells["cell"] = np.arange(len(cells), dtype=np.int32)
    dep = gpd.read_file(a.departements).to_crs(4326)[["code", "geometry"]]
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    j = gpd.sjoin(pts, dep, how="left", predicate="within").drop_duplicates("cell")
    cells["dep"] = j["code"].to_numpy()
    cells.to_parquet(a.cells_out, index=False)

    fac = pd.read_parquet(a.facilities)
    fc, dist = nearest(cells, fac.lon.to_numpy(), fac.lat.to_numpy())
    out = pd.DataFrame({"fid": fac.fid, "cell": fc, "dist_km": dist.round(2), "layer": fac.layer})
    out.to_parquet(a.fac_out, index=False)

    used = np.unique(out.loc[out.layer == "outdoor", "cell"])
    wc = cells.set_index("cell").loc[used].reset_index()
    wc["k"] = np.arange(len(wc), dtype=np.int32)
    wc.to_parquet(a.wbgt_cells_out, index=False)
    print(f"land cells {len(cells)} ; cells hosting outdoor facilities {len(wc)} ; "
          f"max facility-cell distance {dist.max():.1f} km")


if __name__ == "__main__":
    main()
