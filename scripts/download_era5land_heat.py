#!/usr/bin/env python3
"""ERA5-Land hourly, lean 'heat' profile, one month, France domain.

Same request pattern as climae_era5land_dashboard/scripts/download_era5land_month.py
(CDS GRIB -> CDO NetCDF4 -> cdo merge), same France box [51.5, -5.5, 41.0, 10.0],
so the grid is identical to the dashboard files. Only the 6 variables needed
by the Liljegren WBGT are requested (about 3x cheaper than profile `monthly`):
  2m_temperature, 2m_dewpoint_temperature, surface_pressure,
  10m_u_component_of_wind, 10m_v_component_of_wind,
  surface_solar_radiation_downwards
Requires ~/.cdsapirc and CDO in PATH (as for the dashboard).
"""
import argparse
import calendar
import shutil
import subprocess
import sys
from pathlib import Path

AREA = [51.5, -5.5, 41.0, 10.0]
VARS = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_solar_radiation_downwards"]


def run(cmd):
    print("[CMD] " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-fields", type=int, default=4000)
    a = ap.parse_args()
    if shutil.which("cdo") is None:
        sys.exit("CDO introuvable (sudo apt install cdo)")
    import cdsapi
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nd = calendar.monthrange(a.year, a.month)[1]
    per = max(1, a.max_fields // (nd * 24))
    groups = [VARS[i:i + per] for i in range(0, len(VARS), per)]
    client = cdsapi.Client()
    parts = []
    for gi, g in enumerate(groups, 1):
        grib = out.parent / f"{out.stem}.part{gi:02d}.grib"
        nc = out.parent / f"{out.stem}.part{gi:02d}.nc"
        parts.append(nc)
        if nc.exists() and nc.stat().st_size > 0:
            continue
        if not (grib.exists() and grib.stat().st_size > 0):
            tmp = Path(str(grib) + ".download")
            tmp.unlink(missing_ok=True)
            client.retrieve("reanalysis-era5-land", {
                "variable": g, "year": str(a.year), "month": f"{a.month:02d}",
                "day": [f"{d:02d}" for d in range(1, nd + 1)], "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA, "data_format": "grib", "download_format": "unarchived"}, str(tmp))
            tmp.replace(grib)
        tmpnc = Path(str(nc) + ".tmp")
        run(["cdo", "-f", "nc4c", "-z", "zip_1", "copy", grib, tmpnc])
        tmpnc.replace(nc)
        grib.unlink(missing_ok=True)
    if len(parts) == 1:
        parts[0].replace(out)
    else:
        tmp = Path(str(out) + ".merge.tmp.nc")
        run(["cdo", "-O", "-f", "nc4c", "-z", "zip_1", "merge", *parts, tmp])
        tmp.replace(out)
        for p in parts:
            p.unlink(missing_ok=True)
    print(f"[CDS] {out} ({out.stat().st_size / 1e6:.0f} Mo)")


if __name__ == "__main__":
    main()
