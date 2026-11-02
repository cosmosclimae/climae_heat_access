"""Port check: scripts/liljegren_np.py vs frozen climae.io scalar reference.

Run:  python -m pytest tests/ -q      (or: python tests/test_liljegren_np.py)
Draws random May-September daytime cases over France and compares sun and
shade WBGT end-to-end (hourly energy + instantaneous states -> WBGT).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "climae_school"))

import liljegren_np as L                                   # noqa: E402
from exposure_inputs import ExposurePolicy, prepare_exposure   # noqa: E402
from wbgt_liljegren import SensorConfig, compute_prepared_exposure, PhysicsError  # noqa: E402

POL_PATH = ROOT / "config" / "exposure_reference_v2.json"
SENS_PATH = ROOT / "config" / "liljegren_sensors_v1.json"


def _cases(n, seed=7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        lat = rng.uniform(42.0, 51.0)
        lon = rng.uniform(-4.5, 8.0)
        day = datetime(int(rng.integers(1995, 2025)), int(rng.integers(5, 10)), int(rng.integers(1, 29)), tzinfo=timezone.utc)
        start = day + timedelta(hours=int(rng.integers(4, 21)))
        t0 = rng.uniform(285, 312)
        t1 = t0 + rng.normal(0, 1)
        rh0 = rng.uniform(15, 95)
        rh1 = np.clip(rh0 + rng.normal(0, 4), 1, 100)
        p0 = rng.uniform(88000, 102500)
        ws0, ws1 = rng.uniform(0, 8, 2)
        kt = rng.uniform(0.05, 0.8)
        yield dict(lat=lat, lon=lon, start=start, t0=t0, t1=t1, rh0=rh0, rh1=rh1, p0=p0, p1=p0 + 20,
                   ws0=ws0, ws1=ws1, kt=kt)


def run(n=300, tol=0.01):
    pol_s = ExposurePolicy.load(POL_PATH)
    sens_s = SensorConfig.load(SENS_PATH)
    pol = L.Policy.load(POL_PATH)
    sens = L.Sensors.load(SENS_PATH)
    errs, skipped, rows = [], 0, []
    for c in _cases(n):
        # energy consistent with a clearness index (so the reference accepts it)
        mu, el, ext = L.solar_geometry(c["lat"], c["lon"], np.datetime64(c["start"].replace(tzinfo=None) + timedelta(minutes=30)), 1361.0)
        energy = max(0.0, float(c["kt"] * ext * max(mu, 0) * 3600 * 0.95))
        st = lambda t, T, RH, P, W: {"time": t.isoformat(), "temperature_k": T, "relative_humidity_percent": RH,
                                     "pressure_pa": P, "wind_speed_m_s": W}
        end = c["start"] + timedelta(hours=1)
        try:
            prep = prepare_exposure(global_energy_j_m2=energy, start=c["start"], end=end, lat=c["lat"], lon=c["lon"],
                                    start_state=st(c["start"], c["t0"], c["rh0"], c["p0"], c["ws0"]),
                                    end_state=st(end, c["t1"], c["rh1"], c["p1"], c["ws1"]),
                                    policy=pol_s, source_provenance="synthetic test")
            ref = compute_prepared_exposure(prep, sensor_config=sens_s)
        except (PhysicsError, ValueError):
            skipped += 1
            continue
        rows.append((c, energy, ref))
    # one vectorised call for all accepted cases
    get = lambda k: np.array([r[0][k] for r in rows])
    out = L.wbgt_interval(
        energy_j_m2=np.array([r[1] for r in rows]),
        start_utc=np.array([np.datetime64(r[0]["start"].replace(tzinfo=None)) for r in rows]),
        lat=get("lat"), lon=get("lon"), t0_k=get("t0"), t1_k=get("t1"), rh0=get("rh0"), rh1=get("rh1"),
        p0_pa=get("p0"), p1_pa=get("p1"), ws0=get("ws0"), ws1=get("ws1"), pol=pol, sens=sens)
    sun_ref = np.array([r[2]["wbgt_sun_c"] for r in rows])
    sha_ref = np.array([r[2]["wbgt_shade_c"] for r in rows])
    d_sun = np.abs(out["wbgt_sun_reference"] - sun_ref)
    d_sha = np.abs(out["wbgt_shade"] - sha_ref)
    report = {"compared": len(rows), "skipped_by_reference": skipped,
              "max_abs_diff_sun_c": float(np.nanmax(d_sun)), "max_abs_diff_shade_c": float(np.nanmax(d_sha)),
              "nan": int(np.isnan(d_sun).sum() + np.isnan(d_sha).sum())}
    return report, tol


def test_port_matches_reference():
    report, tol = run()
    assert report["compared"] >= 200, report
    assert report["nan"] == 0, report
    assert report["max_abs_diff_sun_c"] < tol, report
    assert report["max_abs_diff_shade_c"] < tol, report


def test_surface_modifier_monotonic():
    pol = L.Policy.load(POL_PATH)
    sens = L.Sensors.load(SENS_PATH)
    kw = dict(energy_j_m2=np.array([3.0e6]), start_utc=np.array([np.datetime64("2019-06-27T11:00")]),
              lat=np.array([43.6]), lon=np.array([3.9]), t0_k=np.array([307.0]), t1_k=np.array([308.0]),
              rh0=np.array([30.0]), rh1=np.array([28.0]), p0_pa=np.array([101000.0]), p1_pa=np.array([101000.0]),
              ws0=np.array([2.0]), ws1=np.array([2.0]), pol=pol, sens=sens)
    out = L.wbgt_interval(surfaces={"reference": (0.2, 0.0), "synthetic": (0.10, 0.025)}, **kw)
    assert out["wbgt_sun_synthetic"][0] > out["wbgt_sun_reference"][0] > out["wbgt_shade"][0]


if __name__ == "__main__":
    print(run()[0])
