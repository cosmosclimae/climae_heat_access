"""Vectorised Liljegren WBGT, numerically aligned on the climae.io reference.

Reference implementation (scalar, single source of truth):
    climae-school/scripts/wbgt_liljegren.py      (sensor energy balances)
    climae-school/scripts/exposure_inputs.py     (Erbs split, midpoint, wind, LW)
    config/exposure_reference_v2.json            (policy, fingerprinted)
    config/liljegren_sensors_v1.json             (sensors, fingerprinted)
A frozen copy lives in vendor/climae_school/ and tests/test_liljegren_np.py
checks this port against it (|dWBGT| < 0.01 degC on random cases).

What differs from the scalar reference, by design:
  * arrays instead of scalars; invalid cases return NaN + a flag code instead
    of raising (a 30-year national run cannot stop on one cell-hour);
  * Kt > 1 is clipped to 1 and flagged (reference: needs_review);
  * positive energy in a fully nocturnal hour is set to 0 and flagged;
  * natural wet bulb below 0 degC (ice wick) -> NaN + flag (reference raises);
  * OPTIONAL ground-surface modifier for the sun scene (paper-specific, not in
    climae.io): ground albedo alpha_s and ground temperature
    Ts = Ta + k_s * GHI_mid. alpha_s = policy ground_albedo and k_s = 0
    reproduce the reference exactly ("reference" surface class).

Units: K, %, Pa, m/s, W/m2, J/m2. Times are UTC numpy datetime64.
The Argonne/Liljegren licence notice is in vendor/climae_school/.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CP, M_AIR, M_WATER, R_GAS = 1003.5, 28.97, 18.015, 8314.34
R_AIR = R_GAS / M_AIR
PR = CP / (CP + 1.25 * R_AIR)
RATIO = CP * M_AIR / M_WATER

# flag bits
F_KT_CLIPPED = 1
F_NIGHT_ENERGY = 2
F_ICE_WICK = 4
F_NO_ROOT = 8
F_NONCONV = 16
F_VAPOUR = 32
F_EMIS_CAP = 64


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    solar_constant_w_m2: float
    quadrature_step_seconds: int
    min_daylight_elevation_deg: float
    low_sun_policy: str
    source_wind_height_m: float
    exposure_height_m: float
    roughness_length_m: float
    displacement_height_m: float
    convection_min_wind_m_s: float
    ground_albedo: float
    ground_emissivity: float

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("partition_method") != "erbs_1982_hourly":
            raise ValueError("policy: only erbs_1982_hourly is ported")
        if raw.get("shade_method") != "osha_zero_shortwave":
            raise ValueError("policy: only osha_zero_shortwave shade is ported")
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Sensors:
    globe_diameter_m: float
    globe_emissivity: float
    globe_shortwave_albedo: float
    wick_diameter_m: float
    wick_length_m: float
    wick_emissivity: float
    wick_shortwave_albedo: float
    stefan_boltzmann_w_m2_k4: float
    temperature_tolerance_k: float
    residual_tolerance_w_m2: float
    max_iterations: int

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("model") != "liljegren_energy_balance_explicit_radiation_v1":
            raise ValueError("unknown sensor model")
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# thermophysics (vectorised copies of wbgt_liljegren.py)
# --------------------------------------------------------------------------
def saturation_pressure_pa(tk):
    return 100 * 1.004 * 6.1121 * np.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def viscosity(tk):
    omega = (tk / 97.0 - 2.9) / .4 * (-.034) + 1.048
    return 2.6693e-6 * np.sqrt(M_AIR * tk) / (3.617 ** 2 * omega)


def conductivity(tk):
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, p):
    return (3.640e-4 * (tk / math.sqrt(132 * 647.3)) ** 2.334
            * (36.4 * 218) ** (1 / 3) * (132 * 647.3) ** (5 / 12)
            * math.sqrt(1 / M_AIR + 1 / M_WATER) / (p / 101325) * 1e-4)


def latent_heat(tk):
    return (313.15 - tk) / 30 * (-71100) + 2.4073e6


def convection(tk, p, wind, diameter, sphere):
    density = p / (R_AIR * tk)
    reynolds = wind * density * diameter / viscosity(tk)
    if sphere:
        nu = 2 + .6 * np.sqrt(reynolds) * PR ** .3333
    else:
        nu = .281 * reynolds ** .6 * PR ** .44
    return nu * conductivity(tk) / diameter


def _bisect(residual, low, high, s: Sensors, flags):
    """Vectorised copy of root_bisection (same stopping rule)."""
    low = low.astype(np.float64).copy()
    high = high.astype(np.float64).copy()
    fl = residual(low)
    fh = residual(high)
    bad = ~np.isfinite(fl) | ~np.isfinite(fh) | (fl * fh > 0)
    flags |= np.where(bad, F_NO_ROOT, 0).astype(flags.dtype)
    out = np.full(low.shape, np.nan)
    active = ~bad
    for _ in range(s.max_iterations):
        if not active.any():
            break
        mid = (low + high) / 2
        fm = residual(mid)
        done = active & (np.abs(fm) <= s.residual_tolerance_w_m2) & ((high - low) <= s.temperature_tolerance_k)
        out[done] = mid[done]
        active &= ~done
        go_low = active & (fl * fm <= 0)
        go_high = active & ~(fl * fm <= 0)
        high = np.where(go_low, mid, high)
        low = np.where(go_high, mid, low)
        fl = np.where(go_high, fm, fl)
    flags |= np.where(active, F_NONCONV, 0).astype(flags.dtype)
    return out


def solve_sensors(ta, rh_pct, p, wind, dni, dhi, refl, ldown, lup, mu, s: Sensors):
    """Natural wet bulb and globe (degC); all inputs broadcastable arrays.

    wind = exposure-height wind already floored for convection.
    """
    ta, rh_pct, p, wind, dni, dhi, refl, ldown, lup, mu = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64) for x in (ta, rh_pct, p, wind, dni, dhi, refl, ldown, lup, mu)])
    flags = np.zeros(ta.shape, dtype=np.int16)
    rh = rh_pct / 100
    eair = rh * saturation_pressure_pa(ta)
    vap_bad = eair >= .95 * p
    flags |= np.where(vap_bad, F_VAPOUR, 0).astype(np.int16)
    sigma = s.stefan_boltzmann_w_m2_k4
    incoming_lw = .5 * (ldown + lup)
    sw_globe = (1 - s.globe_shortwave_albedo) * (dni / 4 + dhi / 2 + refl / 2)
    cap = .25 * s.wick_diameter_m / s.wick_length_m
    sw_wick = (1 - s.wick_shortwave_albedo) * (
        dhi * (1 + cap) + dni * (np.sqrt(np.maximum(0, 1 - mu * mu)) / math.pi + cap * np.maximum(mu, 0)) + refl)

    def globe_residual(tg):
        tref = .5 * (tg + ta)
        h = convection(tref, p, wind, s.globe_diameter_m, True)
        return s.globe_emissivity * (incoming_lw - sigma * tg ** 4) + sw_globe - h * (tg - ta)

    def wick_residual(tw):
        tref = .5 * (tw + ta)
        h = convection(tref, p, wind, s.wick_diameter_m, False)
        schmidt = viscosity(tref) / ((p / (R_AIR * tref)) * diffusivity(tref, p))
        ew = saturation_pressure_pa(tw)
        evaporation = h * latent_heat(tref) / RATIO * (ew - eair) / (p - ew) * (PR / schmidt) ** .56
        radiation = s.wick_emissivity * (incoming_lw - sigma * tw ** 4) + sw_wick
        return h * (ta - tw) - evaporation + radiation

    upper = np.minimum(363.15, ta + 45)
    for _ in range(200):                       # same downward walk as reference
        too_high = saturation_pressure_pa(upper) >= .95 * p
        if not too_high.any():
            break
        upper = np.where(too_high, upper - 1, upper)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tw = _bisect(wick_residual, np.full(ta.shape, 223.15), upper, s, flags)
        tg = _bisect(globe_residual, np.full(ta.shape, 223.15), np.full(ta.shape, 423.15), s, flags)
    ice = tw < 273.15
    flags |= np.where(ice, F_ICE_WICK, 0).astype(np.int16)
    tw = np.where(ice | vap_bad, np.nan, tw)
    tg = np.where(vap_bad, np.nan, tg)
    return tw - 273.15, tg - 273.15, flags


# --------------------------------------------------------------------------
# solar geometry and shortwave preparation (exposure_inputs.py)
# --------------------------------------------------------------------------
def _year_frac(t_utc):
    """t_utc: datetime64[ns] array -> (doy, hour_float, days_in_year)."""
    t = t_utc.astype("datetime64[s]")
    year = t.astype("datetime64[Y]")
    doy = (t.astype("datetime64[D]") - year.astype("datetime64[D]")).astype(np.int64) + 1
    sec = (t - t.astype("datetime64[D]")).astype(np.int64)
    hour = sec / 3600.0
    y = year.astype(np.int64) + 1970
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    return doy, hour, np.where(leap, 366, 365)


def solar_geometry(lat_deg, lon_deg, t_utc, solar_constant):
    """NOAA fractional-year approx (same as exposure_inputs.solar_geometry)."""
    doy, hour, ndays = _year_frac(t_utc)
    lat = np.radians(lat_deg)
    gamma = 2 * np.pi / ndays * (doy - 1 + (hour - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                   - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    dec = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
           - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
           - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    angle = np.radians((hour * 60 + eq + 4 * lon_deg) / 4 - 180)
    mu = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(angle)
    mu = np.clip(mu, -1.0, 1.0)
    b = 2 * np.pi * (doy - 1) / 365
    orbital = (1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b)
               + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))
    return mu, np.degrees(np.arcsin(mu)), solar_constant * orbital


def erbs(k):
    return np.where(k <= 0.22, 1 - 0.09 * k,
                    np.where(k <= 0.8, 0.9511 - 0.1604 * k + 4.388 * k ** 2 - 16.638 * k ** 3 + 12.336 * k ** 4,
                             0.165))


def prepare_shortwave(energy_j_m2, start_utc, lat, lon, pol: Policy):
    """Hourly interval [start, start+1h]: midpoint DNI/DHI/GHI + flags.

    energy_j_m2: (..., ) broadcast with lat/lon/start_utc.
    Returns dict of arrays: dni, dhi, ghi, bhi, mu_mid, elev_mid, flags.
    """
    energy = np.asarray(energy_j_m2, dtype=np.float64)
    start = np.asarray(start_utc).astype("datetime64[ns]")
    step = pol.quadrature_step_seconds
    n = 3600 // step
    shape = np.broadcast(energy, start, lat, lon).shape
    mu_sum = np.zeros(shape)
    toa_sum = np.zeros(shape)
    elev_min = np.full(shape, np.inf)
    elev_max = np.full(shape, -np.inf)
    for t_off in [0, 3600]:                                   # interval bounds
        _, el, _ = solar_geometry(lat, lon, start + np.timedelta64(t_off, "s"), pol.solar_constant_w_m2)
        elev_min = np.minimum(elev_min, el)
        elev_max = np.maximum(elev_max, el)
    for i in range(n):
        tt = start + np.timedelta64(int((i + 0.5) * step * 1e6), "us")
        mu, el, ext = solar_geometry(lat, lon, tt, pol.solar_constant_w_m2)
        mu_sum += np.maximum(0, mu)
        toa_sum += np.maximum(0, mu) * ext
        elev_min = np.minimum(elev_min, el)
        elev_max = np.maximum(elev_max, el)
    mu_mean = mu_sum / n
    toa_mean = toa_sum / n
    mu_mid, el_mid, _ = solar_geometry(lat, lon, start + np.timedelta64(1800, "s"), pol.solar_constant_w_m2)
    flags = np.zeros(shape, dtype=np.int16)
    energy = np.broadcast_to(energy, shape).copy()
    night = (energy > 0) & ((mu_mean <= 0) | (elev_max <= 0))
    flags |= np.where(night, F_NIGHT_ENERGY, 0).astype(np.int16)
    energy = np.where(night, 0.0, energy)
    extended = pol.low_sun_policy == "all_diffuse_positive_cosine_profile"
    low_sun = elev_min < pol.min_daylight_elevation_deg
    if not extended and np.any((energy > 0) & low_sun):
        raise ValueError("policy v1 refuses low-sun energy; use exposure_reference_v2")
    ghi_mean = energy / 3600.0
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(toa_mean > 0, ghi_mean / toa_mean, 0.0)
    clip = k > 1
    flags |= np.where(clip, F_KT_CLIPPED, 0).astype(np.int16)
    k = np.minimum(k, 1.0)
    fd = np.where(extended & low_sun, 1.0, erbs(k))
    dhi_mean = fd * ghi_mean
    bhi_mean = ghi_mean - dhi_mean
    with np.errstate(invalid="ignore", divide="ignore"):
        dni = np.where(mu_mean > 0, bhi_mean / mu_mean, 0.0)
        mu_m = np.maximum(0.0, mu_mid)
        dhi_mid = np.where(mu_mean > 0, dhi_mean * mu_m / mu_mean, 0.0)
    if extended:
        dni = np.where(mu_m == 0, 0.0, dni)
    bhi_mid = dni * mu_m
    return {"dni": dni, "dhi": dhi_mid, "bhi": bhi_mid, "ghi": dhi_mid + bhi_mid,
            "mu_mid": mu_mid, "elev_mid": el_mid, "flags": flags}


def wind_factor(pol: Policy):
    z0, d = pol.roughness_length_m, pol.displacement_height_m
    return math.log((pol.exposure_height_m - d) / z0) / math.log((pol.source_wind_height_m - d) / z0)


def longwave(ta_k, rh_pct, pol: Policy):
    tc = ta_k - 273.15
    vap = rh_pct / 100 * 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    raw = 0.575 * vap ** 0.143
    em = np.minimum(1.0, raw)
    bb = 5.670374419e-8 * ta_k ** 4
    down = em * bb
    up = pol.ground_emissivity * bb + (1 - pol.ground_emissivity) * down
    return down, up, raw > 1


def rh_from_dew(t_k, td_k):
    """RH (%) from T and Td, Bolton over water (same relation as the dashboard)."""
    tc, tdc = t_k - 273.15, td_k - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    e = 6.112 * np.exp(17.67 * tdc / (tdc + 243.5))
    return np.clip(100 * e / es, 0, 100)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def wbgt_interval(*, energy_j_m2, start_utc, lat, lon,
                  t0_k, t1_k, rh0, rh1, p0_pa, p1_pa, ws0, ws1,
                  pol: Policy, sens: Sensors, surfaces=None):
    """WBGT for hourly intervals, sun (per surface class) and OSHA shade.

    Instantaneous states at interval start (0) and end (1) are averaged
    (constant_transmission_midpoint + linear_T_RH_P_speed_midpoint).
    surfaces: dict name -> (albedo, k_surface [K per W/m2]). None -> reference.
    Returns dict: wbgt_shade, wbgt_sun_<name>, elev_mid, ghi, flags.
    """
    sw = prepare_shortwave(energy_j_m2, start_utc, lat, lon, pol)
    ta = (np.asarray(t0_k) + np.asarray(t1_k)) / 2
    rh = (np.asarray(rh0) + np.asarray(rh1)) / 2
    p = (np.asarray(p0_pa) + np.asarray(p1_pa)) / 2
    ws = (np.asarray(ws0) + np.asarray(ws1)) / 2
    w_conv = np.maximum(ws * wind_factor(pol), pol.convection_min_wind_m_s)
    ldown, lup_ref, emis_cap = longwave(ta, rh, pol)
    flags = sw["flags"] | np.where(emis_cap, F_EMIS_CAP, 0).astype(np.int16)
    out = {"elev_mid": sw["elev_mid"], "ghi": sw["ghi"]}
    zero = np.zeros_like(ta)
    tw, tg, fl = solve_sensors(ta, rh, p, w_conv, zero, zero, zero, ldown, lup_ref, sw["mu_mid"], sens)
    out["wbgt_shade"] = .7 * tw + .3 * tg
    flags = flags | fl
    if surfaces is None:
        surfaces = {"reference": (pol.ground_albedo, 0.0)}
    sigma_ref = 5.670374419e-8
    for name, (alb, ks) in surfaces.items():
        refl = alb * sw["ghi"]
        if ks:
            ts = ta + ks * sw["ghi"]
            lup = pol.ground_emissivity * sigma_ref * ts ** 4 + (1 - pol.ground_emissivity) * ldown
        else:
            lup = lup_ref
        tw, tg, fl = solve_sensors(ta, rh, p, w_conv, sw["dni"], sw["dhi"], refl, ldown, lup, sw["mu_mid"], sens)
        out[f"wbgt_sun_{name}"] = .7 * tw + .2 * tg + .1 * (ta - 273.15)
        flags = flags | fl
    out["flags"] = flags
    return out
