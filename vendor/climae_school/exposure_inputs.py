"""Explicit reference exposure conditions, NOT a WBGT/Liljegren solver.

All choices come from a mandatory, versioned policy. See METHODE_PHYSIQUE_v04.md.
Standard library only; no API request, no GRIB relabelling, no hidden defaults.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import calendar
import hashlib
import json
import math
from pathlib import Path


def number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: nombre requis")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name}: valeur finie attendue dans [{lower}, {upper}]")
    return float(value)


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Date avec fuseau horaire obligatoire")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class ExposurePolicy:
    schema_version: int
    scenario_id: str
    partition_method: str
    solar_position_method: str
    solar_constant_w_m2: float
    quadrature_step_seconds: int
    min_daylight_elevation_deg: float
    high_clearness_policy: str
    low_sun_policy: str
    temporal_reconstruction: str
    meteorology_interpolation: str
    wind_method: str
    terrain: str
    source_wind_height_m: float
    exposure_height_m: float
    roughness_length_m: float
    displacement_height_m: float
    convection_min_wind_m_s: float
    shade_method: str
    shade_direct_transmission: float
    shade_diffuse_transmission: float
    shade_reflected_transmission: float
    shade_wind_factor: float
    ground_albedo: float
    longwave_method: str
    ground_emissivity: float
    atmosphere_emissivity_overflow: str
    sun_wbgt_formula: str
    shade_wbgt_formula: str
    shade_label: str
    shade_definition: str

    def __post_init__(self):
        fixed = {
            "schema_version": 1,
            "partition_method": "erbs_1982_hourly",
            "solar_position_method": "noaa_fractional_year_spencer_distance",
            "high_clearness_policy": "reject_above_one",
            "temporal_reconstruction": "constant_transmission_midpoint",
            "meteorology_interpolation": "linear_T_RH_P_speed_midpoint",
            "wind_method": "neutral_log_profile",
            "terrain": "open_reference_not_local_courtyard",
            "shade_method": "osha_zero_shortwave",
            "shade_direct_transmission": 0.0,
            "shade_diffuse_transmission": 0.0,
            "shade_reflected_transmission": 0.0,
            "shade_wind_factor": 1.0,
            "longwave_method": "oke_emissivity_ground_at_air_temperature",
            "atmosphere_emissivity_overflow": "cap_at_one_and_flag",
            "sun_wbgt_formula": "0.7*Tnwb_sun+0.2*Tg_sun+0.1*Ta",
            "shade_wbgt_formula": "0.7*Tnwb_shade+0.3*Tg_shade",
            "shade_label": "WBGT à l'ombre",
            "shade_definition": "scénario extérieur ombragé, sans charge solaire directe",
        }
        for name, expected in fixed.items():
            if isinstance(getattr(self, name), bool) or getattr(self, name) != expected:
                raise ValueError(f"Politique non implémentée : {name}={getattr(self, name)!r}")
        if self.low_sun_policy not in {"reject_positive_energy_below_threshold", "all_diffuse_positive_cosine_profile"}:
            raise ValueError("Politique Soleil bas non implémentée")
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id obligatoire")
        number(self.solar_constant_w_m2, "solar_constant_w_m2", 1300, 1400)
        if type(self.quadrature_step_seconds) is not int or not 1 <= self.quadrature_step_seconds <= 300:
            raise ValueError("quadrature_step_seconds: entier entre 1 et 300")
        if 3600 % self.quadrature_step_seconds:
            raise ValueError("Le pas de quadrature doit diviser une heure")
        number(self.min_daylight_elevation_deg, "min_daylight_elevation_deg", 1, 15)
        number(self.source_wind_height_m, "source_wind_height_m", 0.1, 100)
        number(self.exposure_height_m, "exposure_height_m", 0.1, 100)
        number(self.roughness_length_m, "roughness_length_m", 0.0001, 0.1)
        number(self.displacement_height_m, "displacement_height_m", 0, 1)
        for z in (self.source_wind_height_m, self.exposure_height_m):
            if z - self.displacement_height_m < 10 * self.roughness_length_m:
                raise ValueError("Profil ouvert refusé à moins de 10 z0 au-dessus de d")
        number(self.convection_min_wind_m_s, "convection_min_wind_m_s", 0.01, 1)
        number(self.ground_albedo, "ground_albedo", 0, 1)
        number(self.ground_emissivity, "ground_emissivity", 0.01, 1)

    @classmethod
    def load(cls, path):
        # No merging with defaults: missing keys and spelling mistakes fail.
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()


def solar_geometry(lat, lon, when, *, solar_constant_w_m2):
    """NOAA fractional-year approximation, geometric Sun, no refraction.

    Orbital irradiance factor: Spencer Fourier series, evaluated with 365 days.
    This is not NREL SPA, nor the solarposition routine in the original C model.
    """
    lat = math.radians(number(lat, "latitude", -90, 90))
    lon = number(lon, "longitude", -180, 180)
    solar_constant_w_m2 = number(solar_constant_w_m2, "solar_constant_w_m2", 1300, 1400)
    when = utc(when)
    hour = when.hour + when.minute / 60 + (when.second + when.microsecond / 1e6) / 3600
    doy = when.timetuple().tm_yday
    gamma = 2 * math.pi / (366 if calendar.isleap(when.year) else 365) * (doy - 1 + (hour - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                   - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    dec = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
           - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
           - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    angle = math.radians((hour * 60 + eq + 4 * lon) / 4 - 180)
    mu = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(angle)
    mu = min(1.0, max(-1.0, mu))  # round-off protection for asin only
    b = 2 * math.pi * (doy - 1) / 365
    orbital = (1.00011 + 0.034221 * math.cos(b) + 0.001280 * math.sin(b)
               + 0.000719 * math.cos(2 * b) + 0.000077 * math.sin(2 * b))
    return {"cos_zenith": mu, "elevation_deg": math.degrees(math.asin(mu)),
            "extraterrestrial_normal_w_m2": solar_constant_w_m2 * orbital}


def erbs_diffuse_fraction(clearness_index):
    """Erbs et al. (1982), hourly correlation; NOT an AROME direct observation."""
    k = number(clearness_index, "clearness_index", 0, 1)
    if k <= 0.22:
        return 1 - 0.09 * k
    if k <= 0.8:
        return 0.9511 - 0.1604 * k + 4.388 * k**2 - 16.638 * k**3 + 12.336 * k**4
    return 0.165


def prepare_shortwave(*, global_energy_j_m2, start, end, lat, lon, policy):
    """One already-decumulated PT1H interval, with explicit midpoint reconstruction.

    No native-direct argument is accepted until its meaning has been validated.
    v1 refuses dawn/dusk; v2 explicitly assumes all diffuse for low-Sun hours.
    Both preserve the original hourly energy and reject positive all-night energy.
    """
    energy = number(global_energy_j_m2, "global_energy_j_m2", 0, 10_000_000)
    start, end = utc(start), utc(end)
    seconds = (end - start).total_seconds()
    if seconds != 3600:
        raise ValueError("Un intervalle PT1H de 3600 s est requis; ne pas différencier les PT1H")
    midpoint = start + (end - start) / 2
    def geometry(t):
        return solar_geometry(lat, lon, t, solar_constant_w_m2=policy.solar_constant_w_m2)
    step = policy.quadrature_step_seconds
    nodes = [geometry(start + timedelta(seconds=(i + 0.5) * step)) for i in range(3600 // step)]
    middle = geometry(midpoint)
    bounds = [geometry(start), geometry(end), *nodes]
    min_elevation = min(x["elevation_deg"] for x in bounds)
    max_elevation = max(x["elevation_deg"] for x in bounds)
    mu_mean = sum(max(0, x["cos_zenith"]) for x in nodes) / len(nodes)
    toa_mean = sum(max(0, x["cos_zenith"]) * x["extraterrestrial_normal_w_m2"] for x in nodes) / len(nodes)
    flags = ["direct_diffuse_estimated_erbs", "subhourly_radiation_reconstructed_not_observed"]
    extended = policy.low_sun_policy == "all_diffuse_positive_cosine_profile"
    low_sun = min_elevation < policy.min_daylight_elevation_deg
    if energy > 0 and low_sun and not extended:
        raise ValueError("Rayonnement positif avec Soleil bas, lever/coucher ou nuit : needs_review; aucune substitution par zéro")
    if energy > 0 and (mu_mean <= 0 or max_elevation <= 0):
        raise ValueError("Énergie solaire positive sur une heure entièrement nocturne : needs_review; aucune substitution par zéro")
    mean_ghi = energy / seconds
    k = mean_ghi / toa_mean if toa_mean > 0 else 0.0
    if k > 1:
        raise ValueError("Kt > 1 : needs_review; pas d'écrêtage du rayonnement AROME")
    fd = 1.0 if extended and low_sun else erbs_diffuse_fraction(k)
    mean_dhi = fd * mean_ghi
    mean_bhi = mean_ghi - mean_dhi
    dni = mean_bhi / mu_mean if mu_mean > 0 else 0.0
    mu_mid = max(0.0, middle["cos_zenith"])
    dhi_mid = mean_dhi * mu_mid / mu_mean if mu_mean > 0 else 0.0
    bhi_mid = dni * mu_mid
    if energy == 0 and middle["elevation_deg"] > policy.min_daylight_elevation_deg:
        flags.append("zero_global_energy_in_daylight_review")
    extra = {}
    if extended:
        if max_elevation <= 0:
            regime = "night"
        elif min_elevation <= 0:
            regime = "sunrise" if bounds[1]["cos_zenith"] > bounds[0]["cos_zenith"] else "sunset"
        else:
            regime = "low_sun" if low_sun else "day"
        if low_sun:
            flags.remove("direct_diffuse_estimated_erbs")
            flags.append("all_diffuse_low_sun_assumption_not_measured_partition" if energy else "zero_shortwave_interval")
        if mu_mid == 0:
            dni = 0.0
            flags.append("midpoint_below_geometric_horizon")
        # The same quadrature used for normalization must recover the input.
        reconstructed = sum(mean_ghi * max(0, g["cos_zenith"]) / mu_mean * step
                            for g in nodes) if mu_mean > 0 else 0.0
        extra = {"solar_regime": regime, "min_elevation_deg": min_elevation,
                 "max_elevation_deg": max_elevation,
                 "daylight_fraction_quadrature": sum(g["cos_zenith"] > 0 for g in nodes)/len(nodes),
                 "reconstructed_energy_j_m2": reconstructed,
                 "energy_closure_error_j_m2": reconstructed-energy,
                 "partition_method_used": "all_diffuse" if low_sun else "erbs_1982_hourly",
                 "midpoint_shortwave_active": dhi_mid+bhi_mid > 0}
    return {"source": "global_hourly_energy_only_native_direct_excluded",
            "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "representative_time_utc": midpoint.isoformat(), "duration_seconds": seconds,
            "global_energy_j_m2": energy, "mean_ghi_w_m2": mean_ghi,
            "mean_dhi_w_m2": mean_dhi, "mean_bhi_w_m2": mean_bhi,
            "extraterrestrial_horizontal_mean_w_m2": toa_mean,
            "mean_positive_cos_zenith": mu_mean, "clearness_index": k,
            "diffuse_fraction_estimated": fd, "midpoint_geometry": middle,
            "midpoint_ghi_w_m2": dhi_mid + bhi_mid, "midpoint_dhi_w_m2": dhi_mid,
            "midpoint_bhi_w_m2": bhi_mid, "midpoint_dni_w_m2": dni,
            "flags": flags, **extra}


def wind_at_exposure(speed_m_s, *, policy):
    speed = number(speed_m_s, "source_wind_speed_m_s", 0, 100)
    z0, d = policy.roughness_length_m, policy.displacement_height_m
    factor = math.log((policy.exposure_height_m - d) / z0) / math.log((policy.source_wind_height_m - d) / z0)
    estimated = speed * factor
    flags = ["neutral_open_terrain_assumption_not_school_microclimate"]
    if estimated < policy.convection_min_wind_m_s:
        flags.append("convection_wind_floor_applied_estimated_wind_preserved")
    return {"source_speed_m_s": speed, "source_height_m": policy.source_wind_height_m,
            "exposure_height_m": policy.exposure_height_m, "profile_factor": factor,
            "estimated_speed_m_s": estimated,
            "convection_speed_m_s": max(estimated, policy.convection_min_wind_m_s),
            "flags": flags}


def interpolate_meteorology(start_state, end_state, *, start, end):
    """Explicit arithmetic midpoint, including RH and scalar speed (not vector mean).

    This is an approximation, not an hour mean or a conservation of water vapour.
    Temperatures remain at the AROME 2 m level, even if exposure height changes.
    """
    ranges = {"temperature_k": (233.15, 333.15), "relative_humidity_percent": (0, 100),
              "pressure_pa": (40_000, 110_000), "wind_speed_m_s": (0, 100)}
    states = []
    for state, instant in ((start_state, start), (end_state, end)):
        if set(state) != {*ranges, "time"}:
            raise ValueError("État météo : clés time, temperature_k, relative_humidity_percent, pressure_pa, wind_speed_m_s requises")
        if utc(state["time"]) != utc(instant):
            raise ValueError("Les états instantanés doivent encadrer exactement l'intervalle radiatif")
        states.append({k: number(state[k], k, *limits) for k, limits in ranges.items()})
    return {k: (states[0][k] + states[1][k]) / 2 for k in ranges}


def longwave_reference(temperature_k, relative_humidity_percent, *, policy):
    """Oke emissivity relation used in Liljegren, with an explicit physical cap.

    Saturation pressure: Bolton (1980) over liquid water, hPa.
    Ground temperature assumed equal to air; no measured/cloud-resolved longwave.
    Reflected downwelling longwave included for opaque grey ground.
    """
    tk = number(temperature_k, "temperature_k", 233.15, 333.15)
    rh = number(relative_humidity_percent, "relative_humidity_percent", 0, 100)
    tc = tk - 273.15
    vapor_hpa = rh / 100 * 6.112 * math.exp(17.67 * tc / (tc + 243.5))
    raw_emissivity = 0.575 * vapor_hpa**0.143
    emissivity = min(1.0, raw_emissivity)
    blackbody = 5.670374419e-8 * tk**4
    down = emissivity * blackbody
    up = policy.ground_emissivity * blackbody + (1 - policy.ground_emissivity) * down
    flags = ["longwave_empirical_not_arome", "ground_temperature_assumed_equal_air"]
    if raw_emissivity > 1:
        flags.append("atmospheric_emissivity_capped_at_one")
    return {"down_w_m2": down, "up_w_m2": up, "ground_temperature_k": tk,
            "atmospheric_emissivity_raw": raw_emissivity, "atmospheric_emissivity_used": emissivity,
            "vapor_pressure_hpa": vapor_hpa, "flags": flags}


def make_scenes(shortwave, longwave, wind, *, policy):
    """User-selected OSHA shade: ALL shortwave zero, longwave retained.

    Both natural wet-bulb and globe temperatures must be solved again for shade.
    This is a reference convention, not a simulation of a specific tree or roof.
    """
    reflected = policy.ground_albedo * shortwave["midpoint_ghi_w_m2"]
    common = {"diffuse_horizontal_w_m2": shortwave["midpoint_dhi_w_m2"],
              "reflected_upward_shortwave_w_m2": reflected,
              "longwave_down_w_m2": longwave["down_w_m2"],
              "longwave_up_w_m2": longwave["up_w_m2"],
              "estimated_wind_m_s": wind["estimated_speed_m_s"],
              "convection_wind_m_s": wind["convection_speed_m_s"],
              "sky_hemisphere_view_factor_sphere": 0.5,
              "ground_hemisphere_view_factor_sphere": 0.5}
    sun = {**common, "direct_normal_w_m2": shortwave["midpoint_dni_w_m2"],
           "direct_horizontal_w_m2": shortwave["midpoint_bhi_w_m2"],
           "solar_irradiance_w_m2": shortwave["midpoint_ghi_w_m2"]}
    shade = {**common, "direct_normal_w_m2": 0.0, "direct_horizontal_w_m2": 0.0,
             "diffuse_horizontal_w_m2": 0.0, "reflected_upward_shortwave_w_m2": 0.0,
             "solar_irradiance_w_m2": 0.0}
    # Incident SW averaged over a sphere's full surface, before absorption.
    # DNI/4 + DHI/2 + reflected/2, for isotropic diffuse and Lambertian ground.
    for scene in (sun, shade):
        scene["sphere_incident_shortwave_w_m2"] = (
            scene["direct_normal_w_m2"] / 4 + scene["diffuse_horizontal_w_m2"] / 2
            + scene["reflected_upward_shortwave_w_m2"] / 2)
    return {"sun": sun, "osha_shade": shade}


def prepare_exposure(*, global_energy_j_m2, start, end, lat, lon,
                     start_state, end_state, policy, source_provenance):
    if not isinstance(source_provenance, str) or not source_provenance.strip():
        raise ValueError("source_provenance obligatoire (réel/synthétique, run, maille et fichier)")
    sw = prepare_shortwave(global_energy_j_m2=global_energy_j_m2, start=start, end=end,
                           lat=lat, lon=lon, policy=policy)
    met = interpolate_meteorology(start_state, end_state, start=start, end=end)
    wind = wind_at_exposure(met["wind_speed_m_s"], policy=policy)
    lw = longwave_reference(met["temperature_k"], met["relative_humidity_percent"], policy=policy)
    return {"schema_version": 1, "kind": "physical_input_preparation_not_wbgt",
            "wbgt_computed": False, "scientifically_validated_for_schools": False,
            "policy": asdict(policy), "policy_sha256": policy.fingerprint(),
            "source_provenance": source_provenance,
            "location": {"latitude": lat, "longitude": lon},
            "radiation": sw, "meteorology_midpoint": met, "wind": wind,
            "longwave": lw, "scenes": make_scenes(sw, lw, wind, policy=policy),
            "flags": [*sw["flags"], *wind["flags"], *lw["flags"],
                      "linear_interpolation_of_instantaneous_T_RH_P_scalar_speed",
                      "osha_shade_all_shortwave_zero_longwave_retained",
                      "shade_requires_separate_natural_wet_bulb_and_globe_solves",
                      "reference_shade_not_tree_or_roof", "no_WBGT_solver_in_this_release"]}
