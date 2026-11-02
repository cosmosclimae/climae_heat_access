"""Liljegren heat/mass balances; Climae.io Python adaptation, 2026.

This product includes software produced by UChicago Argonne, LLC under
Contract No. DE-AC02-06CH11357 with the Department of Energy.
Original copyright/license retained at the end of this source and in
THIRD_PARTY_NOTICES.txt. Reference: doi:10.1080/15459620802310770.

Changes: separate SW/LW inputs, configurable globe, bracketed root solving,
strict units, separate shade solves. No empirical simplified WBGT equation.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from exposure_inputs import number, utc

CP, M_AIR, M_WATER, R_GAS = 1003.5, 28.97, 18.015, 8314.34
R_AIR = R_GAS / M_AIR
PR = CP / (CP + 1.25 * R_AIR)
RATIO = CP * M_AIR / M_WATER


class PhysicsError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class SensorConfig:
    model: str
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

    def __post_init__(self):
        if self.model != "liljegren_energy_balance_explicit_radiation_v1":
            raise PhysicsError("Modèle de capteurs inconnu")
        for k in ("globe_emissivity", "wick_emissivity"):
            number(getattr(self, k), k, 0.1, 1)
        for k in ("globe_shortwave_albedo", "wick_shortwave_albedo"):
            number(getattr(self, k), k, 0, 1)
        number(self.globe_diameter_m, "globe_diameter_m", .025, .2)
        number(self.wick_diameter_m, "wick_diameter_m", .005, .009)
        number(self.wick_length_m, "wick_length_m", .02, .04)
        number(self.stefan_boltzmann_w_m2_k4, "sigma", 5.66e-8, 5.68e-8)
        number(self.temperature_tolerance_k, "temperature_tolerance_k", 1e-8, .001)
        number(self.residual_tolerance_w_m2, "residual_tolerance_w_m2", 1e-6, .1)
        if type(self.max_iterations) is not int or not 2 <= self.max_iterations <= 500:
            raise PhysicsError("max_iterations invalide")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def saturation_pressure_pa(tk):
    """Buck/Wexler liquid-water relation and 1.004 correction from reference C.

    Output Pa, including its explicit hPa -> Pa factor. No ice-wick solution.
    The original constant moist-air correction is retained, not pressure-fitted.
    """
    return 100 * 1.004 * 6.1121 * math.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def viscosity(tk):
    omega = (tk / 97.0 - 2.9) / .4 * (-.034) + 1.048
    return 2.6693e-6 * math.sqrt(M_AIR * tk) / (3.617**2 * omega)


def conductivity(tk):
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, pressure_pa):
    return (3.640e-4 * (tk / math.sqrt(132 * 647.3))**2.334
            * (36.4 * 218)**(1 / 3) * (132 * 647.3)**(5 / 12)
            * math.sqrt(1 / M_AIR + 1 / M_WATER) / (pressure_pa / 101325) * 1e-4)


def latent_heat(tk):
    # Exact empirical function of the reference code, not substituted silently.
    return (313.15 - tk) / 30 * (-71100) + 2.4073e6


def convection(tk, pressure_pa, wind_m_s, diameter_m, *, sphere):
    density = pressure_pa / (R_AIR * tk)
    reynolds = wind_m_s * density * diameter_m / viscosity(tk)
    nu = 2 + .6 * math.sqrt(reynolds) * PR**.3333 if sphere else .281 * reynolds**.6 * PR**.44
    return nu * conductivity(tk) / diameter_m


def root_bisection(residual, low, high, config):
    """Both temperature-bracket width and energy residual must converge."""
    fl, fh = residual(low), residual(high)
    if not all(math.isfinite(v) for v in (fl, fh)) or fl * fh > 0:
        raise PhysicsError("Bilan sans racine encadrée : aucun WBGT substitué")
    for iteration in range(1, config.max_iterations + 1):
        mid = (low + high) / 2
        fm = residual(mid)
        if not math.isfinite(fm):
            raise PhysicsError("Résidu non fini")
        if abs(fm) <= config.residual_tolerance_w_m2 and high - low <= config.temperature_tolerance_k:
            return mid, {"converged": True, "iterations": iteration,
                         "residual_w_m2": fm, "bracket_width_k": high - low}
        if fl * fm <= 0:
            high = mid
        else:
            low, fl = mid, fm
    raise PhysicsError("Non-convergence du bilan : aucun WBGT substitué")


def solve_sensors(temperature_k, rh_percent, pressure_pa, wind_m_s, scene, cos_zenith, *, config):
    """Return natural wet bulb and globe in degC, independent of WBGT weights.

    Wind is ALREADY at exposure height and regularized for convection.
    Scene uses DNI, DHI, upward reflected SW, downward/upward LW in W/m2.
    Radiation geometry follows the original vertical-wick/spherical-globe model.
    """
    ta = number(temperature_k, "temperature_k", 273.15, 333.15)
    rh = number(rh_percent, "rh_percent", 0, 100) / 100
    pressure = number(pressure_pa, "pressure_pa", 40000, 110000)
    speed = number(wind_m_s, "convection_wind_m_s", .01, 100)
    mu = number(cos_zenith, "cos_zenith", -1, 1)
    keys = ["direct_normal_w_m2", "diffuse_horizontal_w_m2", "reflected_upward_shortwave_w_m2",
            "longwave_down_w_m2", "longwave_up_w_m2"]
    flux = {k: number(scene[k], k, 0, 2000) for k in keys}
    dni, dhi, reflected, ldown, lup = (flux[k] for k in keys)
    if dni > 0 and mu <= 0:
        raise PhysicsError("Faisceau solaire sous l'horizon")
    eair = rh * saturation_pressure_pa(ta)
    if eair >= .95 * pressure:
        raise PhysicsError("Pression de vapeur incompatible avec la pression totale")
    sigma = config.stefan_boltzmann_w_m2_k4
    incoming_lw = .5 * (ldown + lup)
    sw_globe = (1 - config.globe_shortwave_albedo) * (dni / 4 + dhi / 2 + reflected / 2)
    cap = .25 * config.wick_diameter_m / config.wick_length_m
    sw_wick = (1 - config.wick_shortwave_albedo) * (
        dhi * (1 + cap) + dni * (math.sqrt(max(0, 1 - mu * mu)) / math.pi + cap * max(mu, 0)) + reflected)

    def globe_residual(tg):
        tref = .5 * (tg + ta)
        h = convection(tref, pressure, speed, config.globe_diameter_m, sphere=True)
        return config.globe_emissivity * (incoming_lw - sigma * tg**4) + sw_globe - h * (tg - ta)

    def wick_residual(tw):
        tref = .5 * (tw + ta)
        h = convection(tref, pressure, speed, config.wick_diameter_m, sphere=False)
        schmidt = viscosity(tref) / ((pressure / (R_AIR * tref)) * diffusivity(tref, pressure))
        ew = saturation_pressure_pa(tw)
        evaporation = h * latent_heat(tref) / RATIO * (ew - eair) / (pressure - ew) * (PR / schmidt)**.56
        radiation = config.wick_emissivity * (incoming_lw - sigma * tw**4) + sw_wick
        return h * (ta - tw) - evaporation + radiation

    upper = min(363.15, ta + 45)
    while saturation_pressure_pa(upper) >= .95 * pressure:
        upper -= 1
    tw, wet_diag = root_bisection(wick_residual, 223.15, upper, config)
    tg, globe_diag = root_bisection(globe_residual, 223.15, 423.15, config)
    if tw < 273.15:
        raise PhysicsError("Mèche sous 0 °C : phase glace non implémentée")
    flags = []
    if not (283 <= (ta + tw) / 2 <= 313 and 283 <= (ta + tg) / 2 <= 313):
        flags.append("reference_thermophysical_correlations_extrapolated")
    if pressure < 80000:
        flags.append("pressure_below_reference_moist_air_correction_range")
    return {"tnwb_c": tw - 273.15, "tg_c": tg - 273.15,
            "diagnostics": {"natural_wet_bulb": wet_diag, "globe": globe_diag},
            "flags": flags}


def compute_wbgt_liljegren(T, RH, pressure, wind, solar, lat, lon, datetime, *, sensor_config):
    """Public contract: K, %, Pa, exposure convection m/s, explicit solar package.

    solar contains sun, osha_shade, cos_zenith, representative_time_utc,
    latitude and longitude. No second solar split, wind profile or default LW.
    """
    when = utc(datetime)
    number(lat, "latitude", -90, 90)
    number(lon, "longitude", -180, 180)
    if utc(solar["representative_time_utc"]) != when or solar["latitude"] != lat or solar["longitude"] != lon:
        raise PhysicsError("Géométrie solaire préparée pour un autre lieu ou instant")
    shade = solar["osha_shade"]
    for key in ("direct_normal_w_m2", "diffuse_horizontal_w_m2", "reflected_upward_shortwave_w_m2"):
        if shade[key] != 0:
            raise PhysicsError("Ombre OSHA : toutes les composantes solaires doivent être nulles")
    for key in ("longwave_down_w_m2", "longwave_up_w_m2"):
        if shade[key] != solar["sun"][key]:
            raise PhysicsError("Ce scénario impose la même fermeture infrarouge soleil/ombre")
    sun_result = solve_sensors(T, RH, pressure, wind, solar["sun"], solar["cos_zenith"], config=sensor_config)
    shade_result = solve_sensors(T, RH, pressure, wind, shade, solar["cos_zenith"], config=sensor_config)
    sun_wbgt = .7 * sun_result["tnwb_c"] + .2 * sun_result["tg_c"] + .1 * (T - 273.15)
    shade_wbgt = .7 * shade_result["tnwb_c"] + .3 * shade_result["tg_c"]
    return {"time_utc": when.isoformat(), "tnwb_sun_c": sun_result["tnwb_c"], "tg_sun_c": sun_result["tg_c"],
            "tnwb_shade_c": shade_result["tnwb_c"], "tg_shade_c": shade_result["tg_c"],
            "wbgt_sun_c": sun_wbgt, "wbgt_shade_c": shade_wbgt,
            "diagnostics": {"sun": sun_result["diagnostics"], "shade": shade_result["diagnostics"]},
            "sensor_config_sha256": sensor_config.fingerprint(),
            "flags": sorted(set(sun_result["flags"] + shade_result["flags"]))}


def compute_prepared_exposure(prepared, *, sensor_config):
    """Adapter from v0.4 preparation, without recomputing or hiding its policies."""
    met, rad = prepared["meteorology_midpoint"], prepared["radiation"]
    loc = prepared["location"]
    solar = {**prepared["scenes"], "cos_zenith": rad["midpoint_geometry"]["cos_zenith"],
             "representative_time_utc": rad["representative_time_utc"], **loc}
    result = compute_wbgt_liljegren(met["temperature_k"], met["relative_humidity_percent"], met["pressure_pa"],
                                  prepared["wind"]["convection_speed_m_s"], solar,
                                  loc["latitude"], loc["longitude"], rad["representative_time_utc"], sensor_config=sensor_config)
    result["exposure_policy_sha256"] = prepared["policy_sha256"]
    result["source_provenance"] = prepared["source_provenance"]
    result["flags"] = sorted(set(result["flags"] + [f for f in prepared["flags"] if f != "no_WBGT_solver_in_this_release"]))
    result["flags"].append("modelled_wbgt_not_on_site_measurement")
    if "midpoint_shortwave_active" in rad:
        result["solar_exposure_active"] = rad["midpoint_shortwave_active"]
        if not result["solar_exposure_active"]:
            result["flags"].append("zero_shortwave_fixed_sun_formula_not_a_shade_benefit")
    return result


# Original notice retained for the Climae.io Python adaptation above.
# /*
#  * The MIT License (MIT)
#  * 
#  * Copyright (c) 2016-2017 Max Lieblich
#  *
#  * Permission is hereby granted, free of charge, to any person obtaining a copy
#  * of this software and associated documentation files (the "Software"), to deal
#  * in the Software without restriction, including without limitation the rights
#  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  * copies of the Software, and to permit persons to whom the Software is
#  * furnished to do so, subject to the following conditions:
#  *   
#  * The above copyright notice and this permission notice shall be included in all
#  * copies or substantial portions of the Software.
#  * 
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  * SOFTWARE.
#  *
#  */
# 
# /*
#  * For the original file, see "wbgt.c.original" in the src directory of this 
#  * package. That file has been modified in the following ways:
#  * 
#  * 1. The "main" function has been removed.
#  * 2. All function declarations have been converted from K&R style to ANSI C99
#  *    compliant style and moved to "wbgt.h".
#  * 3. All floats have been converted to doubles to integrate more easily with R.
#  * 4. Parameter documentation for functions is included in "wbgt.h"; these are
#  *    copies of comments in "wbgt.c.original" accompanying the original K&R
#  *    parameter declarations.
#  * 5. For documentation in its original location, see "wbgt.c.original"
#  * 
#  * Max Lieblich, University of Washington, 2016
#  */
# 
# /* Original license follows */
# 
# /*
#                Copyright © 2008, UChicago Argonne, LLC
#                        All Rights Reserved
# 
#                         WBGT, Version 1.1
# 
# 			     James C. Liljegren
#               Decision & Information Sciences Division
# 
# 			     OPEN SOURCE LICENSE
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.  Software changes, 
#    modifications, or derivative works, should be noted with comments and 
#    the author and organization’s name.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
# 
# 3. Neither the names of UChicago Argonne, LLC or the Department of Energy 
#    nor the names of its contributors may be used to endorse or promote products 
#    derived from this software without specific prior written permission.
# 
# 4. The software and the end-user documentation included with the 
#    redistribution, if any, must include the following acknowledgment:
# 
#    "This product includes software produced by UChicago Argonne, LLC 
#    under Contract No. DE-AC02-06CH11357 with the Department of Energy.”
# 
# ******************************************************************************************
# DISCLAIMER
# 
# THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
# 
# NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, 
# NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS 
# OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, 
# COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA, APPARATUS, PRODUCT, OR 
# PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
# *******************************************************************************************
# */
#  
