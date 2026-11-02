#!/usr/bin/env python3
"""H1 test, loss decomposition, temporary deserts, equity, typology, sensitivities.

Metric conventions (v2)
  combined   : main scenario (cover + canopy + surface + lighting) -> B
  heat-only  : analysis.heat_scenario (lighting constraint off)    -> B_heat
               darkness share of the combined loss = 1 - mean(B_heat) / mean(B)
  peak loss  : L_peak = mean loss over climate.peak_window (July 14-17 h local);
               daylight hours, so combined = heat-only there
  deserts    : expected hours per season with >= 1 outdoor facility within tau0
               but none available; reported per SERVED person (carreaux with
               A0 > 0); structural absence (A0 = 0) is reported separately.
All equity statistics are distributional — never population-weighted means of
A, which E2SFCA conservation makes identical across groups.
Bootstrap CIs resample ERA5-Land cells (cluster bootstrap, cluster-level sums).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

DIM = {5: 31, 6: 30, 7: 31, 8: 31, 9: 30}


def wq(x, w, qs):
    m = np.isfinite(x) & (w > 0)
    x, w = x[m], w[m]
    if not len(x):
        return [np.nan] * len(qs)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(np.searchsorted(cw, q), len(x) - 1)]) for q in qs]


def wmean(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def var_decomp(y, w, grp):
    """Population-weighted variance of y split into between- and within-group parts."""
    m = np.average(y, weights=w)
    tot = np.average((y - m) ** 2, weights=w)
    d = pd.DataFrame({"yw": y * w, "w": w, "g": grp}).groupby("g").sum()
    gm = d.yw / d.w
    between = float(((gm - m) ** 2 * d.w).sum() / d.w.sum())
    return tot, between / tot


def desert_hours(path):
    d = pd.read_parquet(path, columns=["jid", "month", "n_desert", "n_days"])
    h = d.n_desert.to_numpy(float) / d.n_days.to_numpy(float) * d.month.map(DIM).to_numpy(float)
    return pd.Series(h).groupby(d.jid.to_numpy()).sum()


def top_share(x, w, frac=0.1):
    """Share of person-hours held by the most exposed `frac` of the population."""
    o = np.argsort(-x)
    cw = np.cumsum(w[o]) / w.sum()
    ph = np.cumsum((x * w)[o])
    return float(ph[np.searchsorted(cw, frac)] / ph[-1]) if ph[-1] > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--fac-qpv", required=True)
    ap.add_argument("--fac-heat", required=True, help="facilities.parquet of the heat-only run (abar)")
    ap.add_argument("--access-main", required=True)
    ap.add_argument("--access-heat", required=True)
    ap.add_argument("--access-naive", required=True)
    ap.add_argument("--access-recent", required=True)
    ap.add_argument("--deserts-ref", required=True, help="directory clim/deserts_ref")
    ap.add_argument("--deserts-recent", required=True, help="directory clim/deserts_recent")
    ap.add_argument("--sens", nargs="*", default=[], help="label=path pairs")
    ap.add_argument("--od", default=None, help="data/od.parquet (supply decomposition of the income gap)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    acfg = cfg.get("analysis", {})
    rng = np.random.default_rng(cfg["seed"])
    nb = cfg["bootstrap_n"]
    q = cfg["climate"]["main_threshold"]
    s_main = cfg["modifiers"]["main"]
    s_heat = acfg.get("heat_scenario", "nolight")
    heavy = acfg.get("heavy_desert_hours", 100)
    lthr = acfg.get("loss_threshold", 0.25)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    groups = cfg["population"]["groups"]

    # ---- carreau table ------------------------------------------------------
    pop = pd.read_parquet(a.population, columns=["jid", "dep", "lon", "lat", "x3035", "y3035", "native200",
                                                 "pauv_share", "q_nivvie", *groups])
    take = ["jid", "A0_ind", "Aheat_ind", "B_ind", "Lpeak_ind", "Fin_ind", "has_outdoor"]
    acc = pd.read_parquet(a.access_main, columns=take)
    hea = pd.read_parquet(a.access_heat, columns=["jid", "Aheat_ind", "B_ind", "Lpeak_ind"]).rename(
        columns={"Aheat_ind": "Aheat_heat", "B_ind": "B_heat", "Lpeak_ind": "Lpeak_heat"})
    nai = pd.read_parquet(a.access_naive, columns=["jid", "B_ind", "Lpeak_ind"]).rename(
        columns={"B_ind": "B_naive", "Lpeak_ind": "Lpeak_naive"})
    rec = pd.read_parquet(a.access_recent, columns=["jid", "Lpeak_ind"]).rename(columns={"Lpeak_ind": "Lpeak_recent"})
    df = pop.merge(acc, on="jid").merge(hea, on="jid").merge(nai, on="jid").merge(rec, on="jid")
    del acc, hea, nai, rec

    cells = pd.read_parquet(a.cells)
    c = np.cos(np.radians(46.5))
    _, kk = cKDTree(np.c_[cells.lon * c, cells.lat]).query(np.c_[df.lon * c, df.lat])
    df["era5_cell"] = cells.cell.to_numpy()[kk]

    fac = pd.read_parquet(a.facilities, columns=["fid", "layer", "lon", "lat", "dep", "region", "density",
                                                 "family", "surface_class", "covered", "lit", "access_public"])
    fac = fac.merge(pd.read_parquet(a.fac_qpv), on="fid", how="left")
    dep2reg = fac.dropna(subset=["region"]).groupby("dep").region.agg(lambda x: x.value_counts().index[0])
    df["region"] = df.dep.map(dep2reg)
    # density category of the commune of the nearest facility (Data ES « Densite Catégorie »)
    fd = fac.dropna(subset=["density"])
    _, kf = cKDTree(np.c_[fd.lon * c, fd.lat]).query(np.c_[df.lon * c, df.lat])
    df["density"] = fd.density.to_numpy()[kf]
    pb = acfg.get("poverty_bins", [0.05, 0.10, 0.20, 0.30])
    labs = [f"<{int(pb[0] * 100)}%"] + [f"{int(lo * 100)}-{int(hi * 100)}%" for lo, hi in zip(pb[:-1], pb[1:])] + \
           [f">{int(pb[-1] * 100)}%"]
    df["poverty_class"] = pd.cut(df.pauv_share, [-np.inf, *pb, np.inf], labels=labs, right=False)
    df["poor"] = df.pauv_share.fillna(0) * df.ind               # persons in poor households (approx.)
    qpv_path = acfg.get("qpv_geojson") or ""
    if qpv_path and Path(qpv_path).exists():
        import geopandas as gpd
        qp = gpd.read_file(qpv_path).to_crs(3035)
        pts = gpd.GeoDataFrame({"jid": df.jid}, geometry=gpd.points_from_xy(df.x3035, df.y3035), crs=3035)
        inside = gpd.sjoin(pts, qp[["geometry"]], predicate="within", how="inner").jid.unique()
        df["qpv"] = df.jid.isin(inside)

    # ---- deserts ------------------------------------------------------------
    D = {}
    for per, d in [("ref", a.deserts_ref), ("recent", a.deserts_recent)]:
        for kind, s in [("heat", s_heat), ("combined", s_main)]:
            D[(per, kind)] = f"dh_{kind}_{per}"
            df[D[(per, kind)]] = df.jid.map(desert_hours(Path(d) / f"deserts_{s}_q{q}.parquet")).fillna(0.0)

    served = df.has_outdoor.to_numpy()
    S = df[served]
    ws = S.ind.to_numpy(float)

    # ---- loss decomposition -------------------------------------------------
    Bc, Bh = S.B_ind.to_numpy(), S.B_heat.to_numpy()
    rows = [{"metric": "B combined (season, 08-22 h)", "mean": wmean(Bc, ws), "p50": wq(Bc, ws, [.5])[0],
             "p90": wq(Bc, ws, [.9])[0]},
            {"metric": "B heat-only (season, 08-22 h)", "mean": wmean(Bh, ws), "p50": wq(Bh, ws, [.5])[0],
             "p90": wq(Bh, ws, [.9])[0]}]
    for lab, col in [("L_peak 1991-2020", "Lpeak_ind"), ("L_peak 2006-2025", "Lpeak_recent")]:
        x = S[col].to_numpy()
        rows.append({"metric": lab, "mean": wmean(x, ws), "p50": wq(x, ws, [.5])[0], "p90": wq(x, ws, [.9])[0],
                     f"share_served_gt_{lthr}": float(ws[x > lthr].sum() / ws.sum())})
    ld = pd.DataFrame(rows)
    ld.loc[len(ld)] = {"metric": "darkness share of combined B (1 - mean B_heat / mean B)",
                       "mean": 1 - wmean(Bh, ws) / wmean(Bc, ws)}
    ld.to_csv(out / "loss_decomposition.csv", index=False)

    # ---- H1 -----------------------------------------------------------------
    rows = []
    for lab, col in [("combined (cover+canopy+surface+lighting)", "B_ind"),
                     ("heat-only (cover+canopy+surface)", "B_heat"), ("naive (regional WBGT only)", "B_naive"),
                     ("peak heat-only L_peak", "Lpeak_heat"), ("peak naive L_peak", "Lpeak_naive")]:
        tot, bshare = var_decomp(S[col].to_numpy(), ws, S.era5_cell.to_numpy())
        rows.append({"model": lab, "mean": wmean(S[col].to_numpy(), ws), "var_total": tot,
                     "share_between_cells": bshare, "share_within_cells": 1 - bshare})
    h1 = pd.DataFrame(rows)
    h1["spearman_vs_naive"] = [spearmanr(S.B_ind, S.B_naive).statistic, spearmanr(S.B_heat, S.B_naive).statistic,
                               1.0, spearmanr(S.Lpeak_heat, S.Lpeak_naive).statistic, 1.0]
    h1["naive_underestimate"] = [np.nan, 1 - wmean(S.B_naive.to_numpy(), ws) / wmean(Bh, ws), np.nan,
                                 1 - wmean(S.Lpeak_naive.to_numpy(), ws) / wmean(S.Lpeak_heat.to_numpy(), ws), np.nan]
    h1.to_csv(out / "h1_decomposition.csv", index=False)

    # ---- deserts summary ----------------------------------------------------
    rows = []
    for (per, kind), col in D.items():
        x = S[col].to_numpy()
        p50, p90 = wq(x, ws, [.5, .9])
        rows.append({"period": per, "kind": kind, "hours_per_served_person": wmean(x, ws), "p50": p50, "p90": p90,
                     f"share_served_gt_{heavy}h": float(ws[x > heavy].sum() / ws.sum()),
                     "share_served_any": float(ws[x > 0].sum() / ws.sum()),
                     "top10pct_share_of_person_hours": top_share(x, ws),
                     "person_hours_M": float((x * ws).sum() / 1e6)})
    pd.DataFrame(rows).to_csv(out / "deserts_summary.csv", index=False)

    # ---- typology (heat-only seasonal loss x indoor fallback) ----------------
    qB = np.quantile(S.B_heat, [1 / 3, 2 / 3])
    qF = np.quantile(S.Fin_ind, [1 / 3, 2 / 3])
    hiB, loB = df.B_heat >= qB[1], df.B_heat <= qB[0]
    hiF, loF = df.Fin_ind >= qF[1], df.Fin_ind <= qF[0]
    df["type"] = "intermediate"
    df.loc[served & loB, "type"] = "low heat loss"
    df.loc[served & hiB & hiF, "type"] = "heat-exposed, indoor fallback"
    df.loc[served & hiB & ~hiF & ~loF, "type"] = "heat-exposed, partial fallback"
    df.loc[served & hiB & loF, "type"] = "compound priority (high loss, low fallback)"
    df.loc[~served, "type"] = "structurally underserved (A0 = 0)"
    typ = df.groupby("type").agg(carreaux=("jid", "size"), population=("ind", "sum"))
    typ["pop_share"] = typ.population / typ.population.sum()
    typ.to_csv(out / "typology.csv")

    # ---- equity -------------------------------------------------------------
    hr, hrc, cr = D[("ref", "heat")], D[("recent", "heat")], D[("ref", "combined")]

    def stats(sub, w):
        sv = sub.has_outdoor.to_numpy()
        wsv = w[sv]
        r = {"population": float(w.sum()), "share_structural": float(w[~sv].sum() / w.sum()) if w.sum() else np.nan}
        if wsv.sum() == 0:
            return r
        g = sub[sv]
        r["B_heat_p50"], r["B_heat_p90"] = wq(g.B_heat.to_numpy(), wsv, [.5, .9])
        r["L_peak_mean"] = wmean(g.Lpeak_ind.to_numpy(), wsv)
        r["L_peak_mean_recent"] = wmean(g.Lpeak_recent.to_numpy(), wsv)
        r[f"share_L_peak_gt_{lthr}"] = float(wsv[g.Lpeak_ind.to_numpy() > lthr].sum() / wsv.sum())
        r[f"share_L_peak_gt_{lthr}_recent"] = float(wsv[g.Lpeak_recent.to_numpy() > lthr].sum() / wsv.sum())
        r["desert_h_heat"] = wmean(g[hr].to_numpy(), wsv)
        r["desert_h_heat_recent"] = wmean(g[hrc].to_numpy(), wsv)
        r["desert_h_combined"] = wmean(g[cr].to_numpy(), wsv)
        r[f"share_desert_gt_{heavy}h"] = float(wsv[g[hr].to_numpy() > heavy].sum() / wsv.sum())
        r[f"share_desert_gt_{heavy}h_recent"] = float(wsv[g[hrc].to_numpy() > heavy].sum() / wsv.sum())
        r["share_compound"] = float(w[(sub.type == "compound priority (high loss, low fallback)").to_numpy()].sum()
                                    / w.sum())
        # cluster bootstrap of served desert hours (heat-only, ref)
        cl = pd.factorize(g.era5_cell.to_numpy())[0]
        N = np.bincount(cl, weights=wsv * g[hr].to_numpy())
        Dn = np.bincount(cl, weights=wsv)
        draws = rng.integers(0, len(N), size=(nb, len(N)))
        vals = N[draws].sum(1) / np.maximum(Dn[draws].sum(1), 1e-9)
        r["desert_h_heat_ci_lo"], r["desert_h_heat_ci_hi"] = np.percentile(vals, [2.5, 97.5])
        return r

    rows = [{"group": g, **stats(df, df[g].to_numpy(float))} for g in groups + ["poor"]]
    pd.DataFrame(rows).to_csv(out / "equity_age.csv", index=False)
    dims = [("q_nivvie", "equity_income_quintile.csv"), ("poverty_class", "equity_poverty.csv"),
            ("density", "equity_density.csv"), ("region", "equity_region.csv")]
    if "qpv" in df:
        dims.append(("qpv", "equity_qpv.csv"))
    for col, fname in dims:
        rows = []
        for k, sub in df.groupby(col, observed=True):
            rows.append({col: k, **stats(sub, sub.ind.to_numpy(float))})
        pd.DataFrame(rows).to_csv(out / fname, index=False)

    # ---- region-standardised equity (direct standardisation) -----------------
    # metric_std(group) = sum_r W_r * metric(group, region r), W_r = national population
    # share of region r (renormalised over regions where the group is present).
    # Removes the part of a social or age gradient that only reflects where groups live.
    def quick(sub, w):
        sv = sub.has_outdoor.to_numpy()
        ws_ = w[sv]
        r = {"share_structural": float(w[~sv].sum() / w.sum()) if w.sum() else np.nan}
        if ws_.sum() > 0:
            g = sub[sv]
            r["L_peak_mean"] = wmean(g.Lpeak_ind.to_numpy(), ws_)
            r[f"share_L_peak_gt_{lthr}"] = float(ws_[g.Lpeak_ind.to_numpy() > lthr].sum() / ws_.sum())
            r["desert_h_heat"] = wmean(g[hr].to_numpy(), ws_)
            r["desert_h_heat_recent"] = wmean(g[hrc].to_numpy(), ws_)
        return r

    Wreg = df.groupby("region").ind.sum()
    Wreg = Wreg / Wreg.sum()
    specs = [(g, None) for g in groups + ["poor"]] + [("q_nivvie", "ind"), ("poverty_class", "ind"),
                                                      ("density", "ind")]
    srows = []
    for key, wcol in specs:
        if wcol is None:                     # age groups / poor: different weights, same carreaux
            levels = [(key, df)]
        else:
            levels = [(k, sub) for k, sub in df.groupby(key, observed=True)]
        for lev, sub in levels:
            wname = key if wcol is None else wcol
            raw = quick(sub, sub[wname].to_numpy(float))
            per = {}
            for reg, sr in sub.groupby("region"):
                w = sr[wname].to_numpy(float)
                if w.sum() > 0:
                    per[reg] = quick(sr, w)
            regs = [r_ for r_ in per if r_ in Wreg.index]
            wr = Wreg.loc[regs] / Wreg.loc[regs].sum()
            for m in raw:
                vals = np.array([per[r_].get(m, np.nan) for r_ in regs], float)
                ok = np.isfinite(vals)
                std = float((vals[ok] * wr.to_numpy()[ok]).sum() / wr.to_numpy()[ok].sum()) if ok.any() else np.nan
                srows.append({"dimension": "age_group" if wcol is None else key,
                              "level": lev, "metric": m, "raw": raw[m], "region_standardised": std})
    pd.DataFrame(srows).to_csv(out / "equity_standardised.csv", index=False)

    # ---- cluster bootstrap of region-standardised gaps ------------------------
    # Resample ERA5-Land cells (same draw for every level, so gaps are paired);
    # region means recomputed from cluster sums, then direct standardisation.
    Sv = df[served].copy()
    cl, cells_u = pd.factorize(Sv.era5_cell.to_numpy())
    reg_of_cell = pd.Series(Sv.region.to_numpy()).groupby(cl).agg(lambda x: x.value_counts().index[0])
    regs_all = sorted(Wreg.index)
    ridx = pd.Series(np.arange(len(regs_all)), index=regs_all)
    Rc = ridx.reindex(reg_of_cell.to_numpy()).to_numpy()
    okc = np.isfinite(Rc)
    Rc = np.where(okc, Rc, 0).astype(int)
    nC, nR = len(cells_u), len(regs_all)
    draws = np.stack([np.bincount(rng.integers(0, nC, nC), minlength=nC) for _ in range(nb)])   # nb x nC
    onehot = np.zeros((nC, nR))
    onehot[np.arange(nC)[okc], Rc[okc]] = 1.0
    Wv = Wreg.reindex(regs_all).to_numpy()

    def std_draws(sub, wcol, ycol):
        idx = cl[sub.index.map(pd.Series(np.arange(len(Sv)), index=Sv.index)).to_numpy()]
        w = sub[wcol].to_numpy(float)
        N = np.bincount(idx, weights=w * sub[ycol].to_numpy(float), minlength=nC)
        D = np.bincount(idx, weights=w, minlength=nC)
        RN = draws @ (onehot * N[:, None])          # nb x nR
        RD = draws @ (onehot * D[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            M = RN / RD
        ok = np.isfinite(M)
        return (np.where(ok, M, 0) * Wv).sum(1) / (ok * Wv).sum(1)

    brows = []
    for key, ref_lev, wcol in [("q_nivvie", 1, "ind"), ("poverty_class", labs[0], "ind")]:
        base = None
        for lev, sub in Sv.groupby(key, observed=True):
            for ycol, mname in [(hr, "desert_h_heat"), ("Lpeak_ind", "L_peak_mean")]:
                v = std_draws(sub, wcol, ycol)
                brows.append({"dimension": key, "level": lev, "metric": mname, "ci_lo": np.percentile(v, 2.5),
                              "ci_hi": np.percentile(v, 97.5), "_draws": v})
    bt = pd.DataFrame(brows)
    gaps = []
    for (dim, m), g in bt.groupby(["dimension", "metric"]):
        refl = 1 if dim == "q_nivvie" else labs[0]
        r0 = g[g.level == refl]["_draws"].iloc[0]
        for _, row in g.iterrows():
            dlt = row["_draws"] - r0
            gaps.append({"dimension": dim, "level": row.level, "metric": m, "vs": refl,
                         "gap_ci_lo": np.percentile(dlt, 2.5), "gap_ci_hi": np.percentile(dlt, 97.5),
                         "p_gap_le_0": float((dlt <= 0).mean())})
    bt.drop(columns="_draws").merge(pd.DataFrame(gaps), on=["dimension", "level", "metric"]).to_csv(
        out / "equity_standardised_ci.csv", index=False)

    # ---- decomposition of the region-standardised income gap ------------------
    # Weighted OLS on served carreaux: desert_h_heat ~ quintile + region FE (+ supply terms).
    # Supply terms within the main catchment (public outdoor, tau <= tau0): log number of
    # facilities (redundancy), mean peak availability (thermal exposure), composition shares.
    if a.od:
        t0 = cfg["network"]["tau0_min"][0]
        od = pd.read_parquet(a.od, columns=["jid", "fid", "layer", "tau_min"])
        od = od[(od.layer == "outdoor") & (od.tau_min <= t0)]
        fh = pd.read_parquet(a.fac_heat, columns=["fid", "abar_peak"])
        od = od.merge(fh, on="fid").merge(fac[["fid", "family", "surface_class"]], on="fid")
        od["tennis"] = od.family.eq("Court de tennis")
        od["citystade"] = od.family.eq("Multisports/City-stades")
        od["boules"] = od.family.eq("Boulodrome")
        od["syn_min"] = od.surface_class.isin(["synthetic", "mineral"])
        sup = od.groupby("jid").agg(n_fac=("fid", "size"), abar_peak_mean=("abar_peak", "mean"),
                                    tennis=("tennis", "mean"), citystade=("citystade", "mean"),
                                    boules=("boules", "mean"), syn_min=("syn_min", "mean"))
        dd = Sv.merge(sup, left_on="jid", right_index=True, how="inner")
        dd = dd[dd.ind > 0]
        dd["log_n"] = np.log(dd.n_fac)
        X0 = pd.get_dummies(dd[["q_nivvie", "region"]].astype(str), drop_first=True).astype(float)
        wsq = np.sqrt(dd.ind.to_numpy(float))
        y = dd[hr].to_numpy(float)
        drows = []
        for lab_, extra in [("region FE", []), ("+ log n facilities", ["log_n"]),
                            ("+ mean peak availability", ["log_n", "abar_peak_mean"]),
                            ("+ composition", ["log_n", "abar_peak_mean", "tennis", "citystade", "boules", "syn_min"])]:
            Xm = pd.concat([X0, dd[extra]], axis=1) if extra else X0.copy()
            Xm.insert(0, "const", 1.0)
            b, *_ = np.linalg.lstsq(Xm.to_numpy() * wsq[:, None], y * wsq, rcond=None)
            bs = pd.Series(b, index=Xm.columns)
            r = {"model": lab_, **{f"Q{c[-1]}_vs_Q1": float(bs[c]) for c in bs.index if c.startswith("q_nivvie_")}}
            r.update({f"coef_{e}": float(bs[e]) for e in extra})
            drows.append(r)
        comp = []
        for q_, s_ in dd.groupby("q_nivvie"):
            per = s_.groupby("region").apply(lambda r_: pd.Series(
                {c: np.average(r_[c], weights=r_.ind) for c in ["n_fac", "abar_peak_mean", "tennis", "citystade",
                                                                 "boules", "syn_min", hr]}), include_groups=False)
            wr_ = Wreg.reindex(per.index)
            comp.append({"q_nivvie": q_, **(per.mul(wr_, axis=0).sum() / wr_.sum()).to_dict()})
        pd.DataFrame(drows).to_csv(out / "equity_decomposition.csv", index=False)
        pd.DataFrame(comp).to_csv(out / "equity_supply_composition.csv", index=False)

    # ---- static vs heat -----------------------------------------------------
    Apk = S.A0_ind * (1 - S.Lpeak_ind)
    q0 = pd.qcut(S.A0_ind.rank(method="first"), 5, labels=False) + 1
    qh = pd.qcut(Apk.rank(method="first"), 5, labels=False) + 1
    pd.crosstab(q0, qh, values=S.ind, aggfunc="sum", normalize=True).to_csv(
        out / "static_vs_heat_quintile_transitions.csv")
    pd.DataFrame([{"spearman_A0_Aheat_season": spearmanr(S.A0_ind, S.Aheat_heat).statistic,
                   "spearman_A0_Apeak": spearmanr(S.A0_ind, Apk).statistic,
                   "pop_share_down_1_quintile_peak": float(S.ind[qh < q0].sum() / S.ind.sum()),
                   "pop_share_down_2_quintiles_peak": float(S.ind[qh <= q0 - 2].sum() / S.ind.sum())}]).to_csv(
        out / "static_vs_heat.csv", index=False)

    # ---- facilities: heat-only availability by attribute -----------------------
    fa = pd.read_parquet(a.fac_heat).merge(fac, on="fid")
    frows = []
    for col in ["surface_class", "covered", "lit", "family", "region", "density", "qpv", "qpv200"]:
        for k, g in fa.groupby(col, observed=True):
            frows.append({"attribute": col, "value": k, "n": len(g), "abar_season": g.abar_mean.mean(),
                          "abar_peak": g.abar_peak.mean(), "share_abar_peak_lt_0.8": float((g.abar_peak < .8).mean())})
    pd.DataFrame(frows).to_csv(out / "facility_availability.csv", index=False)

    # ---- sensitivities --------------------------------------------------------
    def srow(lab, x, ref):
        xs = x[x.has_outdoor]
        w = xs.ind.to_numpy(float)
        mm = xs[["jid", "B_ind", "Lpeak_ind"]].merge(ref, on="jid", suffixes=("", "_main"))
        return {"run": lab, "B_p50": wq(xs.B_ind.to_numpy(), w, [.5])[0],
                "L_peak_mean": wmean(xs.Lpeak_ind.to_numpy(), w),
                f"share_L_peak_gt_{lthr}": float(w[xs.Lpeak_ind.to_numpy() > lthr].sum() / w.sum()),
                "spearman_B_vs_main": spearmanr(mm.B_ind, mm.B_ind_main).statistic,
                "spearman_L_peak_vs_main": spearmanr(mm.Lpeak_ind, mm.Lpeak_ind_main).statistic}
    ref = S[["jid", "B_ind", "Lpeak_ind"]]
    rows = [srow("main", df[["jid", "ind", "B_ind", "Lpeak_ind", "has_outdoor"]], ref)]
    for item in a.sens:
        lab, path = item.split("=", 1)
        x = pd.read_parquet(path, columns=["jid", "B_ind", "Lpeak_ind", "has_outdoor"]).merge(
            pop[["jid", "ind"]], on="jid")
        rows.append(srow(lab, x, ref))
    pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)

    keep = ["jid", "dep", "region", "era5_cell", "density", "poverty_class", "type", "B_heat", "Lpeak_ind",
            "Lpeak_recent", hr, hrc, cr] + (["qpv"] if "qpv" in df else [])
    df[keep].to_parquet(out / "carreaux_metrics.parquet", index=False)
    print(ld.to_string())
    print(h1.to_string())


if __name__ == "__main__":
    main()
