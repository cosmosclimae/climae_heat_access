#!/usr/bin/env python3
"""Paper figures (Fig. 2-7) and compiled tables from pipeline outputs.

Fig 2  supply composition by region (surface classes, lit, public, canopy)
Fig 3  hourly heat-only availability, July: by region (small multiples) and by attribute
Fig 4  H1: naive vs modelled heat-only loss (season B_heat, peak L_peak)
Fig 5  national map of peak loss L_peak (2-km pop-weighted) + zooms Montpellier, Paris, Lyon, Marseille
Fig 6  heat-only temporary deserts: person-hours by month x hour; hours per served person by
       region (1991-2020 vs 2006-2025) and by density class
Fig 7  bivariate map: heat-only loss B_heat x indoor fallback (terciles)
tables.md : all result tables (English).
All metrics are heat-only (analysis.heat_scenario): lighting is a supply constraint reported apart.
Palette: reference dataviz instance (blue sequential, categorical slots in fixed order).
"""
import argparse
import calendar
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap   # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)
BIV = ["#e8e8e8", "#ace4e4", "#5ac8c8", "#dfb0d6", "#a5add3", "#5698b9", "#be64ac", "#8c62aa", "#3b4994"]
CITIES = {"Montpellier": (3.877, 43.611), "Paris": (2.349, 48.857), "Lyon": (4.835, 45.764), "Marseille": (5.37, 43.296)}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def wavg(x, w):
    m = np.isfinite(x) & (w > 0)
    return np.nan if not m.any() else float(np.average(x[m], weights=w[m]))


def raster(df, val, w, res, bounds=None):
    """Pop-weighted mean of `val` on a res-m grid (EPSG:3035 centroids)."""
    x, y = df.x3035.to_numpy(), df.y3035.to_numpy()
    if bounds is None:
        bounds = (x.min() - res, y.min() - res, x.max() + res, y.max() + res)
    x0, y0, x1, y1 = bounds
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    ix = ((x - x0) // res).astype(int)
    iy = ((y1 - y) // res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(val) & (w > 0)
    num = np.zeros((ny, nx))
    den = np.zeros((ny, nx))
    np.add.at(num, (iy[ok], ix[ok]), val[ok] * w[ok])
    np.add.at(den, (iy[ok], ix[ok]), w[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out, (x0, x1, y0, y1)


def fig2(fac, can, out):
    o = fac[fac.layer == "outdoor"].merge(can, on="fid", how="left")
    reg = o.groupby("region")
    classes = ["synthetic", "bitume", "mineral", "grass", "reference"]
    labels = {"synthetic": "Synthetic", "bitume": "Asphalt", "mineral": "Mineral / stabilised",
              "grass": "Natural grass", "reference": "Not reported"}
    comp = pd.crosstab(o.region, o.surface_class, normalize="index").reindex(columns=classes, fill_value=0)
    comp = comp.loc[reg.size().sort_values().index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.4, 1]}, sharey=True)
    left = np.zeros(len(comp))
    for i, c in enumerate(classes):
        axes[0].barh(comp.index, comp[c], left=left, color=CAT[i] if c != "reference" else GRAY,
                     edgecolor="white", linewidth=1, label=labels[c])
        left += comp[c].to_numpy()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Share of outdoor facilities")
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    cc = [c for c in o.columns if str(c).startswith("canopy_r")][0]
    stats = pd.DataFrame({"Lit": reg.lit.mean(), "Public access": reg.access_public.mean(),
                          "Mean canopy": reg[cc].mean()}).loc[comp.index]
    for i, c in enumerate(stats.columns):
        axes[1].plot(stats[c], stats.index, "o", ms=5, color=CAT[i], label=c)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share / fraction")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    n = reg.size().loc[comp.index]
    for yv, (r, v) in enumerate(n.items()):
        axes[0].text(1.01, yv, f"{v:,}".replace(",", " "), va="center", fontsize=6, color=INK2)
    fig.savefig(out / "fig2_supply_composition.png")
    plt.close(fig)


def fig3(fac, can, clim, q, scen, out):
    z = np.load(clim)
    s = list(z["scens"]).index(scen) if scen in list(z["scens"]) else 0
    qi = list(z["qs"]).index(q)
    months, hours = list(z["months"]), list(z["hours"])
    jul = months.index(7) if 7 in months else len(months) // 2
    hcols = [f"h{h}" for h in hours]
    ab = pd.DataFrame(z["abar"][s, qi][:, jul, :], columns=hcols)
    ab["fid"] = z["fid"]
    o = fac[fac.layer == "outdoor"].merge(can, on="fid", how="left").merge(ab, on="fid")
    regions = o.groupby("region").size().sort_values(ascending=False).index.tolist()
    nat = o[hcols].mean()
    ncol = 5
    nrow = int(np.ceil((len(regions) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.45 * nrow + 0.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, r in zip(axes, regions):
        ax.plot(hours, nat, color=GRAY, lw=1, label="France")
        ax.plot(hours, o.loc[o.region == r, hcols].mean(), color=CAT[0], lw=2, label=r)
        ax.set_title(r, fontsize=7, color=INK)
        ax.set_ylim(0, 1.02)
    cc = [c for c in o.columns if str(c).startswith("canopy_r")][0]
    ax = axes[len(regions)]
    terc = pd.qcut(o[cc].rank(method="first"), 3, labels=["low", "medium", "high"])
    for i, t in enumerate(["low", "medium", "high"]):
        ax.plot(hours, o.loc[terc == t, hcols].mean(), color=CAT[i], lw=2, label=f"{t} canopy")
    ax.set_title("By canopy tercile", fontsize=7)
    ax.legend(fontsize=5.5)
    ax = axes[len(regions) + 1]
    for i, (cl, lab) in enumerate([("synthetic", "synthetic"), ("mineral", "mineral"), ("grass", "grass")]):
        ax.plot(hours, o.loc[o.surface_class == cl, hcols].mean(), color=CAT[i], lw=2, label=lab)
    ax.set_title("By surface", fontsize=7)
    ax.legend(fontsize=5.5)
    for ax in axes[len(regions) + 2:]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("ā (July)")
    for ax in axes[-ncol:]:
        ax.set_xlabel("Local hour")
    fig.suptitle(f"Mean heat-only availability of outdoor facilities, July 1991-2020, WBGT < {q} °C",
                 fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_hourly_availability.png")
    plt.close(fig)


def fig4(m, out):
    """H1: naive (regional WBGT only) vs modelled heat-only loss, season and peak."""
    d = m[m.has_outdoor]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (xn, yn, lab) in zip(axes, [("B_naive", "B_heat", "seasonal loss B"),
                                         ("Lpeak_naive", "Lpeak_heat", "peak loss L_peak (July 14-17 h)")]):
        hb = ax.hexbin(d[xn], d[yn], C=d.ind, reduce_C_function=np.sum, gridsize=60, cmap=SEQ_CMAP,
                       bins="log", mincnt=1)
        mx = float(np.nanpercentile(np.r_[d[xn], d[yn]], 99.5))
        ax.plot([0, mx], [0, mx], color=GRAY, lw=1)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel(f"Naive {lab}")
        ax.set_ylabel(f"Modelled {lab}")
        fig.colorbar(hb, ax=ax, label="Population")
    fig.tight_layout()
    fig.savefig(out / "fig4_h1_naive_vs_modelled.png")
    plt.close(fig)


STRUCT_GREY = "#cfcac0"


def struct_layer(m, res, bounds):
    """Inhabited cells with no facility in reach (A0 = 0): flat grey layer."""
    s = m[(~m.has_outdoor) & (m.ind > 0)]
    img, _ = raster(s, np.ones(len(s)), s.ind.to_numpy(), res, bounds=bounds)
    return np.where(np.isfinite(img), 1.0, np.nan)


def fig5(m, dep3035, out):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    d = m[m.has_outdoor]
    vmax = float(np.nanpercentile(d.Lpeak_heat, 98))
    inh = m[m.ind > 0]
    bnat = (inh.x3035.min() - 2000, inh.y3035.min() - 2000, inh.x3035.max() + 2000, inh.y3035.max() + 2000)
    img, ext = raster(d, d.Lpeak_heat.to_numpy(), d.ind.to_numpy(), 2000, bounds=bnat)
    grey = ListedColormap([STRUCT_GREY])
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
    ax.imshow(struct_layer(m, 2000, bnat), extent=ext, cmap=grey, interpolation="nearest")
    ax.imshow(img, extent=ext, cmap=SEQ_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    ax.legend(handles=[Patch(color=STRUCT_GREY, label="No facility within 15 min"),
                       Patch(facecolor="white", edgecolor=INK2, lw=0.4, label="Uninhabited")],
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6, frameon=False)
    dep3035.boundary.plot(ax=ax, color=INK2, lw=0.2)
    ax.set_axis_off()
    ax.set_title("Peak heat loss of potential accessibility, July 14-17 h, 1991-2020", fontsize=8)
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    for i, (name, (lo, la)) in enumerate(CITIES.items()):
        cx, cy = tr.transform(lo, la)
        b = (cx - 15000, cy - 15000, cx + 15000, cy + 15000)
        zi, ze = raster(d, d.Lpeak_heat.to_numpy(), d.ind.to_numpy(), 200, bounds=b)
        zax = fig.add_axes([0.64 + (i % 2) * 0.18, 0.52 - (i // 2) * 0.46, 0.17, 0.4])
        zax.imshow(struct_layer(m, 200, b), extent=ze, cmap=grey, interpolation="nearest")
        zax.imshow(zi, extent=ze, cmap=SEQ_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        dep3035.boundary.plot(ax=zax, color=INK2, lw=0.3)
        zax.set_xlim(b[0], b[2]); zax.set_ylim(b[1], b[3])
        zax.set_axis_off()
        zax.set_title(name, fontsize=7)
        ax.plot([b[0], b[2], b[2], b[0], b[0]], [b[1], b[1], b[3], b[3], b[1]], color=INK, lw=0.5)
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.08, 0.06, 0.4, 0.02])
    fig.colorbar(sm, cax=cax, orientation="horizontal", label="L_peak (fraction of accessibility lost)")
    fig.savefig(out / "fig5_map_Lpeak.png")
    plt.close(fig)


def fig6(des_path, m, eq_region, eq_density, out):
    des = pd.read_parquet(des_path).merge(m.loc[m.has_outdoor, ["jid", "ind"]], on="jid")
    des["dim"] = des.month.map(lambda x: calendar.monthrange(2001, x)[1])
    des["ph"] = des.n_desert / des.n_days * des.dim * des.ind
    mh = des.pivot_table(index="month", columns="hour", values="ph", aggfunc="sum").fillna(0) / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    ax = axes[0]
    im = ax.imshow(mh.to_numpy(), aspect="auto", cmap=SEQ_CMAP, vmin=0)
    ax.set_yticks(range(len(mh.index)), [calendar.month_abbr[x] for x in mh.index])
    ax.set_xticks(range(0, len(mh.columns), 3), [f"{h}h" for h in mh.columns[::3]])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="M person-hours / season")
    ax.set_title("Heat-only deserts, 1991-2020", fontsize=8)
    r = pd.read_csv(eq_region).sort_values("desert_h_heat")
    y = np.arange(len(r))
    axes[1].barh(y + 0.2, r.desert_h_heat, height=0.4, color=SEQ[2], label="1991-2020")
    axes[1].barh(y - 0.2, r.desert_h_heat_recent, height=0.4, color=SEQ[5], label="2006-2025")
    axes[1].set_yticks(y, r.region, fontsize=6)
    axes[1].set_xlabel("Hours / served person / season")
    axes[1].legend(fontsize=6, loc="lower right")
    axes[1].set_title("By region", fontsize=8)
    order = ["Grands centres urbains", "Centres urbains intermédiaires", "Ceintures urbaines", "Petites villes",
             "Bourgs ruraux", "Rural à habitat dispersé", "Rural à habitat très dispersé"]
    en = {"Grands centres urbains": "Large urban centres", "Centres urbains intermédiaires": "Intermediate centres",
          "Ceintures urbaines": "Urban belts", "Petites villes": "Small towns", "Bourgs ruraux": "Rural towns",
          "Rural à habitat dispersé": "Dispersed rural", "Rural à habitat très dispersé": "Very dispersed rural"}
    dd = pd.read_csv(eq_density).set_index("density").reindex(order).dropna(how="all")
    axes[2].bar(range(len(dd)), dd.desert_h_heat, color=CAT[0], width=0.6)
    axes[2].errorbar(range(len(dd)), dd.desert_h_heat, yerr=[dd.desert_h_heat - dd.desert_h_heat_ci_lo,
                                                            dd.desert_h_heat_ci_hi - dd.desert_h_heat],
                     fmt="none", ecolor=INK2, lw=0.8)
    axes[2].set_xticks(range(len(dd)), [en[x] for x in dd.index], rotation=45, ha="right", fontsize=6)
    axes[2].set_ylabel("Hours / served person / season")
    axes[2].set_title("By municipal density", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_deserts.png")
    plt.close(fig)


def fig7(m, dep3035, out):
    d = m[m.has_outdoor].copy()
    tb = pd.qcut(d.B_heat.rank(method="first"), 3, labels=False)
    tf = pd.qcut(d.Fin_ind.rank(method="first"), 3, labels=False)
    d["cls"] = (tb * 3 + (2 - tf)).astype(float)      # rows: loss low->high ; cols: fallback high->low
    img, ext = raster(d, d.cls.to_numpy(), d.ind.to_numpy(), 2000)
    img = np.round(img)
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_axes([0, 0, 0.8, 1])
    ax.imshow(img, extent=ext, cmap=ListedColormap(BIV), vmin=-0.5, vmax=8.5, interpolation="nearest")
    dep3035.boundary.plot(ax=ax, color=INK2, lw=0.2)
    ax.set_axis_off()
    ax.set_title("Heat-only outdoor loss B_heat x indoor fallback (terciles, served carreaux)", fontsize=8)
    lax = fig.add_axes([0.8, 0.08, 0.18, 0.18])
    lax.imshow(np.arange(9).reshape(3, 3)[::-1], cmap=ListedColormap(BIV), vmin=-0.5, vmax=8.5)
    lax.set_xticks([0, 2], ["high\nfallback", "low\nfallback"], fontsize=6)
    lax.set_yticks([0, 2], ["high\nloss", "low\nloss"], fontsize=6)
    lax.grid(False)
    lax.text(2, 0, "★", ha="center", va="center", fontsize=8, color="white")
    fig.text(0.8, 0.27, "★ = compound priority", fontsize=6, color=INK2)
    fig.savefig(out / "fig7_bivariate.png")
    plt.close(fig)


TABLES = [
    ("loss_decomposition.csv", "Table 3 — Loss decomposition (combined vs heat-only; peak loss by period)"),
    ("h1_decomposition.csv", "Table 4 — H1: variance decomposition between / within ERA5-Land cells"),
    ("deserts_summary.csv", "Table 5 — Temporary deserts per served person (May-Sep, 08-22 h)"),
    ("equity_age.csv", "Table 6a — Equity by age group (and persons in poor households)"),
    ("equity_income_quintile.csv", "Table 6b — Equity by standard-of-living quintile"),
    ("equity_poverty.csv", "Table 6c — Equity by household poverty rate"),
    ("equity_density.csv", "Table 6d — Equity by municipal density class"),
    ("equity_region.csv", "Table 6e — Equity by region"),
    ("equity_qpv.csv", "Table 6f — Equity by priority neighbourhood (QPV)"),
    ("static_vs_heat.csv", "Table 7 — Static vs heat-constrained rankings"),
    ("typology.csv", "Table 8 — Territorial typology"),
    ("facility_availability.csv", "Table S1 — Heat-only facility availability by attribute"),
    ("sensitivity.csv", "Table S2 — Sensitivity analyses"),
    ("sens_3sfca.csv", "Table S3 — Heat-modulated 3SFCA vs E2SFCA (8 metropolitan areas, total population)"),
    ("../adaptation/strategies_budgets.csv", "Table 9 — Shading strategies at budget levels"),
]


def tables(res, out, fac_report):
    lines = ["# Tables — climae_heat_access", ""]
    if fac_report and Path(fac_report).exists():
        r = yaml.safe_load(open(fac_report, encoding="utf-8"))
        lines += ["## Table 2 — Data ES supply retained", "", "| Item | Value |", "|---|---|"]
        for k in ["rows_raw", "outdoor", "indoor", "excluded_type_or_nature", "dropped_hors_service",
                  "outdoor_covered_share", "outdoor_lit_share", "outdoor_public_share"]:
            if k in r:
                lines.append(f"| {k} | {r[k]} |")
        lines.append("")
    for name, title in TABLES:
        p = res / name
        if p.exists():
            df = pd.read_csv(p)
            if name == "sens_3sfca.csv" and "group" in df:
                df = df[df.group == "ind"]
            if name.endswith("strategies_budgets.csv"):
                keep = ["budget_pct", "k", "strategy"] + [c for c in df.columns if c.startswith("avoided_share")]
                df = df[df.strategy.isin(["coverage", "equity_low_income", "population", "heat"])][keep]
            num = df.select_dtypes("number").columns
            df[num] = df[num].round(3)
            lines += [f"## {title}", "", df.to_markdown(index=False), ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--departements", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    W = Path(a.work)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = cfg["climate"]["main_threshold"]
    fac = pd.read_parquet(W / "data/facilities.parquet")
    fac["region"] = fac["region"].fillna(fac["dep"].map(
        fac.dropna(subset=["region"]).groupby("dep").region.agg(lambda s: s.value_counts().index[0])))
    can = pd.read_parquet(W / "data/facility_canopy.parquet")
    pop = pd.read_parquet(W / "data/population.parquet")
    hs = cfg.get("analysis", {}).get("heat_scenario", "nolight")
    cols = ["jid", "A0_ind", "B_ind", "Lpeak_ind", "Fin_ind", "has_outdoor"]
    m = pop[["jid", "ind", "x3035", "y3035"]].merge(
        pd.read_parquet(W / f"access/scen_{hs}/access.parquet", columns=cols).rename(
            columns={"B_ind": "B_heat", "Lpeak_ind": "Lpeak_heat"}), on="jid").merge(
        pd.read_parquet(W / "access/naive/access.parquet", columns=["jid", "B_ind", "Lpeak_ind"]).rename(
            columns={"B_ind": "B_naive", "Lpeak_ind": "Lpeak_naive"}), on="jid")
    dep = gpd.read_file(a.departements).to_crs(3035)
    res = W / "results/analysis"
    fig2(fac, can, out)
    fig3(fac, can, W / "clim/abar_ref.npz", q, hs, out)
    fig4(m, out)
    fig5(m, dep, out)
    fig6(W / f"clim/deserts_ref/deserts_{hs}_q{q}.parquet", m, res / "equity_region.csv",
         res / "equity_density.csv", out)
    fig7(m, dep, out)
    tables(res, out, W / "data/facilities_report.yaml")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
