# -*- coding: utf-8 -*-
# Snakefile - Heat-constrained potential accessibility to outdoor sport facilities (France)
# Full replay of the paper from the raw inputs listed in config/config.yaml.
#
# Workflow:
#   Data ES ─► facilities ─► canopy ─┐
#   ERA5-Land ─► grid ───────────────┼─► wbgt_month (vectorised Liljegren WBGT) ─► gonogo (climate screen)
#   OSM pbf ─► osm_extract[dep] ─► buildings[dep] ─► population (Filosofi 2021, dasymetric disaggregation)
#                               └─► network[dep] (pandana) ─► catchment
#   wbgt_month + catchment ─► availability_month ─► climatology[period]
#   climatology ─► accessibility[run] ─► analysis (H1, equity, typology, sensitivity runs)
#   wbgt_month ─► adaptation_sample ─► adaptation_dep[dep] ─► adaptation_merge (shading priorities)
#   review: out-of-sample shading evaluation, alternative desert definitions, diagnostics
#
# Examples:
#   snakemake -n                                     # dry run
#   snakemake -j 8 --resources cds=4 gonogo          # climate screen; cds = simultaneous CDS requests
#   snakemake -j 8                                   # everything
#   snakemake -j 8 --config work_root=/data/xxx      # other working disk
#   snakemake -j 4 --config deps='["34","30","13"]'  # subset of départements (test)
#
import json
from pathlib import Path

configfile: "config/config.yaml"

W = Path(config["work_root"])
I = config["inputs"]
CFG = "config/config.yaml"
PY = "python"

# ---------------------------------------------------------------- départements
_geo = json.load(open(I["departements_geojson"], encoding="utf-8"))
def _bbox(geom):
    xs, ys = [], []
    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for x in c: walk(x)
    walk(geom["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)
BBOX = {f["properties"]["code"]: _bbox(f["geometry"]) for f in _geo["features"]}
DEPS = config.get("deps", sorted(BBOX))
DEPS = json.loads(DEPS) if isinstance(DEPS, str) else DEPS

# ---------------------------------------------------------------- calendar
P = config["climate"]["periods"]
MONTHS = [f"{m:02d}" for m in config["climate"]["months"]]
YEARS = sorted({y for a, b in P.values() for y in range(a, b + 1)})
YM = [(str(y), m) for y in YEARS for m in MONTHS]
def ym_of(period, months=None):
    a, b = P[period]
    ms = MONTHS if months is None else [f"{m:02d}" for m in months]
    return [(str(y), m) for y in range(a, b + 1) for m in ms]
# ERA5-Land: local file if it exists, otherwise a light profile downloaded by rule era5_download
def ERA5(y, m):
    dash = Path(I["era5land_dir"]) / I["era5land_pattern"].format(year=y, month=m)
    return str(dash) if dash.exists() else str(W / f"era5land/era5land_heat_{y}-{m}.nc")
GP = config["climate"].get("gonogo_period", "ref")
TEMPLATE = ERA5(str(P[GP][0]), MONTHS[0])

# ---------------------------------------------------------------- accessibility runs
QM = config["climate"]["main_threshold"]
T0 = config["network"]["tau0_min"]
MAIN = dict(period="ref", scen=config["modifiers"]["main"], q=QM, tau0=T0[0],
            access=config["access"]["main"], cap="unit", avail="binary")
RUNS = {"main": MAIN, "naive": {**MAIN, "scen": "naive"}}
for s in [k for k in config["modifiers"] if k not in ("main", MAIN["scen"], "naive")]:
    RUNS[f"scen_{s}"] = {**MAIN, "scen": s}
for q in config["climate"]["thresholds"]:
    if q != QM: RUNS[f"q{q}"] = {**MAIN, "q": q}
for t in T0[1:]:
    RUNS[f"tau{t}"] = {**MAIN, "tau0": t}
for acc in config["access"]["scenarios"]:
    if acc != MAIN["access"]: RUNS[f"access_{acc}"] = {**MAIN, "access": acc}
RUNS["cap_surface"] = {**MAIN, "cap": "surface"}
RUNS["graded"] = {**MAIN, "avail": "graded"}
for p in P:
    if p != "ref": RUNS[f"period_{p}"] = {**MAIN, "period": p}
SENS = [r for r in RUNS if r not in ("main", "naive") and not r.startswith("period_urb")]

wildcard_constraints:
    year=r"\d{4}", month=r"\d{2}", dep=r"[0-9AB]{2,3}", period=r"[a-z]+", run=r"[a-z0-9_]+"

rule all:
    input:
        W / "results/gonogo/verdict.yaml",
        W / "results/analysis/h1_decomposition.csv",
        W / "results/analysis/sens_3sfca.csv",
        W / "results/adaptation/strategies_budgets.csv",
        W / "results/figures/tables.md",
        W / "results/adaptation/out_of_sample.csv",
        W / "results/analysis/robustness_deserts.csv",
        W / "results/analysis/diag_shapley.csv"
# The urban heat island sensitivity (Table S7) needs UrbClim data and is run separately:
#   snakemake -j 4 <work_root>/results/analysis/uhi_sensitivity.csv

# ======================================================================= supply
rule facilities:
    input: es=I["data_es_zip"], dep=I["departements_geojson"]
    output: fac=W / "data/facilities.parquet", rep=W / "data/facilities_report.yaml"
    shell: "{PY} scripts/prepare_facilities.py --config {CFG} --data-es {input.es} --departements {input.dep} "
           "--out {output.fac} --report {output.rep}"

rule canopy:
    input: W / "data/facilities.parquet"
    output: W / "data/facility_canopy.parquet"
    params: r=I["canopy_raster"], b=" ".join(str(x) for x in config["facilities"]["canopy_buffer_m"])
    shell: "{PY} scripts/facility_canopy.py --facilities {input} --raster '{params.r}' --buffers {params.b} --out {output}"

rule grid:
    input: tpl=TEMPLATE, fac=W / "data/facilities.parquet", dep=I["departements_geojson"]
    output: cells=W / "data/cells.parquet", fc=W / "data/facility_cells.parquet", wc=W / "data/wbgt_cells.parquet"
    shell: "{PY} scripts/grid_assign.py --template {input.tpl} --facilities {input.fac} --departements {input.dep} "
           "--cells-out {output.cells} --fac-out {output.fc} --wbgt-cells-out {output.wc}"

# ======================================================================= OSM / population / network
rule osm_extract:
    input: I["osm_pbf"]
    output: W / "osm/dep_{dep}.osm.pbf"
    params:
        bb=lambda w: ",".join(f"{v:.4f}" for v in (BBOX[w.dep][0] - 0.06, BBOX[w.dep][1] - 0.04,
                                                    BBOX[w.dep][2] + 0.06, BBOX[w.dep][3] + 0.04))
    resources: osm_slots=1
    shell: "osmium extract -b {params.bb} --overwrite -o {output} {input}"

rule buildings:
    input: W / "osm/dep_{dep}.osm.pbf"
    output: W / "osm/buildings_{dep}.parquet"
    resources: mem_mb=12000
    shell: "{PY} scripts/buildings_dep.py --config {CFG} --pbf {input} --out {output}"

rule population:
    input:
        gpkg=I["filosofi_200m_gpkg"] if config["population"]["mode"] == "native_200m" else I["filosofi_gpkg"],
        bld=expand(W / "osm/buildings_{dep}.parquet", dep=DEPS) if config["population"]["mode"] != "native_200m" else [],
        dep=I["departements_geojson"]
    output: popf=W / "data/population.parquet", rep=W / "data/population_report.yaml"
    shell: "{PY} scripts/prepare_population.py --config {CFG} --gpkg {input.gpkg} --buildings {input.bld} "
           "--departements {input.dep} --out {output.popf} --report {output.rep}"

rule network:
    input: pbf=W / "osm/dep_{dep}.osm.pbf", popf=W / "data/population.parquet", fac=W / "data/facilities.parquet"
    output: od=W / "od/od_{dep}.parquet", log=W / "od/od_{dep}.yaml"
    threads: 2
    resources: mem_mb=16000          # pyrosm + pandana on a large département; --resources mem_mb=... limits parallelism
    shell: "{PY} scripts/network_dep.py --config {CFG} --pbf {input.pbf} --population {input.popf} "
           "--facilities {input.fac} --dep {wildcards.dep} --out {output.od} --log {output.log}"

rule catchment:
    input: od=expand(W / "od/od_{dep}.parquet", dep=DEPS), popf=W / "data/population.parquet",
           fac=W / "data/facilities.parquet"
    output: od=W / "data/od.parquet", b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet"
    shell: "{PY} scripts/catchment.py --config {CFG} --od {input.od} --population {input.popf} --facilities {input.fac} "
           "--od-out {output.od} --b-out {output.b} --meta-out {output.meta}"

# ======================================================================= climate
rule era5_download:
    output: W / "era5land/era5land_heat_{year}-{month}.nc"
    resources: cds=1
    retries: 2
    shell: "{PY} scripts/download_era5land_heat.py --year {wildcards.year} --month {wildcards.month} --output {output}"

rule wbgt_month:
    input: era5=lambda w: ERA5(w.year, w.month), cells=W / "data/wbgt_cells.parquet"
    output: W / "wbgt/wbgt_{year}-{month}.nc"
    shell: "{PY} scripts/wbgt_month.py --config {CFG} --era5 {input.era5} --cells {input.cells} --out {output}"

rule gonogo:
    input:
        wbgt=[W / f"wbgt/wbgt_{y}-{m}.nc" for y, m in ym_of(GP)],
        wc=W / "data/wbgt_cells.parquet", fac=W / "data/facilities.parquet", fc=W / "data/facility_cells.parquet"
    output: csv=W / "results/gonogo/exceedance.csv", fig=W / "results/gonogo/exceedance_july.png",
            verdict=W / "results/gonogo/verdict.yaml"
    shell: "{PY} scripts/gonogo.py --config {CFG} --wbgt {input.wbgt} --wbgt-cells {input.wc} --facilities {input.fac} "
           "--facility-cells {input.fc} --csv {output.csv} --fig {output.fig} --verdict {output.verdict}"

rule availability_month:
    input:
        wbgt=W / "wbgt/wbgt_{year}-{month}.nc", fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet",
        wc=W / "data/wbgt_cells.parquet", b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet"
    output: W / "avail/avail_{year}-{month}.npz"
    resources: mem_mb=4000
    shell: "{PY} scripts/availability_month.py --config {CFG} --wbgt {input.wbgt} --facilities {input.fac} "
           "--canopy {input.can} --facility-cells {input.fc} --wbgt-cells {input.wc} --catchment {input.b} "
           "--catchment-meta {input.meta} --out {output}"

# ---------------------------------------------- urban heat island (UrbClim, sensitivity)
UH = config["uhi"]
rule uhi_offsets:
    input: fac=W / "data/facilities.parquet", popf=W / "data/population.parquet"
    output: W / "uhi/offsets_{year}.parquet"
    shell: "{PY} scripts/uhi_offsets.py --config {CFG} --urbclim-dir {UH[urbclim_dir]} --facilities {input.fac} "
           "--departements {I[departements_geojson]} --population {input.popf} --year {wildcards.year} --out {output}"

rule availability_uhi:
    input:
        wbgt=W / "wbgt/wbgt_{year}-{month}.nc", fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet",
        wc=W / "data/wbgt_cells.parquet", b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet",
        off=W / "uhi/offsets_{year}.parquet"
    output: W / "avail_uhi/avail_{year}-{month}.npz"
    resources: mem_mb=4000
    shell: "{PY} scripts/availability_month.py --config {CFG} --wbgt {input.wbgt} --facilities {input.fac} "
           "--canopy {input.can} --facility-cells {input.fc} --wbgt-cells {input.wc} --catchment {input.b} "
           "--catchment-meta {input.meta} --offsets {input.off} --out {output}"

rule uhi:
    input:
        popf=W / "data/population.parquet",
        base=W / "access/period_urb/access.parquet", uhi=W / "access/period_urbuhi/access.parquet",
        dbase=W / "clim/deserts_urb", duhi=W / "clim/deserts_urbuhi",
        off=expand(W / "uhi/offsets_{y}.parquet", y=range(UH["years"][0], UH["years"][1] + 1))
    output: W / "results/analysis/uhi_sensitivity.csv"
    shell: "{PY} scripts/uhi_compare.py --config {CFG} --population {input.popf} --access-base {input.base} "
           "--access-uhi {input.uhi} --deserts-base {input.dbase} --deserts-uhi {input.duhi} "
           "--offsets {input.off} --out {output}"

def clim_inputs(w):
    sub = "avail_uhi" if w.period.endswith("uhi") else "avail"
    return [W / f"{sub}/avail_{y}-{m}.npz" for y, m in ym_of(w.period)]

rule climatology:
    input: clim_inputs
    output: npz=W / "clim/abar_{period}.npz", d=directory(W / "clim/deserts_{period}")
    resources: mem_mb=10000
    params: months=" ".join(str(m) for m in config["climate"]["months"])
    shell: "{PY} scripts/climatology.py --inputs {input} --months {params.months} --out-npz {output.npz} --out-dir {output.d}"

# ======================================================================= accessibility / analysis
rule accessibility:
    input:
        od=W / "data/od.parquet", popf=W / "data/population.parquet", fac=W / "data/facilities.parquet",
        clim=lambda w: W / f"clim/abar_{RUNS[w.run]['period']}.npz"
    output: acc=W / "access/{run}/access.parquet", fac=W / "access/{run}/facilities.parquet"
    resources: mem_mb=12000
    params:
        r=lambda w: RUNS[w.run],
        save=lambda w: "--save-L" if w.run == "main" else ""
    shell: "{PY} scripts/accessibility.py --config {CFG} --od {input.od} --population {input.popf} --facilities {input.fac} "
           "--clim {input.clim} --scen {params.r[scen]} --q {params.r[q]} --tau0 {params.r[tau0]} "
           "--access {params.r[access]} --capacity {params.r[cap]} --avail {params.r[avail]} {params.save} "
           "--out {output.acc} --fac-out {output.fac}"

HS = config.get("analysis", {}).get("heat_scenario", "nolight")
rule facility_qpv:
    input: es=I["data_es_zip"], fac=W / "data/facilities.parquet"
    output: W / "data/facility_qpv.parquet"
    shell: "{PY} scripts/facility_qpv.py --data-es {input.es} --facilities {input.fac} --out {output}"

rule analysis:
    input:
        popf=W / "data/population.parquet", cells=W / "data/cells.parquet", fac=W / "data/facilities.parquet",
        qpv=W / "data/facility_qpv.parquet", facheat=W / f"access/scen_{HS}/facilities.parquet",
        main=W / "access/main/access.parquet", heat=W / f"access/scen_{HS}/access.parquet",
        naive=W / "access/naive/access.parquet", recent=W / "access/period_recent/access.parquet",
        dref=W / "clim/deserts_ref", drec=W / "clim/deserts_recent", od=W / "data/od.parquet",
        sens=[W / f"access/{r}/access.parquet" for r in SENS]
    output: W / "results/analysis/h1_decomposition.csv", W / "results/analysis/equity_region.csv",
            W / "results/analysis/equity_density.csv", W / "results/analysis/carreaux_metrics.parquet"
    resources: mem_mb=10000
    params:
        sens=" ".join(f"{r}={W / f'access/{r}/access.parquet'}" for r in SENS),
        out=W / "results/analysis"
    shell: "{PY} scripts/analysis.py --config {CFG} --population {input.popf} --cells {input.cells} "
           "--facilities {input.fac} --fac-qpv {input.qpv} --fac-heat {input.facheat} "
           "--access-main {input.main} --access-heat {input.heat} --access-naive {input.naive} "
           "--access-recent {input.recent} --deserts-ref {input.dref} --deserts-recent {input.drec} --od {input.od} "
           "--sens {params.sens} --outdir {params.out}"

rule sens_3sfca:
    input:
        od=W / "data/od.parquet", popf=W / "data/population.parquet", fac=W / "data/facilities.parquet",
        clim=W / "clim/abar_ref.npz", main=W / "access/main/access.parquet"
    output: W / "results/analysis/sens_3sfca.csv"
    shell: "{PY} scripts/threesfca.py --config {CFG} --od {input.od} --population {input.popf} --facilities {input.fac} "
           "--clim {input.clim} --access-main {input.main} --out {output}"

rule figures:
    input:
        W / "results/analysis/h1_decomposition.csv", W / "results/adaptation/strategies_budgets.csv",
        W / "results/analysis/sens_3sfca.csv", W / "access/naive/access.parquet",
        W / f"access/scen_{HS}/access.parquet", W / "data/facility_canopy.parquet"
    output: W / "results/figures/tables.md"
    params: out=W / "results/figures"
    shell: "{PY} scripts/figures.py --config {CFG} --work {W} --departements {I[departements_geojson]} --out {params.out}"

# ======================================================================= adaptation (shading priorities)
AD = config["adaptation"]
rule adaptation_sample:
    input: [W / f"wbgt/wbgt_{y}-{m}.nc" for y, m in ym_of(AD["period"], AD["months"])]
    output: W / "adaptation/sample.npz"
    shell: "{PY} scripts/adaptation_sample.py --config {CFG} --wbgt {input} --out {output}"

rule adaptation_dep:
    input:
        sample=W / "adaptation/sample.npz", popf=W / "data/population.parquet", fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet", wc=W / "data/wbgt_cells.parquet",
        b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet",
        facacc=W / "access/main/facilities.parquet"
    output: res=W / "adaptation/dep/greedy_{dep}.parquet", tot=W / "adaptation/dep/desert_total_{dep}.parquet"
    shell: "{PY} scripts/adaptation_dep.py --config {CFG} --dep {wildcards.dep} --sample {input.sample} "
           "--population {input.popf} --facilities {input.fac} --canopy {input.can} --facility-cells {input.fc} "
           "--wbgt-cells {input.wc} --catchment {input.b} --catchment-meta {input.meta} --fac-access {input.facacc} "
           "--out {output.res}"

rule adaptation_merge:
    input:
        res=expand(W / "adaptation/dep/greedy_{dep}.parquet", dep=DEPS),
        tot=expand(W / "adaptation/dep/desert_total_{dep}.parquet", dep=DEPS)
    output: curve=W / "results/adaptation/curves.parquet", table=W / "results/adaptation/strategies_budgets.csv",
            fig=W / "results/adaptation/curves.png"
    shell: "{PY} scripts/adaptation_merge.py --config {CFG} --inputs {input.res} --totals {input.tot} "
           "--curve {output.curve} --table {output.table} --fig {output.fig}"

# ------------------------------------------------ review: robustness (out-of-sample shading, desert subsets, diagnostics)
EV = AD.get("eval_samples", {})
REF_YEARS = range(P["ref"][0], P["ref"][1] + 1)
DS = config["analysis"].get("desert_subsets", {})
SHADES = [f"f{int(round(x * 100))}" for x in AD.get("eval_partial_shade", [])]
EVAL_TAGS = list(EV) + [f"seed2_{sh}" for sh in SHADES]
EVAL_FILES = [W / f"adaptation/eval_{e}/eval_{d}.parquet" for e in EVAL_TAGS for d in DEPS]
EVAL_SPECS = " ".join(f"{e}=" + ",".join(str(W / f"adaptation/eval_{e}/eval_{d}.parquet") for d in DEPS) for e in EVAL_TAGS)
SUB_FILES = [W / f"clim/deserts_sub_{s}_ref.parquet" for s in DS]
SUB_SPECS = " ".join(f"{s}=" + ",".join(str(W / f"clim/deserts_sub_{s}_ref{x}.parquet") for x in ["", ".ndays", ".served"])
                     for s in DS)

def subset_wbgt(w):
    return [W / f"wbgt/wbgt_{w.year}-{m}.nc" for m in MONTHS]

def subset_years(w):
    return [W / f"clim/deserts_sub_{w.sub}/deserts_{y}.parquet" for y in REF_YEARS]

rule adaptation_select:
    input: expand(W / "adaptation/dep/greedy_{dep}.parquet", dep=DEPS)
    output: W / "adaptation/selected.parquet"
    shell: "{PY} scripts/adaptation_select.py --config {CFG} --inputs {input} --out {output}"

def eval_sample_inputs(w):
    y0, y1 = EV[w.ev]["years"]
    return [W / f"wbgt/wbgt_{y}-{m:02d}.nc" for y in range(y0, y1 + 1) for m in AD["months"]]

rule adaptation_sample_eval:
    input: eval_sample_inputs
    output: W / "adaptation/sample_{ev}.npz"
    wildcard_constraints: ev="[a-z0-9]+"
    params: seed=lambda w: EV[w.ev]["seed"]
    shell: "{PY} scripts/adaptation_sample.py --config {CFG} --seed {params.seed} --wbgt {input} --out {output}"

rule adaptation_eval_dep:
    input:
        sample=W / "adaptation/sample_{ev}.npz", sel=W / "adaptation/selected.parquet",
        popf=W / "data/population.parquet", fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet", wc=W / "data/wbgt_cells.parquet",
        b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet"
    output: W / "adaptation/eval_{ev}/eval_{dep}.parquet"
    wildcard_constraints: ev="[a-z0-9]+"
    shell: "{PY} scripts/adaptation_eval.py --config {CFG} --dep {wildcards.dep} --sample {input.sample} "
           "--selected {input.sel} --population {input.popf} --facilities {input.fac} --canopy {input.can} "
           "--facility-cells {input.fc} --wbgt-cells {input.wc} --catchment {input.b} --catchment-meta {input.meta} "
           "--out {output}"

rule adaptation_eval_dep_shade:
    input:
        sample=W / "adaptation/sample_{ev}.npz", sel=W / "adaptation/selected.parquet",
        popf=W / "data/population.parquet", fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet", wc=W / "data/wbgt_cells.parquet",
        b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet"
    output: W / "adaptation/eval_{ev}_f{sh}/eval_{dep}.parquet"
    wildcard_constraints: ev="[a-z0-9]+", sh="[0-9]+"
    params: shade=lambda w: int(w.sh) / 100
    shell: "{PY} scripts/adaptation_eval.py --config {CFG} --dep {wildcards.dep} --sample {input.sample} "
           "--selected {input.sel} --population {input.popf} --facilities {input.fac} --canopy {input.can} "
           "--facility-cells {input.fc} --wbgt-cells {input.wc} --catchment {input.b} --catchment-meta {input.meta} "
           "--shade {params.shade} --out {output}"

rule review_diag:
    input:
        popf=W / "data/population.parquet", met=W / "results/analysis/carreaux_metrics.parquet",
        acc=W / "access/main/access.parquet", od=W / "data/od.parquet", fac=W / "data/facilities.parquet",
        facheat=W / f"access/scen_{HS}/facilities.parquet", b=W / "data/catchment_main.npz",
        meta=W / "data/catchment_meta.parquet"
    output: W / "results/analysis/diag_shapley.csv", W / "results/analysis/diag_conservation.csv",
            W / "results/analysis/diag_crossdep.csv"
    resources: mem_mb=16000
    params: out=W / "results/analysis"
    shell: "{PY} scripts/review_diag.py --config {CFG} --population {input.popf} --metrics {input.met} "
           "--access {input.acc} --od {input.od} --facilities {input.fac} --fac-heat {input.facheat} "
           "--catchment {input.b} --catchment-meta {input.meta} --outdir {params.out}"

rule adaptation_eval:
    input:
        ins=W / "results/adaptation/strategies_budgets.csv",
        ev=EVAL_FILES
    output: W / "results/adaptation/out_of_sample.csv"
    params: specs=EVAL_SPECS
    shell: "{PY} scripts/adaptation_eval_merge.py --in-sample {input.ins} --eval {params.specs} --out {output}"

rule deserts_subset_year:
    input:
        wbgt=subset_wbgt, fac=W / "data/facilities.parquet",
        can=W / "data/facility_canopy.parquet", fc=W / "data/facility_cells.parquet",
        wc=W / "data/wbgt_cells.parquet", b=W / "data/catchment_main.npz", meta=W / "data/catchment_meta.parquet"
    output: d=W / "clim/deserts_sub_{sub}/deserts_{year}.parquet",
            n=W / "clim/deserts_sub_{sub}/deserts_{year}.ndays.parquet",
            s=W / "clim/deserts_sub_{sub}/deserts_{year}.served.parquet"
    wildcard_constraints: sub="[a-z0-9]+", year="[0-9]{4}"
    resources: mem_mb=4000
    shell: "{PY} scripts/deserts_subset.py --config {CFG} --subset {wildcards.sub} --wbgt {input.wbgt} "
           "--facilities {input.fac} --canopy {input.can} --facility-cells {input.fc} --wbgt-cells {input.wc} "
           "--catchment {input.b} --catchment-meta {input.meta} --out {output.d}"

rule deserts_subset:
    input: subset_years
    output: d=W / "clim/deserts_sub_{sub}_ref.parquet", n=W / "clim/deserts_sub_{sub}_ref.ndays.parquet",
            s=W / "clim/deserts_sub_{sub}_ref.served.parquet"
    wildcard_constraints: sub="[a-z0-9]+"
    resources: mem_mb=6000
    shell: "{PY} scripts/deserts_subset_merge.py --inputs {input} --out {output.d} --ndays {output.n} --served {output.s}"

rule robustness:
    input:
        met=W / "results/analysis/carreaux_metrics.parquet", popf=W / "data/population.parquet",
        acc=W / "access/main/access.parquet", dref=W / "clim/deserts_ref",
        sub=SUB_FILES
    output: W / "results/analysis/robustness_deserts.csv"
    params: subs=SUB_SPECS
    resources: mem_mb=8000
    shell: "{PY} scripts/robustness.py --config {CFG} --metrics {input.met} --population {input.popf} "
           "--access {input.acc} --deserts-dir {input.dref} --subsets {params.subs} --out {output}"

rule review:
    input: W / "results/adaptation/out_of_sample.csv", W / "results/analysis/robustness_deserts.csv",
           W / "results/figures/tables.md", W / "results/analysis/diag_shapley.csv"

# ======================================================================= tests
rule test_liljegren:
    output: touch(W / "logs/test_liljegren.ok")
    shell: "{PY} -m pytest -q tests/"
