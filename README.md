# climae_heat_access

Snakemake workflow for the paper **"Heat-constrained potential accessibility to outdoor sport facilities in France: A dynamic supply framework for spatial planning"** (manuscript submitted to *Applied Geography*).

The workflow scales the supply of each public outdoor sport facility in metropolitan France by its hourly thermal availability and runs a dynamic enhanced two-step floating catchment area model (E2SFCA) on 200-m population cells:

```
hourly WBGT (Liljegren, ERA5-Land)
  → facility availability (cover, tree canopy, surface type, lighting)
  → dynamic E2SFCA (15-min walk, OSM network, Filosofi 2021 population)
  → peak and seasonal losses, temporary accessibility deserts
  → equity analysis (age, income, region)
  → shading priorities as a maximum-coverage problem
```

## Installation

Tested on Linux (WSL2, Ubuntu) with conda/mamba.

```bash
conda env create -f envs/environment.yml
conda activate climae-heat-access
python -m pytest -q tests/        # the vectorised WBGT port reproduces the scalar reference
```

ERA5-Land downloads (optional, see below) also need a Copernicus CDS account (`~/.cdsapirc`) and [CDO](https://code.mpimet.mpg.de/projects/cdo).

## Input data

Set the paths in `config/config.yaml` (all paths there are placeholders).

| Key | Data | Source | Notes |
|---|---|---|---|
| `data_es_zip` | Data ES, national census of sport facilities (full export) | Ministère des Sports, <https://equipements.sports.gouv.fr> | Licence Ouverte / Etalab |
| `departements_geojson` | Départements of metropolitan France | e.g. IGN Admin Express or <https://github.com/gregoiredavid/france-geojson> | property `code` |
| `filosofi_gpkg` | Gridded population and income, Filosofi 2021, "niveau naturel" | INSEE, <https://www.insee.fr> | |
| `osm_pbf` | OpenStreetMap extract of France | Geofabrik, <https://download.geofabrik.de/europe/france.html> | the paper used the 12 Oct 2026 extract; © OpenStreetMap contributors, ODbL |
| `era5land_dir` | ERA5-Land hourly, May–September 1991–2025 | Copernicus CDS | if the files are absent, rule `era5_download` fetches a light profile (t2m, d2m, sp, u10, v10, ssrd); limit parallel requests with `--resources cds=4` |
| `canopy_raster` | High Resolution Layer Tree Cover Density 2024, 10 m | Copernicus Land Monitoring Service via CDSE | `scripts/tcd_download.py` downloads the tiles and builds the VRT (needs `CDSE_USER` and `CDSE_PASSWORD` environment variables); an empty value means zero canopy |
| `uhi.urbclim_dir` | UrbClim 100-m urban climate simulations (optional, Table S7) | VITO, <https://provide.marvin.vito.be> | `scripts/download_urbclim.py`, then `scripts/dem_urbclim.py` |

Intermediate and final outputs go to `work_root`, outside the repository. The full replay writes monthly WBGT and availability files for 1991–2025 and needs substantial disk space.

## Running

```bash
snakemake -n                                     # dry run
snakemake -j 8 --resources cds=4 gonogo          # climate screen (Table S3)
snakemake -j 4 --config deps='["34","30"]'       # test on two départements
snakemake -j 8 --resources mem_mb=48000          # full replay
```

The urban heat island sensitivity needs UrbClim data and is run separately:

```bash
snakemake -j 4 <work_root>/results/analysis/uhi_sensitivity.csv
```

A plumbing test without OSM or ERA5-Land data is available: `python tests/make_smoke.py --help`.

Snakemake targets are absolute paths under `work_root`. Use `--rerun-triggers mtime` when re-running after code changes you do not want to propagate.

## Outputs and the paper

All outputs are written to `<work_root>/results/`.

| Paper item | File |
|---|---|
| Fig. 2 supply composition | `figures/fig2_supply_composition.png` |
| Fig. 3 hourly availability | `figures/fig3_hourly_availability.png` |
| Fig. 4 naive vs modelled losses (H1) | `figures/fig4_h1_naive_vs_modelled.png` |
| Fig. 5 peak loss map | `figures/fig5_map_Lpeak.png` |
| Fig. 6 deserts | `figures/fig6_deserts.png` |
| Fig. 7 compound typology | `figures/fig7_bivariate.png` |
| Fig. 8 shading curves | `adaptation/curves.png` |
| Main-text tables and most supplementary tables | `figures/tables.md`, `analysis/*.csv` |
| Variance decomposition (H1) | `analysis/h1_decomposition.csv` |
| Equity by region and density | `analysis/equity_region.csv`, `analysis/equity_density.csv` |
| Three-step FCA comparison | `analysis/sens_3sfca.csv` |
| Shading strategies and budgets | `adaptation/strategies_budgets.csv` |
| Table S2 robustness of the income gradient | `analysis/robustness_deserts.csv` |
| Table S3 climate screen | `gonogo/exceedance.csv`, `gonogo/verdict.yaml` |
| Table S5 out-of-sample and partial shading | `adaptation/out_of_sample.csv` |
| Table S7 urban heat island | `analysis/uhi_sensitivity.csv` |
| Text S1 diagnostics (conservation sets, cross-département catchments, Shapley decomposition) | `analysis/diag_conservation.csv`, `analysis/diag_crossdep.csv`, `analysis/diag_shapley.csv` |

## Method choices encoded in the workflow

- **WBGT.** Vectorised NumPy port of a scalar Liljegren implementation (`scripts/liljegren_np.py`): Erbs diffuse split, solar geometry at mid-hour, 10-m wind reduced to 2 m with a logarithmic profile, separate energy balances for the natural wet bulb and the globe. The scalar reference is frozen in `vendor/climae_school/` and `tests/test_liljegren_np.py` compares both. Deviations from the reference: NaN with a flag instead of an exception, clearness index clipped at 1 with a flag.
- **Facility microclimate.** WBGT at a facility = f · WBGT(shade) + (1 − f) · WBGT(sun, ground albedo α_s, ground temperature T_s = T_a + k · GHI). f = 1 for covered facilities, otherwise the tree-canopy fraction within 20 m. Surface classes and their (α_s, k) are in `config/config.yaml` with their sources. The `reference` class reproduces the unmodified Liljegren setup.
- **Availability.** A facility is available in an hour when its WBGT is below the threshold (26 °C main; 28 and 30 °C sensitivity) and, for unlit facilities, the sun is above the horizon.
- **Radiation.** ERA5-Land ssrd is de-accumulated (reset at 01 UTC); the interval (t − 1 h, t] is labelled by its local start hour.
- **Population.** Cells larger than 200 m are disaggregated to 200-m sub-cells in proportion to OSM residential building footprint. A sensitivity run restricts the analysis to native 200-m cells.
- **E2SFCA.** Same weights in both steps, truncated Gaussian with w(τ0) = 0.01. The conservation identity Σ P·A_heat = Σ C·ā is checked at each run, so equity is read on distributions and deserts, not on group means.
- **Deserts.** Computed on actual day × hour values, not on the climatology.
- **Shading priorities.** Heat-attributable desert person-hours avoided, solved by greedy maximum coverage per département and merged nationally by marginal gain. Events are sampled (1,400 hours stratified by local hour, June–August 2006–2025) and the selected sets are re-evaluated on independent samples, including a partial-shading variant.

## Repository layout

```
Snakefile             workflow
config/               parameters, WBGT policy and sensor settings
envs/environment.yml  conda environment
scripts/              one script per rule
tests/                WBGT port test and synthetic smoke test
vendor/climae_school/ frozen scalar WBGT reference and third-party licence notices
```

## Licence

Code: MIT licence (see `LICENSE`). The Liljegren WBGT code in `vendor/climae_school/` keeps its original notices (UChicago Argonne LLC; Max Lieblich, MIT), see `vendor/climae_school/THIRD_PARTY_NOTICES.txt`. Input data keep their own licences (see the table above); derived databases built from OpenStreetMap are subject to the ODbL.

## Citation

If you use this workflow, please cite the paper (reference to be completed on publication) and this repository (see `CITATION.cff`).
