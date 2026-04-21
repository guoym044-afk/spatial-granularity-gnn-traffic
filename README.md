# When Does Spatial Granularity Matter?

**A Diagnostic Study of Spatial Granularity Effects in GNN-Based Traffic Prediction**

## Key Findings

- On METR-LA, **all GNNs fail to beat a trivial persistence baseline** (MAE=3.51). Best GNN: GraphWaveNet 3.67 (skill=-0.047)
- Prediction error follows an empirical power law: MAE &prop; N^b, b&asymp;0.16 (METR-LA), b&asymp;0.28 (PeMS-Bay)
- A **same-N smoothing control** reveals this scaling is primarily driven by temporal signal smoothing, not spatial information loss
- GAT's learned attention degenerates to near-uniform (entropy=0.993), matching a fixed-uniform RandomGAT baseline
- Lag-1 autocorrelation explains b divergence: METR-LA ACF=0.939, PeMS-Bay ACF=0.365

## Repository Structure

```
paper.tex                        # Paper source (LaTeX, 8 pages + references)
references.bib                   # BibTeX bibliography (22 entries)
figures/                         # Publication figures (PDF)
  fig1_power_law.pdf             #   Log-log power-law scaling
  fig2_gat_entropy.pdf           #   GAT attention entropy distribution
  fig3_marginal_returns.pdf      #   Marginal returns curve
plot_figures.py                  # Script to regenerate all figures

code/                            # Experiment code (GPU required)
  config.py                      #   Paths, scenarios, hyperparameters
  data_utils.py                  #   Data loading, node merging (Algorithm 1)
  model.py                       #   All 5 GNN architectures + baselines
  train.py                       #   Training loop with early stopping
  run_v5_experiments.py          #   METR-LA: 11 granularities x 10 seeds
  run_metr_multigran_v14.py      #   PeMS-Bay: multi-granularity experiments
  run_smoothing_control.py       #   Same-N smoothing control (sigma in {0,1,2,3})
  run_autocorr.py                #   Lag-1 autocorrelation computation
  run_arima_baseline.py          #   Historical Average baseline

results/                         # Sample result JSONs (full set: 505 files on server)
```

## How to Compile Paper

Requires a LaTeX installation with `aaai2026.sty` (included in the AAAI 2027 template).

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

## Datasets

| Dataset | Sensors | Timesteps | Interval | Source |
|---------|---------|-----------|----------|--------|
| METR-LA | 207 | 34,272 | 5 min | LA highway network |
| PeMS-Bay | 326 | 51,840 | 5 min | SF Bay Area |

Data not included in repo due to size. Download from:
- METR-LA: https://github.com/liyaguang/DCRNN
- PeMS-Bay: https://pems.dot.ca.gov/

## Reproducing Experiments

Requires: Python 3.8+, PyTorch 1.12+, PyG (PyTorch Geometric), GPU.

```bash
# 1. Set up data paths in code/config.py

# 2. Run METR-LA multi-granularity (3 models x 11 granularities x 10 seeds = 330 runs)
python code/run_v5_experiments.py --model dcrnn --device cuda:0

# 3. Run smoothing control (3 models x 4 sigma x 3 seeds = 36 runs)
python code/run_smoothing_control.py --model dcrnn --sigma 2 --seed 42 --device cuda:0

# 4. Compute lag-1 autocorrelation (CPU, ~30 seconds)
python code/run_autocorr.py

# 5. Run HA baseline (CPU, ~1 minute)
python code/run_arima_baseline.py

# 6. Generate figures
python plot_figures.py
```

## Experiment Scale

| Experiment | Runs | Models | Variables |
|-----------|------|--------|-----------|
| METR-LA granularity | 330 | 3 (DCRNN, GraphWaveNet, LSTGCNN) | 11 granularities x 10 seeds |
| PeMS-Bay granularity | 120+ | 4 (DCRNN, STGCN, GraphWaveNet, LSTGCNN) | 11 granularities x 10 seeds |
| Smoothing control | 36 | 3 | 4 sigma levels x 3 seeds |
| GAT diagnosis | 10+ | GAT, RandomGAT | 3-10 seeds |
| Baselines | 4 | Persistence, HA | both datasets |
| **Total** | **500+** | **5 architectures** | |
