# When Does Spatial Granularity Matter? A Smoothing-Aware Diagnostic for GNN Traffic Forecasting [Experiment]

**SIGSPATIAL 2026 — Research Track (Experiment Paper)**

Yiming Guo, Sijin Liu

## Overview

This repository accompanies our SIGSPATIAL 2026 experiment paper on spatial granularity effects in GNN-based traffic prediction. We propose a **smoothing-aware diagnostic framework** that separates genuine spatial modeling gains from aggregation-induced temporal smoothing artifacts.

**Core finding:** Prediction error follows a power law (MAE &prop; N^b), but same-N smoothing and sensor-subset controls reveal this scaling is substantially driven by temporal smoothing, not spatial information loss. On METR-LA (lag-1 ACF = 0.939), no GNN beats persistence; on PeMS-Bay, GNNs achieve positive skill (+0.26).

## Repository Structure

```
paper_sigspatial.tex         # LaTeX source (ACM sigconf, 8 pages + references)
paper_sigspatial.pdf         # Compiled PDF
references.bib               # BibTeX bibliography
acmart.cls                   # ACM article class (v1.93)
ACM-Reference-Format.bst     # ACM bibliography style
acmart-1.93/                 # Full acmart template directory
figures/
  fig_sensor_map.pdf         # Sensor location maps (METR-LA + PeMS-Bay)
  fig3_results_overview.pdf  # 4-panel results overview (coarsening, smoothing, subset, ablation)
  plot_sensor_map.py         # Script to regenerate sensor map
  plot_fig3_results.py       # Script to regenerate results overview
  metr_la_locations.csv      # METR-LA sensor coordinates
```

## How to Compile

Requires a LaTeX installation with the ACM acmart class (included).

```bash
pdflatex paper_sigspatial
bibtex paper_sigspatial
pdflatex paper_sigspatial
pdflatex paper_sigspatial
```

## Datasets

| Dataset | Sensors | Timesteps | Interval | Source |
|---------|---------|-----------|----------|--------|
| METR-LA | 207 | 34,272 | 5 min | LA highway network |
| PeMS-Bay | 326 | 51,840 | 5 min | SF Bay Area |

Data not included in repo due to size. Download from:
- METR-LA: https://github.com/liyaguang/DCRNN
- PeMS-Bay: https://pems.dot.ca.gov/

## Diagnostic Framework

The paper proposes five diagnostics for evaluating spatial granularity effects:

1. **Persistence Skill Test** — Check if any model beats last-value prediction
2. **Coarsening Curve** — Fit MAE(N) = a &middot; N^b across granularity levels
3. **Sensor-Subset Control** — Vary N without averaging (tests if b &asymp; 0)
4. **Same-N Smoothing Control** — Apply explicit Gaussian smoothing at full resolution
5. **Feature-vs-Graph Ablation** — Decompose coarsening into feature averaging vs graph compression

## Experiment Scale

| Experiment | Runs | Models | Variables |
|-----------|------|--------|-----------|
| METR-LA granularity | 330 | DCRNN, GraphWaveNet, LSTGCNN | 11 levels x 10 seeds |
| PeMS-Bay granularity | 120+ | DCRNN, STGCN, GraphWaveNet, LSTGCNN | 11 levels x 10 seeds |
| Smoothing control | 60 | DCRNN, STGCN, GraphWaveNet | 4 &sigma; x 3 seeds x 2 datasets |
| Sensor subset | 72 | DCRNN, GraphWaveNet | 6 N-levels x 3 seeds x 2 datasets |
| Feature-vs-graph ablation | 72 | DCRNN, GraphWaveNet | 4 conditions x 3 seeds x 2 datasets |
| Distance-aware coarsening | 24 | DCRNN | 4 N-levels x 3 seeds x 2 datasets |
| GAT diagnosis | 10+ | GAT, RandomGAT | 3-10 seeds |
| Baselines | 4 | Persistence, HA | both datasets |
| **Total** | **651+** | **5 architectures** | |

## Citation

```bibtex
@inproceedings{guo2026granularity,
  title={When Does Spatial Granularity Matter? A Smoothing-Aware Diagnostic for GNN Traffic Forecasting},
  author={Guo, Yiming and Liu, Sijin},
  booktitle={ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems},
  year={2026}
}
```

## License

This repository contains the research paper and associated materials for academic use.
