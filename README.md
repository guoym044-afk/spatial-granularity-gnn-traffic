# Spatial Granularity in GNN Traffic Forecasting

Code and materials for studying spatial granularity effects in GNN-based traffic prediction.

## Overview

This project proposes a **smoothing-aware diagnostic framework** for evaluating spatial granularity effects in traffic sensor network forecasting. The framework separates genuine spatial modeling gains from aggregation-induced temporal smoothing artifacts through five diagnostics:

1. **Persistence Skill Test** — Check if any model beats last-value prediction
2. **Coarsening Curve** — Fit MAE(N) = a &middot; N^b across granularity levels
3. **Sensor-Subset Control** — Vary N without averaging (tests if b &asymp; 0)
4. **Same-N Smoothing Control** — Apply explicit Gaussian smoothing at full resolution
5. **Feature-vs-Graph Ablation** — Decompose coarsening into feature averaging vs graph compression

## Repository Structure

```
figures/
  fig_sensor_map.pdf         # Sensor location maps (METR-LA + PeMS-Bay)
  fig3_results_overview.pdf  # 4-panel results overview
  plot_sensor_map.py         # Script to regenerate sensor map
  plot_fig3_results.py       # Script to regenerate results overview
  metr_la_locations.csv      # METR-LA sensor coordinates
```

## Datasets

| Dataset | Sensors | Timesteps | Interval | Source |
|---------|---------|-----------|----------|--------|
| METR-LA | 207 | 34,272 | 5 min | LA highway network |
| PeMS-Bay | 326 | 51,840 | 5 min | SF Bay Area |

Data not included in repo due to size. Download from:
- METR-LA: https://github.com/liyaguang/DCRNN
- PeMS-Bay: https://pems.dot.ca.gov/

## Experiment Scale

| Experiment | Runs | Models | Variables |
|-----------|------|--------|-----------|
| METR-LA granularity | 330 | DCRNN, GraphWaveNet, LSTGCNN | 11 levels x 10 seeds |
| PeMS-Bay granularity | 120+ | DCRNN, STGCN, GraphWaveNet, LSTGCNN | 11 levels x 10 seeds |
| Smoothing control | 60 | DCRNN, STGCN, GraphWaveNet | 4 sigma x 3 seeds x 2 datasets |
| Sensor subset | 72 | DCRNN, GraphWaveNet | 6 N-levels x 3 seeds x 2 datasets |
| Feature-vs-graph ablation | 72 | DCRNN, GraphWaveNet | 4 conditions x 3 seeds x 2 datasets |
| Distance-aware coarsening | 24 | DCRNN | 4 N-levels x 3 seeds x 2 datasets |
| GAT diagnosis | 10+ | GAT, RandomGAT | 3-10 seeds |
| Baselines | 4 | Persistence, HA | both datasets |
| **Total** | **651+** | **5 architectures** | |
