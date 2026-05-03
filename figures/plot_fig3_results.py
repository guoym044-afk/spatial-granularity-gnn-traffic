"""Generate Figure 3: 4-panel results overview for SIGSPATIAL paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

# ============================================================
# (a) Log-log coarsening curves: MAE vs N
# ============================================================
ax = axes[0, 0]

# METR-LA data (Table 9: 10-seed means)
N_metr = np.array([5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 207])
mae_dcrnn_metr = np.array([2.03, 2.26, 2.46, 2.70, 2.88, 3.06, 3.30, 3.46, 3.58, 3.71, 3.82])
mae_lstgcnn_metr = np.array([3.00, 3.32, 3.48, 3.85, 4.02, 4.23, 4.52, 4.70, 4.85, 5.00, 5.11])
mae_gwn_metr = np.array([1.98, 2.22, 2.35, 2.63, 2.78, 2.96, 3.19, 3.35, 3.46, 3.58, 3.67])

# PeMS-Bay data (Table 10)
N_pems = np.array([5, 10, 33, 65, 98, 326])
mae_dcrnn_pems = np.array([0.93, 1.03, 1.12, 1.19, 1.23, 1.27])
mae_stgcn_pems = np.array([0.91, 1.01, 1.10, 1.16, 1.19, 1.23])

ax.loglog(N_metr, mae_dcrnn_metr, 'o-', color='#2171b5', ms=3, lw=1.2, label='METR DCRNN')
ax.loglog(N_metr, mae_gwn_metr, 's-', color='#6baed6', ms=3, lw=1.2, label='METR GWN')
ax.loglog(N_metr, mae_lstgcnn_metr, '^-', color='#c6dbef', ms=3, lw=1.2, label='METR LST')
ax.loglog(N_pems, mae_dcrnn_pems, 'o--', color='#e6550d', ms=3, lw=1.2, label='PeMS DCRNN')
ax.loglog(N_pems, mae_stgcn_pems, 's--', color='#fd8d3c', ms=3, lw=1.2, label='PeMS STGCN')

# Power-law fit lines
N_fit = np.logspace(np.log10(5), np.log10(210), 50)
a_dcrnn, b_dcrnn = 1.57, 0.173
ax.loglog(N_fit, a_dcrnn * N_fit**b_dcrnn, ':', color='#2171b5', lw=0.8, alpha=0.6)

ax.set_xlabel('Node count $N$')
ax.set_ylabel('MAE')
ax.set_title('(a) Coarsening curves (log-log)')
ax.legend(loc='lower right', framealpha=0.8)
ax.grid(True, alpha=0.2, which='both')
ax.set_xlim(4, 350)

# ============================================================
# (b) Smoothing control: MAE vs sigma
# ============================================================
ax = axes[0, 1]

sigma = np.array([0, 1, 2, 3])

# METR-LA smoothing (Table 3)
metr_dcrnn_smooth = np.array([3.68, 1.49, 0.58, 0.39])
metr_gwn_smooth = np.array([3.61, 2.22, 1.35, 1.01])
metr_lst_smooth = np.array([5.08, 3.59, 2.97, 2.89])
metr_pers_smooth = np.array([3.38, 1.98, 1.37, 1.12])

# PeMS-Bay smoothing (Table 4)
pems_dcrnn_smooth = np.array([1.256, 0.892, 0.800, 0.794])
pems_stgcn_smooth = np.array([1.206, 0.844, 0.740, 0.741])
pems_pers_smooth = np.array([1.716, 1.208, 1.006, 0.961])

ax.plot(sigma, metr_dcrnn_smooth, 'o-', color='#2171b5', ms=4, lw=1.2, label='METR DCRNN')
ax.plot(sigma, metr_gwn_smooth, 's-', color='#6baed6', ms=4, lw=1.2, label='METR GWN')
ax.plot(sigma, metr_lst_smooth, '^-', color='#c6dbef', ms=4, lw=1.2, label='METR LST')
ax.plot(sigma, metr_pers_smooth, 'x:', color='#2171b5', ms=4, lw=0.8, label='METR Pers.')
ax.plot(sigma, pems_dcrnn_smooth, 'o--', color='#e6550d', ms=4, lw=1.2, label='PeMS DCRNN')
ax.plot(sigma, pems_stgcn_smooth, 's--', color='#fd8d3c', ms=4, lw=1.2, label='PeMS STGCN')
ax.plot(sigma, pems_pers_smooth, 'x:', color='#e6550d', ms=4, lw=0.8, label='PeMS Pers.')

ax.set_xlabel('Smoothing $\\sigma$')
ax.set_ylabel('MAE')
ax.set_title('(b) Smoothing control ($N{=}N_{\\max}$)')
ax.legend(loc='upper right', framealpha=0.8, ncol=2)
ax.set_xticks([0, 1, 2, 3])
ax.grid(True, alpha=0.2)

# ============================================================
# (c) Coarsened vs sensor-subset (METR-LA)
# ============================================================
ax = axes[1, 0]

N_sub = np.array([5, 20, 50, 100, 150, 207])

# DCRNN (Table 6)
dcrnn_coarsened = np.array([2.05, 2.68, 3.14, 3.43, 3.67, 3.82])
dcrnn_subset = np.array([3.91, 3.78, 3.77, 3.66, 3.75, 3.72])
dcrnn_subset_std = np.array([0.36, 0.26, 0.09, 0.13, 0.05, 0.09])

# GWN (Table 6)
gwn_coarsened = np.array([2.05, 2.67, 3.13, 3.31, 3.54, 3.67])
gwn_subset = np.array([3.87, 3.68, 3.65, 3.57, 3.58, 3.57])
gwn_subset_std = np.array([0.34, 0.22, 0.07, 0.12, 0.11, 0.05])

ax.plot(N_sub, dcrnn_coarsened, 'o-', color='#2171b5', ms=4, lw=1.2, label='DCRNN coarsened')
ax.fill_between(N_sub, dcrnn_subset - dcrnn_subset_std, dcrnn_subset + dcrnn_subset_std,
                alpha=0.15, color='#e6550d')
ax.plot(N_sub, dcrnn_subset, 'o--', color='#e6550d', ms=4, lw=1.2, label='DCRNN subset')

ax.plot(N_sub, gwn_coarsened, 's-', color='#6baed6', ms=4, lw=1.2, label='GWN coarsened')
ax.fill_between(N_sub, gwn_subset - gwn_subset_std, gwn_subset + gwn_subset_std,
                alpha=0.15, color='#fd8d3c')
ax.plot(N_sub, gwn_subset, 's--', color='#fd8d3c', ms=4, lw=1.2, label='GWN subset')

ax.set_xlabel('Node count $N$')
ax.set_ylabel('MAE')
ax.set_title('(c) Coarsened vs. subset (METR-LA)')
ax.legend(loc='lower right', framealpha=0.8)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 220)
ax.set_ylim(1.5, 4.5)

# Annotate b values
ax.annotate('$b{=}0.17$', xy=(80, 3.35), fontsize=6, color='#2171b5')
ax.annotate('near-zero slope', xy=(80, 3.85), fontsize=5.5, color='#e6550d')

# ============================================================
# (d) Feature-vs-graph ablation (METR-LA)
# ============================================================
ax = axes[1, 1]

# Table 7 data at N=100
conds = ['Subset', 'Coarsened', 'Feat-only', 'Graph-only']
dcrnn_100 = [3.66, 3.49, 5.48, 7.80]
dcrnn_100_std = [0.13, 0.07, 0.14, 0.02]
gwn_100 = [3.57, 3.33, 4.85, 7.79]
gwn_100_std = [0.12, 0.03, 0.05, 0.08]

x = np.arange(len(conds))
width = 0.32

bars1 = ax.bar(x - width/2, dcrnn_100, width, yerr=dcrnn_100_std, capsize=2,
               color='#2171b5', label='DCRNN', edgecolor='white', lw=0.5, error_kw={'lw': 0.6})
bars2 = ax.bar(x + width/2, gwn_100, width, yerr=gwn_100_std, capsize=2,
               color='#e6550d', label='GWN', edgecolor='white', lw=0.5, error_kw={'lw': 0.6})

# Persistence baseline
ax.axhline(y=3.51, color='#636363', ls=':', lw=0.8, label='Persistence')

ax.set_xlabel('Condition')
ax.set_ylabel('MAE')
ax.set_title('(d) Feature-vs-graph ablation ($N{=}100$)')
ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=6)
ax.legend(loc='upper left', framealpha=0.8)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 9)

plt.tight_layout(pad=0.8)
plt.savefig('fig3_results_overview.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
print('Saved fig3_results_overview.pdf')
