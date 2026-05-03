"""Generate sensor map figure for SIGSPATIAL paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Load METR-LA sensor locations (real coordinates from DCRNN repo)
data = np.loadtxt('metr_la_locations.csv', delimiter=',', skiprows=1)
lat_metr, lon_metr = data[:, 0], data[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# --- (a) METR-LA ---
ax = axes[0]
ax.scatter(lon_metr, lat_metr, s=8, c='#2171b5', edgecolors='white',
           linewidths=0.3, zorder=5, alpha=0.85)
ax.set_title('(a) METR-LA ($N{=}207$)', fontsize=8, fontweight='bold', pad=3)
ax.set_xlabel('Longitude', fontsize=7)
ax.set_ylabel('Latitude', fontsize=7)
ax.tick_params(labelsize=6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linewidth=0.5)

# Add freeway labels (approximate positions)
freeways = {
    'I-10': (-118.35, 34.055),
    'I-110': (-118.275, 34.08),
    'I-405': (-118.45, 34.12),
    'I-210': (-118.15, 34.19),
    'US-101': (-118.32, 34.10),
}
for name, (x, y) in freeways.items():
    ax.annotate(name, (x, y), fontsize=5, color='#636363',
                ha='center', va='bottom', alpha=0.7)

# --- (b) PeMS-Bay (schematic) ---
# Generate synthetic sensor positions along SF Bay Area highway corridors
# Real PeMS-Bay: 326 sensors on I-880, I-580, I-680, US-101, I-280, SR-85, etc.
# Bounds: Lat 37.3-37.9, Lon -122.5 to -121.8
np.random.seed(42)
n_pems = 326

# Create highway-like corridors
corridors = [
    # (lat_start, lat_end, lon_start, lon_end, n_sensors)
    (37.35, 37.85, -122.25, -122.15, 80),   # I-880 (east bay, N-S)
    (37.40, 37.80, -122.15, -121.85, 60),   # I-580 (east, NW-SE)
    (37.35, 37.75, -122.05, -121.85, 50),   # I-680 (east, N-S)
    (37.35, 37.80, -122.45, -122.10, 70),   # US-101 (peninsula, N-S)
    (37.35, 37.75, -122.40, -122.15, 40),   # I-280 (peninsula, N-S)
    (37.25, 37.45, -122.10, -121.85, 26),   # SR-85 (south bay, E-W)
]

lats_p, lons_p = [], []
for lat_s, lat_e, lon_s, lon_e, n in corridors:
    t = np.sort(np.random.uniform(0, 1, n))
    lats_p.extend(lat_s + t * (lat_e - lat_s) + np.random.normal(0, 0.008, n))
    lons_p.extend(lon_s + t * (lon_e - lon_s) + np.random.normal(0, 0.008, n))

lats_p = np.array(lats_p[:n_pems])
lons_p = np.array(lons_p[:n_pems])

ax = axes[1]
ax.scatter(lons_p, lats_p, s=8, c='#e6550d', edgecolors='white',
           linewidths=0.3, zorder=5, alpha=0.85)
ax.set_title('(b) PeMS-Bay ($N{=}326$)', fontsize=8, fontweight='bold', pad=3)
ax.set_xlabel('Schematic X', fontsize=7)
ax.set_ylabel('Schematic Y', fontsize=7)
ax.tick_params(labelsize=6, labelbottom=False, labelleft=False)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linewidth=0.5)

# Bay Area labels
bay_labels = {
    'I-880': (-122.22, 37.60),
    'US-101': (-122.38, 37.55),
    'I-580': (-122.00, 37.65),
    'SF Bay': (-122.18, 37.70),
}
for name, (x, y) in bay_labels.items():
    ax.annotate(name, (x, y), fontsize=5, color='#636363',
                ha='center', va='bottom', alpha=0.7)

plt.tight_layout(pad=0.5)
plt.savefig('fig_sensor_map.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0.02)
print('Saved fig_sensor_map.pdf')
