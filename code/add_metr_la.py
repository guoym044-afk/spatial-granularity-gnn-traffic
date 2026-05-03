"""
Patch data_utils.py to add load_metr_la() function.
Also patch config.py to add METR-LA paths.
Upload + run this script on the server to apply patches.
"""
import os

# ============================================================
# 1. Patch config.py: add METR-LA paths
# ============================================================
config_addition = """
# === METR-LA Dataset ===
METR_LA_RAW = os.path.join(DATA_DIR, "metr-la.h5")
"""

config_path = "/c20250521/lane_granularity_study/code/config.py"
with open(config_path, 'r') as f:
    cfg = f.read()

if "METR_LA_RAW" not in cfg:
    # Insert after PROCESSED_DATA_DIR line
    cfg = cfg.replace(
        'PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")',
        'PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")\nMETR_LA_RAW = os.path.join(DATA_DIR, "metr-la.h5")'
    )
    with open(config_path, 'w') as f:
        f.write(cfg)
    print("Patched config.py: added METR_LA_RAW")
else:
    print("config.py already has METR_LA_RAW")

# ============================================================
# 2. Patch data_utils.py: add load_metr_la() at the end
# ============================================================
metr_la_loader = '''

def load_metr_la(data_path, lookback=12, horizon=3):
    """
    Load METR-LA speed data from HDF5 file.

    METR-LA has 207 sensors on LA highways.
    Data format: pandas DataFrame (timestamps x sensor_ids), values are speed in mph.

    Returns:
        X: (samples, lookback, num_nodes)
        Y: (samples, horizon, num_nodes)
        timestamps: array of timestamp indices
        sensor_ids: list of sensor IDs
    """
    import pandas as pd

    df = pd.read_hdf(data_path)
    speed = df.values  # (T, 207)
    timestamps = df.index
    sensor_ids = list(df.columns)

    num_timesteps, num_nodes = speed.shape

    # Generate sliding windows
    samples_per_ts = num_timesteps - lookback - horizon + 1
    X = np.zeros((samples_per_ts, lookback, num_nodes), dtype=np.float32)
    Y = np.zeros((samples_per_ts, horizon, num_nodes), dtype=np.float32)

    for i in range(samples_per_ts):
        X[i] = speed[i:i + lookback]
        Y[i] = speed[i + lookback:i + lookback + horizon]

    # Clip to reasonable speed range
    X = np.clip(X, 0, 100)
    Y = np.clip(Y, 0, 100)

    print(f"  METR-LA loaded: {num_nodes} sensors, {samples_per_ts} samples, "
          f"range [{speed.min():.1f}, {speed.max():.1f}] mph")

    return X, Y, timestamps, sensor_ids


def build_metr_la_graph(data_path):
    """
    Build adjacency matrix for METR-LA from the HDF5 file columns.

    Uses distance-based threshold (same as DCRNN paper: Gaussian kernel with threshold 0.1).
    Since we don't have sensor locations readily, build a simple k-NN graph
    based on correlation of speed time series.
    """
    import pandas as pd

    df = pd.read_hdf(data_path)
    speed = df.values  # (T, 207)
    num_nodes = speed.shape[1]

    # Use correlation-based adjacency (fallback when no spatial coords available)
    corr = np.corrcoef(speed.T)  # (207, 207)
    # Threshold: keep edges with |corr| > 0.5
    adj = (np.abs(corr) > 0.5).astype(np.float32)
    np.fill_diagonal(adj, 0)

    # Make symmetric
    adj = np.maximum(adj, adj.T)

    # Node list (sensor indices)
    node_list = list(range(num_nodes))

    print(f"  METR-LA graph: {num_nodes} nodes, {int(adj.sum())} edges")

    return adj, node_list


def load_metr_la_scenes(data_path):
    """
    Simple scene classification for METR-LA based on sensor speed statistics.

    Since METR-LA doesn't have lane-level info, we classify sensors by their
    mean speed and speed variance:
    - freeway: high mean speed (>45 mph), low variance
    - arterial: medium speed (25-45 mph), medium variance
    - urban: low speed (<25 mph), high variance
    """
    import pandas as pd

    df = pd.read_hdf(data_path)
    speed = df.values

    mean_speed = np.nanmean(speed, axis=0)
    std_speed = np.nanstd(speed, axis=0)

    scenes = []
    for i in range(len(mean_speed)):
        if mean_speed[i] > 45:
            scenes.append("S1_freeway")
        elif mean_speed[i] > 25:
            scenes.append("S2_arterial")
        else:
            scenes.append("S3_urban")

    scene_counts = {}
    for s in scenes:
        scene_counts[s] = scene_counts.get(s, 0) + 1
    print(f"  METR-LA scenes: {scene_counts}")

    return scenes
'''

data_utils_path = "/c20250521/lane_granularity_study/code/data_utils.py"
with open(data_utils_path, 'r') as f:
    duf = f.read()

if "def load_metr_la" not in duf:
    with open(data_utils_path, 'a') as f:
        f.write(metr_la_loader)
    print("Patched data_utils.py: added load_metr_la, build_metr_la_graph, load_metr_la_scenes")
else:
    print("data_utils.py already has load_metr_la")

print("\nDone! Both files patched.")
