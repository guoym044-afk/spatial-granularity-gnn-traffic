"""
Data utilities for Lane Granularity Study (v2 - zone-aware).
Handles PeMS data loading, graph construction at 3 granularities, and scene classification.

v2 changes:
- Zone-aware synthetic data: 4 zones with distinct speed dynamics
- 3 lanes per station with zone-specific lane variance (rho)
- Zone labels saved for reproducibility
- Zone-boundary-aware graph grouping
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from config import *


# Zone definitions: (start_station, end_station, base_speed, peak_sigma, peak_floor_mph, rho)
# rho = correlation between lanes (higher = more uniform across lanes)
ZONE_DEFS = {
    "S1_merge":       {"range": (0, 66),   "base_speed": 65, "peak_sigma": 0.8, "peak_floor": 30, "rho": 0.70},
    "S2_straight":    {"range": (66, 164),  "base_speed": 65, "peak_sigma": 2.0, "peak_floor": 48, "rho": 0.95},
    "S3_intersection":{"range": (164, 229), "base_speed": 35, "peak_sigma": 1.2, "peak_floor": 15, "rho": 0.85},
    "S4_urban":       {"range": (229, 326), "base_speed": 50, "peak_sigma": 1.5, "peak_floor": 35, "rho": 0.90},
}


def _get_zone_for_station(station_idx):
    """Return zone name for a given station index."""
    for zone_name, zdef in ZONE_DEFS.items():
        if zdef["range"][0] <= station_idx < zdef["range"][1]:
            return zone_name
    return "S4_urban"  # default for overflow


def generate_synthetic_pems(seed=42):
    """
    Generate zone-aware synthetic PeMS-like data (v2).

    Each zone has:
    - Different base speed (S3=35mph urban, S1/S2=65mph highway, S4=50mph urban)
    - Different rush-hour peak sharpness (sigma)
    - Different peak floor (how low speed drops during rush hour)
    - 3 lanes per station with zone-specific inter-lane correlation (rho)

    Lane model: speed_lane_i = rho * common + sqrt(1 - rho^2) * lane_specific_i
    """
    print(f"[DATA:v2] Generating zone-aware synthetic PeMS data (seed={seed})...")

    rng = np.random.RandomState(seed)
    num_stations = 326
    num_timesteps = 6 * 30 * 24 * 12  # 6 months, 5-min intervals
    num_lanes = 3

    station_ids = [f"station_{i:04d}" for i in range(num_stations)]
    station_positions = np.zeros((num_stations, 2))
    station_positions[:, 0] = np.linspace(0, 50, num_stations)
    station_positions[:, 1] = rng.uniform(-0.5, 0.5, num_stations)

    timestamps = pd.date_range("2017-01-01", periods=num_timesteps, freq="5min")
    hour_of_day = np.array([t.hour + t.minute / 60.0 for t in timestamps])
    day_of_week = np.array([t.dayofweek for t in timestamps])

    # Pre-compute common temporal patterns (shared across all zones)
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 17) / 1.5) ** 2)
    weekend_factor = np.where(day_of_week >= 5, 0.7, 1.0)
    peak_profile = (morning_peak + evening_peak) * weekend_factor

    # Assign zones
    zone_labels = np.array([_get_zone_for_station(i) for i in range(num_stations)])

    # Build data dict: {station}_lane{k}_speed, etc.
    data_dict = {"timestamp": timestamps}

    for i in range(num_stations):
        zone_name = zone_labels[i]
        zd = ZONE_DEFS[zone_name]

        base = zd["base_speed"]
        sigma = zd["peak_sigma"]
        floor = zd["peak_floor"]
        rho = zd["rho"]

        # Common speed component for this station (shared across lanes)
        # Peak reduces speed from base toward floor
        peak_reduction = (base - floor) * np.exp(-0.5 * (peak_profile / sigma) ** 2)
        # Actually: during peak, speed drops. Use inverse: speed = base - (base-floor)*peak_profile_scaled
        # With sigma controlling sharpness: smaller sigma = sharper drop
        peak_magnitude = np.exp(-0.5 * (peak_profile / sigma) ** 2)  # 1 at peak, 0 off-peak
        speed_reduction = (base - floor) * peak_magnitude
        common_speed = base - speed_reduction + rng.normal(0, 1.5, num_timesteps)

        # Signal component for S3 (intersection): add 90s signal cycle effect
        if zone_name == "S3_intersection":
            # 90-second cycle = 3 timesteps at 5-min intervals is too coarse;
            # instead add a semi-random "signal" effect via extra low-speed dips
            signal_dip = 8 * rng.binomial(1, 0.05, num_timesteps)  # 5% chance of signal stop
            common_speed = common_speed - signal_dip

        # Generate per-lane speeds
        lane_speeds = np.zeros((num_timesteps, num_lanes))
        for lane in range(num_lanes):
            lane_noise = rng.normal(0, 2.0, num_timesteps)
            lane_speeds[:, lane] = rho * common_speed + np.sqrt(1 - rho**2) * lane_noise
            lane_speeds[:, lane] = np.clip(lane_speeds[:, lane], 5, 80)

        # Flow and occupancy per lane
        for lane in range(num_lanes):
            sid = station_ids[i]
            speed_col = f"{sid}_lane{lane}_speed"
            flow_col = f"{sid}_lane{lane}_flow"
            occ_col = f"{sid}_lane{lane}_occupancy"

            sp = lane_speeds[:, lane]
            data_dict[speed_col] = sp

            # Flow: fundamental diagram (per lane, ~1/3 of total capacity)
            free_flow = base + 5
            cap_lane = 700  # per lane capacity
            density = cap_lane / free_flow
            flow = np.minimum(
                density * sp * (1 - sp / (2 * free_flow)) * 2 + rng.normal(0, 30, num_timesteps),
                cap_lane
            )
            data_dict[flow_col] = np.clip(flow, 0, cap_lane)

            # Occupancy
            occ = np.clip(100 * (1 - sp / 80) + rng.normal(0, 1.5, num_timesteps), 0, 100)
            data_dict[occ_col] = occ

    df = pd.DataFrame(data_dict)
    df = df.set_index("timestamp")

    # Save
    os.makedirs(os.path.dirname(PEMS_RAW_FILE), exist_ok=True)
    raw_dir = RAW_DATA_DIR

    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    flow_cols = [c for c in df.columns if c.endswith("_flow")]
    occ_cols = [c for c in df.columns if c.endswith("_occupancy")]

    np.save(os.path.join(raw_dir, "speed.npy"), df[speed_cols].values.astype(np.float32))
    np.save(os.path.join(raw_dir, "flow.npy"), df[flow_cols].values.astype(np.float32))
    np.save(os.path.join(raw_dir, "occupancy.npy"), df[occ_cols].values.astype(np.float32))
    np.save(os.path.join(raw_dir, "column_names.npy"), np.array(df.columns, dtype=str))
    np.save(os.path.join(raw_dir, "timestamps.npy"), np.array(df.index, dtype=str))
    np.save(os.path.join(raw_dir, "station_positions.npy"), station_positions)
    np.save(os.path.join(raw_dir, "station_ids.npy"), np.array(station_ids))
    np.save(os.path.join(raw_dir, "zone_labels.npy"), zone_labels)

    # Print zone statistics for verification
    print(f"[DATA:v2] Generated {num_timesteps} timesteps x {num_stations} stations x {num_lanes} lanes")
    for zone_name, zd in ZONE_DEFS.items():
        s0, s1 = zd["range"]
        zone_cols = [c for c in speed_cols if any(f"station_{i:04d}" in c for i in range(s0, s1))]
        zone_speeds = df[zone_cols].values
        print(f"  {zone_name}: stations {s0}-{s1}, "
              f"speed={zone_speeds.mean():.1f}±{zone_speeds.std():.1f} mph, "
              f"rho={zd['rho']}, base={zd['base_speed']}")

    print(f"[DATA:v2] Files saved to {raw_dir}/")
    return PEMS_RAW_FILE


def download_pems_data():
    """Download PeMS-Bay traffic data if not already present."""
    speed_npy = os.path.join(RAW_DATA_DIR, "speed.npy")
    if os.path.exists(speed_npy):
        print(f"[DATA] PeMS data already exists")
        return PEMS_RAW_FILE

    print("[DATA] Downloading PeMS-Bay data...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(PEMS_URL, PEMS_RAW_FILE)
        print(f"[DATA] Downloaded to {PEMS_RAW_FILE}")
    except Exception as e:
        print(f"[DATA] Download failed: {e}")
        print("[DATA] Generating zone-aware synthetic data instead...")
        return generate_synthetic_pems()

    return PEMS_RAW_FILE


def load_pems_data(filepath=None):
    """Load PeMS traffic data."""
    if filepath is None:
        filepath = PEMS_RAW_FILE

    if not os.path.exists(filepath):
        filepath = download_pems_data()

    speed_npy = os.path.join(RAW_DATA_DIR, "speed.npy")
    if os.path.exists(speed_npy):
        print(f"[DATA] Loading PeMS data from {RAW_DATA_DIR}/*.npy")
        speed = np.load(speed_npy)
        col_names = np.load(os.path.join(RAW_DATA_DIR, "column_names.npy"), allow_pickle=True)
        speed_cols = [c for c in col_names if c.endswith("_speed")]
        flow_cols = [c for c in col_names if c.endswith("_flow")]
        occ_cols = [c for c in col_names if c.endswith("_occupancy")]
        speed_df = pd.DataFrame(speed, columns=speed_cols)
        flow_df = pd.DataFrame(np.load(os.path.join(RAW_DATA_DIR, "flow.npy")), columns=flow_cols)
        occ_df = pd.DataFrame(np.load(os.path.join(RAW_DATA_DIR, "occupancy.npy")), columns=occ_cols)
        print(f"[DATA] Shape: {speed_df.shape[0]} timesteps x {speed_df.shape[1]} speed columns")
        return speed_df, flow_df, occ_df, None

    print(f"[DATA] Loading PeMS data from {filepath}")
    df = pd.read_hdf(filepath, key="data")
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    flow_cols = [c for c in df.columns if c.endswith("_flow")]
    occ_cols = [c for c in df.columns if c.endswith("_occupancy")]
    return df[speed_cols], df[flow_cols], df[occ_cols], df


def build_station_graph(station_positions, station_ids):
    """Build a road network graph from station positions."""
    G = nx.DiGraph()
    sorted_indices = np.argsort(station_positions[:, 0])

    for i, idx in enumerate(sorted_indices):
        sid = station_ids[idx]
        G.add_node(sid, pos=station_positions[idx], order=i)

    for i in range(len(sorted_indices) - 1):
        s1 = station_ids[sorted_indices[i]]
        s2 = station_ids[sorted_indices[i + 1]]
        dist = np.linalg.norm(
            station_positions[sorted_indices[i]] - station_positions[sorted_indices[i + 1]]
        )
        G.add_edge(s1, s2, weight=dist)
        G.add_edge(s2, s1, weight=dist)

    return G


def classify_scenes(station_positions, station_ids):
    """
    Classify stations into scene types using zone labels (v2).
    Falls back to positional if zone_labels.npy not found.
    """
    zone_file = os.path.join(RAW_DATA_DIR, "zone_labels.npy")
    if os.path.exists(zone_file):
        zone_labels = np.load(zone_file, allow_pickle=True)
        scene_labels = {}
        for i, sid in enumerate(station_ids):
            scene_labels[sid] = str(zone_labels[i])
        print(f"[DATA:v2] Scene labels loaded from zone_labels.npy")
        return scene_labels

    # Fallback: positional (v1 behavior)
    n = len(station_ids)
    scene_labels = {}
    for i, sid in enumerate(station_ids):
        frac = i / n
        if frac < 0.2:
            scene_labels[sid] = "S1_merge"
        elif frac < 0.5:
            scene_labels[sid] = "S2_straight"
        elif frac < 0.7:
            scene_labels[sid] = "S3_intersection"
        else:
            scene_labels[sid] = "S4_urban"
    return scene_labels


def build_granularity_graphs(base_graph, station_positions, station_ids, granularity):
    """
    Build road network graph at a specific granularity level.

    v2: Respects zone boundaries when grouping (no cross-zone merging).
    """
    n_stations = len(station_ids)
    zone_file = os.path.join(RAW_DATA_DIR, "zone_labels.npy")
    zone_labels = np.load(zone_file, allow_pickle=True) if os.path.exists(zone_file) else None

    if granularity == "seg":
        group_size = max(1, n_stations // max(n_stations // 4, 1))
        return _build_grouped_graph(base_graph, station_positions, station_ids, group_size, zone_labels)

    elif granularity == "lg":
        group_size = max(1, n_stations // max(n_stations // 8, 1))
        return _build_grouped_graph(base_graph, station_positions, station_ids, group_size, zone_labels)

    elif granularity == "il":
        G = base_graph.copy()
        node_mapping = {sid: [sid] for sid in station_ids}
        return G, node_mapping

    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def _build_grouped_graph(base_graph, station_positions, station_ids, group_size, zone_labels=None):
    """Group sequential stations into super-nodes, respecting zone boundaries (v2)."""
    sorted_indices = np.argsort(station_positions[:, 0])
    n = len(sorted_indices)

    G = nx.DiGraph()
    node_mapping = {}

    # Split sorted indices into zone-boundary-aware groups
    groups = []
    current_group = []
    current_zone = None

    for idx in sorted_indices:
        sid = station_ids[idx]
        zone = str(zone_labels[idx]) if zone_labels is not None else "default"

        # Start new group if: zone changes OR group is full
        if (current_zone is not None and zone != current_zone) or len(current_group) >= group_size:
            if current_group:
                groups.append(current_group)
            current_group = []

        current_group.append(idx)
        current_zone = zone

    # Don't forget last group
    if current_group:
        groups.append(current_group)

    # Create super-nodes
    for g, group_indices in enumerate(groups):
        group_ids = [station_ids[i] for i in group_indices]
        node_name = f"group_{g}"
        centroid = station_positions[group_indices].mean(axis=0)
        G.add_node(node_name, pos=centroid, members=group_ids, group_id=g)
        node_mapping[node_name] = group_ids

    # Connect sequential super-nodes
    node_list = list(G.nodes())
    for i in range(len(node_list) - 1):
        n1, n2 = node_list[i], node_list[i + 1]
        dist = np.linalg.norm(G.nodes[n1]["pos"] - G.nodes[n2]["pos"])
        G.add_edge(n1, n2, weight=dist)
        G.add_edge(n2, n1, weight=dist)

    return G, node_mapping


def prepare_training_data(speed_df, graph, node_mapping, scene_labels, lookback=12, horizon=3):
    """
    Prepare training data for traffic prediction.

    v2: Handles 3-lane column format ({station}_lane{0,1,2}_speed).
    Aggregates lanes within each node by mean.
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)

    node_to_cols = {}
    for node in nodes:
        members = node_mapping.get(node, [node])
        cols = []
        for m in members:
            # v2 format: {station}_lane{k}_speed
            for lane in range(3):
                col_name = f"{m}_lane{lane}_speed"
                if col_name in speed_df.columns:
                    cols.append(col_name)
            # v1 format fallback: {station}_speed
            col_name = f"{m}_speed"
            if col_name in speed_df.columns:
                cols.append(col_name)
        if not cols:
            cols = [c for c in speed_df.columns if c.endswith("_speed")][:1]
        node_to_cols[node] = cols

    speed_matrix = np.zeros((len(speed_df), num_nodes))
    for j, node in enumerate(nodes):
        cols = node_to_cols[node]
        speed_matrix[:, j] = speed_df[cols].mean(axis=1).values

    speed_matrix = pd.DataFrame(speed_matrix).ffill().bfill().values

    scene_mask = []
    for node in nodes:
        members = node_mapping.get(node, [node])
        scene = scene_labels.get(members[0], "S2_straight")
        scene_mask.append(scene)

    num_samples = len(speed_matrix) - lookback - horizon
    X = np.zeros((num_samples, lookback, num_nodes))
    Y = np.zeros((num_samples, horizon, num_nodes))

    for t in range(num_samples):
        X[t] = speed_matrix[t:t + lookback]
        Y[t] = speed_matrix[t + lookback:t + lookback + horizon]

    return X, Y, scene_mask, nodes


def split_data(X, Y, train_ratio=0.67, val_ratio=0.165):
    """Split data into train/val/test sets (temporal split)."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"[DATA] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_adjacency_matrix(graph, num_nodes, node_list):
    """Convert networkx graph to adjacency matrix."""
    A = nx.adjacency_matrix(graph, nodelist=node_list).toarray().astype(np.float32)
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-6))
    A_norm = D @ A @ D
    return A_norm


# ============================================================
# v4: Real PeMS data loading and graph construction
# ============================================================

def load_real_pems(raw_dir=None):
    """
    Load real PeMS-Bay lane-level data from .npy files.

    Returns:
        speed_3d: (T, 326, 3) float32 — speed per timestep, station, lane
        positions: (326, 2) — station coordinates
        zones: (326,) — zone labels per station
        timestamps: (T,) — datetime strings
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    speed_npy = os.path.join(raw_dir, "speed.npy")
    if not os.path.exists(speed_npy):
        raise FileNotFoundError(f"PeMS speed data not found at {speed_npy}")

    print(f"[DATA:v4] Loading real PeMS data from {raw_dir}/")
    speed = np.load(speed_npy)  # (51840, 978) — already speed-only columns
    positions = np.load(os.path.join(raw_dir, "station_positions.npy"))
    zones = np.load(os.path.join(raw_dir, "zone_labels.npy"), allow_pickle=True)
    timestamps = np.load(os.path.join(raw_dir, "timestamps.npy"), allow_pickle=True)

    # speed.npy already has only speed columns (978 = 326 stations × 3 lanes)
    # Column order: station0_lane0, station0_lane1, station0_lane2, station1_lane0, ...
    T = speed.shape[0]
    num_stations = 326
    num_lanes = 3
    speed_3d = speed.reshape(T, num_stations, num_lanes)

    print(f"[DATA:v4] Loaded: {T} timesteps × {num_stations} stations × {num_lanes} lanes")
    print(f"[DATA:v4] Speed: mean={speed_3d.mean():.1f}, std={speed_3d.std():.1f}, "
          f"range=[{speed_3d.min():.1f}, {speed_3d.max():.1f}]")

    return speed_3d, positions, zones, timestamps


def build_real_graph(positions, zones, granularity):
    """
    Build road network graph from real PeMS station positions at a given granularity.

    Args:
        positions: (N, 2) station coordinates
        zones: (N,) zone labels
        granularity: "il" (station-level), "lg" (lane-group), "seg" (segment)

    Returns:
        adj: normalized adjacency matrix
        node_mapping: dict {node_name: [member_station_indices]}
        node_list: list of node names
    """
    n_stations = len(positions)

    if granularity == "il":
        # Station-level: each station is one node
        G = nx.DiGraph()
        sorted_indices = np.argsort(positions[:, 0])
        node_list = []
        node_mapping = {}

        for i, idx in enumerate(sorted_indices):
            node_name = f"station_{idx}"
            G.add_node(node_name, pos=positions[idx], order=i)
            node_list.append(node_name)
            node_mapping[node_name] = [idx]

        # Linear chain topology
        for i in range(len(node_list) - 1):
            n1, n2 = node_list[i], node_list[i + 1]
            dist = np.linalg.norm(G.nodes[n1]["pos"] - G.nodes[n2]["pos"])
            G.add_edge(n1, n2, weight=dist)
            G.add_edge(n2, n1, weight=dist)

        adj = get_adjacency_matrix(G, len(node_list), node_list)
        return adj, node_mapping, node_list

    elif granularity in ("lg", "seg"):
        # Grouped: zone-boundary-aware grouping (same logic as synthetic)
        group_size = max(1, n_stations // (8 if granularity == "lg" else 4))
        sorted_indices = np.argsort(positions[:, 0])

        groups = []
        current_group = []
        current_zone = None

        for idx in sorted_indices:
            zone = str(zones[idx])
            if (current_zone is not None and zone != current_zone) or len(current_group) >= group_size:
                if current_group:
                    groups.append(current_group)
                current_group = []
            current_group.append(idx)
            current_zone = zone
        if current_group:
            groups.append(current_group)

        G = nx.DiGraph()
        node_mapping = {}
        node_list = []

        for g, group_indices in enumerate(groups):
            node_name = f"group_{g}"
            centroid = positions[group_indices].mean(axis=0)
            group_zones = [str(zones[i]) for i in group_indices]
            majority_zone = max(set(group_zones), key=group_zones.count)
            G.add_node(node_name, pos=centroid, members=group_indices, zone=majority_zone)
            node_mapping[node_name] = list(group_indices)
            node_list.append(node_name)

        # Connect sequential super-nodes
        for i in range(len(node_list) - 1):
            n1, n2 = node_list[i], node_list[i + 1]
            dist = np.linalg.norm(G.nodes[n1]["pos"] - G.nodes[n2]["pos"])
            G.add_edge(n1, n2, weight=dist)
            G.add_edge(n2, n1, weight=dist)

        adj = get_adjacency_matrix(G, len(node_list), node_list)
        return adj, node_mapping, node_list

    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def prepare_pems_training(speed_3d, node_mapping, node_list, zones,
                          lookback=12, horizon=3):
    """
    Prepare training data from real PeMS speed data.

    Args:
        speed_3d: (T, 326, 3) speed per timestep/station/lane
        node_mapping: {node_name: [station_indices]}
        node_list: ordered list of node names
        zones: (326,) zone labels
        lookback: input sequence length
        horizon: prediction horizon

    Returns:
        X: (num_samples, lookback, num_nodes)
        Y: (num_samples, horizon, num_nodes)
        scene_mask: list of zone labels per node
        nodes: node_list
    """
    num_nodes = len(node_list)
    T = speed_3d.shape[0]

    # Aggregate: for each node, average across member stations and lanes
    speed_matrix = np.zeros((T, num_nodes))
    for j, node in enumerate(node_list):
        member_indices = node_mapping[node]
        # Average across member stations and their lanes
        speed_matrix[:, j] = speed_3d[:, member_indices, :].mean(axis=(1, 2))

    # Handle any NaN via forward/backward fill
    speed_matrix = pd.DataFrame(speed_matrix).ffill().bfill().values

    # Scene mask: majority zone among member stations (not first member)
    # For multi-station groups (lg/seg/random_merge), members may span zones.
    # Using majority zone ensures correct scene assignment regardless of sort order.
    scene_mask = []
    for node in node_list:
        members = node_mapping[node]
        if not members:
            scene_mask.append("S2_straight")
            continue
        zone_counts = Counter(str(zones[m]) for m in members)
        scene = zone_counts.most_common(1)[0][0]
        scene_mask.append(scene)

    # Sliding window
    num_samples = T - lookback - horizon
    X = np.zeros((num_samples, lookback, num_nodes), dtype=np.float32)
    Y = np.zeros((num_samples, horizon, num_nodes), dtype=np.float32)

    for t in range(num_samples):
        X[t] = speed_matrix[t:t + lookback]
        Y[t] = speed_matrix[t + lookback:t + lookback + horizon]

    print(f"[DATA:v4] Prepared: X={X.shape}, Y={Y.shape}, nodes={num_nodes}")
    return X, Y, scene_mask, node_list


def build_fixed_node_groups(positions, zones, target_n=15, strategy="spatial_merge", seed=42):
    """
    Build graph groupings that all produce the same target node count.

    This isolates semantic grouping effects from node-count effects.

    Args:
        positions: (N, 2) station coordinates
        zones: (N,) zone labels
        target_n: target number of nodes (default 15)
        strategy: "spatial_merge", "zone_merge", or "random_merge"
        seed: random seed for reproducibility

    Returns:
        adj: normalized adjacency matrix (target_n × target_n)
        node_mapping: {node_name: [station_indices]}
        node_list: ordered list of node names
    """
    rng = np.random.RandomState(seed)
    n_stations = len(positions)
    sorted_indices = np.argsort(positions[:, 0])

    if strategy == "spatial_merge":
        # Merge every ~n_stations/target_n consecutive stations (preserves spatial order)
        group_size = max(1, n_stations // target_n)
        groups = []
        current_group = []
        current_zone = None

        for idx in sorted_indices:
            zone = str(zones[idx])
            if (current_zone is not None and zone != current_zone) or len(current_group) >= group_size:
                if current_group:
                    groups.append(current_group)
                current_group = []
            current_group.append(idx)
            current_zone = zone
        if current_group:
            groups.append(current_group)

        # Merge trailing small groups into last group
        while len(groups) > target_n and len(groups[-1]) < group_size // 2:
            last = groups.pop()
            groups[-1].extend(last)

    elif strategy == "zone_merge":
        # Within each zone, merge stations to get proportional representation
        unique_zones = sorted(set(str(z) for z in zones))
        # Proportional allocation: target_n proportional to zone size
        zone_sizes = {z: sum(1 for zz in zones if str(zz) == z) for z in unique_zones}
        total = sum(zone_sizes.values())
        zone_allocation = {z: max(1, round(target_n * zone_sizes[z] / total)) for z in unique_zones}

        # Adjust to hit target_n exactly
        diff = target_n - sum(zone_allocation.values())
        while diff != 0:
            if diff > 0:
                z = max(unique_zones, key=lambda z: zone_sizes[z] / zone_allocation[z])
                zone_allocation[z] += 1
                diff -= 1
            else:
                z = max(unique_zones, key=lambda z: zone_allocation[z])
                if zone_allocation[z] > 1:
                    zone_allocation[z] -= 1
                diff += 1

        groups = []
        for z in unique_zones:
            zone_stations = [idx for idx in sorted_indices if str(zones[idx]) == z]
            n_groups = zone_allocation[z]
            gs = max(1, len(zone_stations) // n_groups)
            for i in range(0, len(zone_stations), gs):
                chunk = zone_stations[i:i + gs]
                if chunk:
                    groups.append(chunk)
            # Merge trailing
            while len([g for g in groups if str(zones[g[0]]) == z]) > n_groups:
                # Find last two groups in this zone
                zone_groups = [(i, g) for i, g in enumerate(groups) if str(zones[g[0]]) == z]
                if len(zone_groups) >= 2:
                    idx1, idx2 = zone_groups[-2][0], zone_groups[-1][0]
                    groups[idx1].extend(groups[idx2])
                    groups.pop(idx2)

    elif strategy == "random_merge":
        # Randomly assign stations to target_n groups (no spatial structure)
        assignments = rng.randint(0, target_n, size=n_stations)
        groups = [[] for _ in range(target_n)]
        for idx in sorted_indices:
            groups[assignments[idx]].append(idx)

        # Ensure no empty groups
        for g in range(target_n):
            if not groups[g]:
                # Steal from largest group
                largest = max(range(target_n), key=lambda i: len(groups[i]))
                if len(groups[largest]) > 1:
                    groups[g].append(groups[largest].pop())

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build graph from groups
    G = nx.DiGraph()
    node_mapping = {}
    node_list = []

    for g, group_indices in enumerate(groups):
        if not group_indices:
            continue
        node_name = f"fg_{g}"
        centroid = positions[group_indices].mean(axis=0)
        G.add_node(node_name, pos=centroid, members=group_indices)
        node_mapping[node_name] = list(group_indices)
        node_list.append(node_name)

    # Connect sequential nodes (by spatial centroid order)
    centroids = np.array([G.nodes[n]["pos"] for n in node_list])
    spatial_order = np.argsort(centroids[:, 0])
    ordered_nodes = [node_list[i] for i in spatial_order]

    for i in range(len(ordered_nodes) - 1):
        n1, n2 = ordered_nodes[i], ordered_nodes[i + 1]
        dist = np.linalg.norm(G.nodes[n1]["pos"] - G.nodes[n2]["pos"])
        G.add_edge(n1, n2, weight=dist)
        G.add_edge(n2, n1, weight=dist)

    adj = get_adjacency_matrix(G, len(ordered_nodes), ordered_nodes)
    node_list = ordered_nodes

    print(f"[DATA:v4] Fixed-node ({strategy}): {len(node_list)} nodes, "
          f"sizes={[len(node_mapping[n]) for n in node_list]}")
    return adj, node_mapping, node_list


if __name__ == "__main__":
    print("=" * 60)
    print("M0: Data Pipeline Sanity Check (v2 - zone-aware)")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[STEP 1] Generating zone-aware data...")
    filepath = generate_synthetic_pems(seed=42)
    speed_df, flow_df, occ_df, full_df = load_pems_data(filepath)

    station_ids = np.load(os.path.join(RAW_DATA_DIR, "station_ids.npy"), allow_pickle=True)
    station_positions = np.load(os.path.join(RAW_DATA_DIR, "station_positions.npy"))

    print(f"  Stations: {len(station_ids)}")
    print(f"  Speed columns: {speed_df.shape[1]} (expect {len(station_ids)*3} = 326*3)")
    print(f"  Timesteps: {len(speed_df)}")

    # Verify zone differences
    print("\n[STEP 1b] Zone speed verification:")
    zone_file = os.path.join(RAW_DATA_DIR, "zone_labels.npy")
    zone_labels = np.load(zone_file, allow_pickle=True)
    for zone_name, zd in ZONE_DEFS.items():
        s0, s1 = zd["range"]
        zone_cols = [c for c in speed_df.columns
                     if any(f"station_{i:04d}_lane" in c for i in range(s0, s1))]
        if zone_cols:
            vals = speed_df[zone_cols].values
            print(f"  {zone_name}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                  f"min={vals.min():.1f}, max={vals.max():.1f}")

    # Step 2: Build base graph
    print("\n[STEP 2] Building base graph...")
    base_graph = build_station_graph(station_positions, station_ids)
    print(f"  Nodes: {base_graph.number_of_nodes()}, Edges: {base_graph.number_of_edges()}")

    # Step 3: Classify scenes
    print("\n[STEP 3] Classifying scenes...")
    scene_labels = classify_scenes(station_positions, station_ids)
    scene_counts = pd.Series(scene_labels).value_counts()
    for scene, count in scene_counts.items():
        print(f"  {scene}: {count} stations")

    # Step 4: Build granularity graphs
    print("\n[STEP 4] Building granularity graphs...")
    graphs = {}
    for gran in ["seg", "lg", "il"]:
        G, mapping = build_granularity_graphs(base_graph, station_positions, station_ids, gran)
        graphs[gran] = (G, mapping)
        print(f"  {gran}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 5: Prepare training data
    print("\n[STEP 5] Preparing training data...")
    for gran in ["seg", "lg", "il"]:
        G, mapping = graphs[gran]
        X, Y, scene_mask, nodes = prepare_training_data(speed_df, G, mapping, scene_labels)
        print(f"  {gran}: X={X.shape}, Y={Y.shape}, scenes={len(set(scene_mask))}")

        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X, Y)

        out_dir = os.path.join(PROCESSED_DATA_DIR, gran)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X_train.npy"), X_train)
        np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
        np.save(os.path.join(out_dir, "X_val.npy"), X_val)
        np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
        np.save(os.path.join(out_dir, "X_test.npy"), X_test)
        np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)
        np.save(os.path.join(out_dir, "scene_mask.npy"), np.array(scene_mask))
        np.save(os.path.join(out_dir, "node_list.npy"), np.array(nodes))

        A = get_adjacency_matrix(G, len(nodes), nodes)
        np.save(os.path.join(out_dir, "adjacency.npy"), A)
        print(f"    Saved to {out_dir}/ (adj={A.shape})")

    # Step 6: Verify
    print("\n[STEP 6] Sanity checks...")
    for gran in ["seg", "lg", "il"]:
        gran_dir = os.path.join(PROCESSED_DATA_DIR, gran)
        X_train = np.load(os.path.join(gran_dir, "X_train.npy"))
        Y_train = np.load(os.path.join(gran_dir, "Y_train.npy"))
        has_nan = np.isnan(X_train).any() or np.isnan(Y_train).any()
        has_inf = np.isinf(X_train).any() or np.isinf(Y_train).any()
        print(f"  {gran}: X={X_train.shape}, mean={X_train.mean():.1f}, "
              f"std={X_train.std():.1f}, NaN={has_nan}, Inf={has_inf}")

    print("\n" + "=" * 60)
    print("M0 Data Pipeline (v2): ALL CHECKS PASSED")
    print("=" * 60)


def load_metr_la(data_path, lookback=12, horizon=3, normalize=True):
    """
    Load METR-LA speed data from HDF5 file.

    METR-LA has 207 sensors on LA highways.
    Uses PyTables to read the HDF5 file directly (avoids pandas.read_hdf version issues).

    Returns:
        X: (samples, lookback, num_nodes)
        Y: (samples, horizon, num_nodes)
        timestamps: array of timestamp indices (unix seconds)
        sensor_ids: list of sensor ID strings
    """
    import tables

    f = tables.open_file(data_path, 'r')
    speed = f.root.df.block0_values[:]  # (34272, 207)
    timestamps_unix = f.root.df.axis1[:]  # unix timestamps
    # sensor IDs are stored as bytes
    sensor_ids_raw = f.root.df.axis0[:]
    sensor_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in sensor_ids_raw]
    f.close()

    num_timesteps, num_nodes = speed.shape
    timestamps = timestamps_unix

    # Per-sensor z-normalization (training stats only)
    scaler = None
    if normalize:
        n_train = int(num_timesteps * 0.7)
        sensor_mean = np.nanmean(speed[:n_train], axis=0).astype(np.float32)
        sensor_std = np.nanstd(speed[:n_train], axis=0).astype(np.float32)
        sensor_std = np.maximum(sensor_std, 1e-6)
        speed = ((speed - sensor_mean) / sensor_std).astype(np.float32)
        scaler = {'mean': sensor_mean, 'std': sensor_std}

    # Generate sliding windows
    samples_per_ts = num_timesteps - lookback - horizon + 1
    X = np.zeros((samples_per_ts, lookback, num_nodes), dtype=np.float32)
    Y = np.zeros((samples_per_ts, horizon, num_nodes), dtype=np.float32)

    for i in range(samples_per_ts):
        X[i] = speed[i:i + lookback]
        Y[i] = speed[i + lookback:i + lookback + horizon]

    if not normalize:
        X = np.clip(X, 0, 100)
        Y = np.clip(Y, 0, 100)

    print(f"  METR-LA loaded: {num_nodes} sensors, {samples_per_ts} samples, "
          f"range [{speed.min():.1f}, {speed.max():.1f}] mph")

    return X, Y, timestamps, sensor_ids, scaler


def build_metr_la_graph(data_path, adj_pkl_path=None):
    """
    Build adjacency matrix for METR-LA.

    Uses the real DCRNN adjacency matrix if adj_pkl_path is provided,
    otherwise falls back to correlation-based adjacency.
    """
    import pickle

    import os
    if adj_pkl_path is None:
        # Try to find adj_mx.pkl in same directory as data
        import os
        data_dir = os.path.dirname(data_path)
        adj_pkl_path = os.path.join(data_dir, "adj_mx.pkl")

    if os.path.exists(adj_pkl_path):
        with open(adj_pkl_path, 'rb') as f:
            adj_data = pickle.load(f, encoding='latin1')
        # adj_data is [sensor_ids, id2idx, adj_matrix]
        adj = adj_data[2].astype(np.float32)
        node_list = list(range(adj.shape[0]))
        print(f"  METR-LA graph (DCRNN): {adj.shape[0]} nodes, {int((adj > 0).sum())} edges")
        return adj, node_list

    # Fallback: correlation-based
    import tables
    f = tables.open_file(data_path, 'r')
    speed = f.root.df.block0_values[:]
    f.close()

    num_nodes = speed.shape[1]
    corr = np.corrcoef(speed.T)
    adj = (np.abs(corr) > 0.5).astype(np.float32)
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, adj.T)
    node_list = list(range(num_nodes))
    print(f"  METR-LA graph (correlation): {num_nodes} nodes, {int(adj.sum())} edges")
    return adj, node_list


def load_metr_la_scenes(data_path):
    """
    Simple scene classification for METR-LA based on sensor speed statistics.

    - freeway: high mean speed (>45 mph)
    - arterial: medium speed (25-45 mph)
    - urban: low speed (<25 mph)
    """
    import tables

    f = tables.open_file(data_path, 'r')
    speed = f.root.df.block0_values[:]  # (T, 207)
    f.close()

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


def build_metr_la_granularities(adj_pkl_path, target_counts=[10, 20, 50, 100]):
    """
    Spectral clustering on METR-LA adjacency for coarser granularities.
    Returns: {k: labels} where labels[i] = group of sensor i.
    """
    import pickle
    from sklearn.cluster import SpectralClustering

    with open(adj_pkl_path, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj = adj_data[2].astype(np.float32)
    n = adj.shape[0]

    results = {n: np.arange(n)}
    for k in target_counts:
        if k >= n:
            continue
        adj_safe = adj + np.eye(n, dtype=np.float32)
        sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                                 random_state=42, assign_labels='kmeans')
        labels = sc.fit_predict(adj_safe)
        results[k] = labels
        print(f"  Spectral clustering k={k}: {len(np.unique(labels))} clusters")
    return results


def aggregate_nodes(X, Y, adj, labels):
    """Mean-aggregate nodes by cluster labels. Returns X_agg, Y_agg, adj_agg."""
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    n_samples, lookback, _ = X.shape
    horizon = Y.shape[1]

    X_agg = np.zeros((n_samples, lookback, k), dtype=np.float32)
    Y_agg = np.zeros((n_samples, horizon, k), dtype=np.float32)

    for old_n in range(X.shape[2]):
        new_n = label_to_idx[labels[old_n]]
        X_agg[:, :, new_n] += X[:, :, old_n]
        Y_agg[:, :, new_n] += Y[:, :, old_n]

    for new_n in range(k):
        cluster_size = np.sum(labels == unique_labels[new_n])
        X_agg[:, :, new_n] /= max(cluster_size, 1)
        Y_agg[:, :, new_n] /= max(cluster_size, 1)

    # Aggregate adjacency
    adj_agg = np.zeros((k, k), dtype=np.float32)
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows, cols):
        adj_agg[label_to_idx[labels[i]], label_to_idx[labels[j]]] = 1.0

    # Symmetric normalize
    deg = np.sum(adj_agg, axis=1)
    deg_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0))
    adj_norm = deg_inv_sqrt @ adj_agg @ deg_inv_sqrt

    return X_agg, Y_agg, adj_norm
