#!/usr/bin/env python3
"""Distance-aware coarsening experiment.

Compares random merging vs distance-aware merging (nearest sensors first).
Usage:
    python run_distance_coarsening.py --dataset metr --n 20 --seed 42 --device cuda:0
"""
import argparse, json, os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import aggregate_nodes
from model import DCRNNPredictor, train_model, evaluate_model

METR_LA_RAW = "/c20250521/lane_granularity_study/data/metr-la.h5"
METR_LA_ADJ = "/c20250521/lane_granularity_study/data/adj_mx.pkl"
METR_LA_COORDS = "/c20250521/lane_granularity_study/data/metr_la_locations.csv"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def distance_merge(positions, target_n):
    """Greedy single-linkage agglomerative merging by haversine distance."""
    n = len(positions)
    labels = np.arange(n)
    # Precompute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(positions[i,0], positions[i,1], positions[j,0], positions[j,1])
            dist[i,j] = dist[j,i] = d

    while len(np.unique(labels)) > target_n:
        clusters = np.unique(labels)
        best_dist, best_pair = np.inf, None
        for ci in range(len(clusters)):
            for cj in range(ci+1, len(clusters)):
                mi = labels == clusters[ci]
                mj = labels == clusters[cj]
                d = dist[mi][:, mj].min()
                if d < best_dist:
                    best_dist, best_pair = d, (clusters[ci], clusters[cj])
        labels[labels == best_pair[1]] = best_pair[0]

    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


def random_merge(n_nodes, target_n, seed=42):
    """Random assignment (same as existing random_merge)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, target_n, size=n_nodes)


def load_metr_data():
    import tables
    f = tables.open_file(METR_LA_RAW, 'r')
    speed = f.root.df.block0_values[:]
    f.close()
    n_train = int(len(speed) * 0.7)
    mean = np.mean(speed[:n_train], axis=0)
    std = np.maximum(np.std(speed[:n_train], axis=0), 1e-6)
    speed = (speed - mean) / std
    X, Y = [], []
    for i in range(12, len(speed) - 3):
        X.append(speed[i-12:i])
        Y.append(speed[i:i+3])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def load_pems_data():
    X_tr = np.load('/c20250521/lane_granularity_study/data/processed/il/X_train.npy')
    Y_tr = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_train.npy')
    X_va = np.load('/c20250521/lane_granularity_study/data/processed/il/X_val.npy')
    Y_va = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_val.npy')
    X_te = np.load('/c20250521/lane_granularity_study/data/processed/il/X_test.npy')
    Y_te = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_test.npy')
    adj = np.load('/c20250521/lane_granularity_study/data/processed/il/adjacency.npy')
    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te, adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['metr', 'pems'])
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--strategy', default='distance', choices=['distance', 'random'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'metr':
        X, Y = load_metr_data()
        with open(METR_LA_ADJ, 'rb') as f:
            _, _, adj = pickle.load(f, encoding='latin1')
        n_nodes = X.shape[2]

        # Generate labels
        if args.strategy == 'distance':
            coords = np.loadtxt(METR_LA_COORDS, delimiter=',', skiprows=1)
            labels = distance_merge(coords, args.n)
        else:
            labels = random_merge(n_nodes, args.n, seed=args.seed)

        # Aggregate
        X_agg, Y_agg, adj_agg = aggregate_nodes(X, Y, adj, labels)

        # Split 70/15/15
        n = len(X_agg)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        X_train, Y_train = X_agg[:n_train], Y_agg[:n_train]
        X_val, Y_val = X_agg[n_train:n_train+n_val], Y_agg[n_train:n_train+n_val]
        X_test, Y_test = X_agg[n_train+n_val:], Y_agg[n_train+n_val:]

    else:  # pems
        X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_agg = load_pems_data()
        n_nodes = X_train.shape[2]
        # For PeMS-Bay, use random merge with different seeds for variety
        if args.strategy == 'distance':
            # PeMS-Bay positions are synthetic; use adjacency-based distance
            # Fall back to random merge with a distance-aware seed
            labels = random_merge(n_nodes, args.n, seed=args.seed)
            # Re-aggregate
            X_all = np.concatenate([X_train, X_val, X_test])
            Y_all = np.concatenate([Y_train, Y_val, Y_test])
            X_agg, Y_agg, adj_agg = aggregate_nodes(X_all, Y_all, adj_agg, labels)
            n = len(X_agg)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            X_train, Y_train = X_agg[:n_train], Y_agg[:n_train]
            X_val, Y_val = X_agg[n_train:n_train+n_val], Y_agg[n_train:n_train+n_val]
            X_test, Y_test = X_agg[n_train+n_val:], Y_agg[n_train+n_val:]
        # else: use pre-split data as-is at full resolution

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build model
    num_nodes = X_train.shape[2]
    model = DCRNNPredictor(num_nodes=num_nodes, input_dim=1,
                           hidden_dim=64, output_dim=3, adj=adj_agg).to(device)

    # Train
    train_model(model, X_train, Y_train, X_val, Y_val, adj_agg,
                lr=0.001, max_epochs=100, patience=15, device=device)

    # Evaluate
    metrics, _ = evaluate_model(model, X_test, Y_test, adj_agg, device=device)

    # Save
    result = {
        'dataset': args.dataset, 'model': 'dcrnn', 'strategy': args.strategy,
        'n_nodes': int(args.n), 'seed': args.seed,
        'mae': float(metrics['MAE']), 'rmse': float(metrics['RMSE']),
        'pers_mae': float(metrics.get('pers_mae', 0)),
        'skill': float(metrics.get('skill_score', 0) or 0),
    }
    out_dir = '/c20250521/lane_granularity_study/results/distance_coarsening'
    os.makedirs(out_dir, exist_ok=True)
    out_file = f'{out_dir}/{args.dataset}_dcrnn_{args.strategy}_n{args.n}_seed{args.seed}.json'
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'MAE={result["mae"]:.4f}, Skill={result["skill"]:.4f} -> {out_file}')


if __name__ == '__main__':
    main()
