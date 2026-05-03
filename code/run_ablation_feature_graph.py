#!/usr/bin/env python3
"""
Feature-vs-Graph Ablation Experiment
Isolate temporal smoothing (feature averaging) from spatial information loss (graph compression).

4 conditions at N in {50, 100, 150} on METR-LA:
  - Coarsened: mean-pool features + coarsen graph (reuse existing results)
  - Feature-only: mean-pool features, keep original 207-node graph
  - Graph-only: original features at cluster centroids, coarsen graph
  - Sensor-subset: select N sensors, no averaging (reuse Experiment 1 results)

New runs: Feature-only + Graph-only = 2 conditions * 3 N * 2 models * 3 seeds = 36 runs

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_ablation_feature_graph.py --model dcrnn --condition feature_only --device cuda
  CUDA_VISIBLE_DEVICES=1 python run_ablation_feature_graph.py --model graphwavenet --condition graph_only --device cuda
  python run_ablation_feature_graph.py --model dcrnn --condition feature_only --n 50 --seed 42 --device cuda
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, '/c20250521/lane_granularity_study/code')

from data_utils import (load_metr_la, build_metr_la_granularities,
                        build_feature_only_data, build_graph_only_data)
from model import (DCRNNPredictor, GraphWaveNetPredictor,
                   train_model, evaluate_model, BaselineModels, compute_metrics)
import torch
import pickle

DATA_PATH = '/c20250521/lane_granularity_study/data/metr-la.h5'
ADJ_PATH = '/c20250521/lane_granularity_study/data/adj_mx.pkl'
RESULTS_DIR = '/c20250521/lane_granularity_study/results'
SEEDS = [42, 123, 456]
TARGET_NS = [50, 100, 150]
CONDITIONS = ['feature_only', 'graph_only']


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_model(model_type, num_nodes, adj, horizon):
    if model_type == 'dcrnn':
        return DCRNNPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'graphwavenet':
        return GraphWaveNetPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon, hidden_dim=64)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def run_one(model_type, condition, n_nodes, X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
            adj_norm, horizon, scaler_dict, seed, device):
    """Run a single experiment."""
    set_seed(seed)

    num_nodes_graph = X_tr.shape[2]
    out_file = os.path.join(RESULTS_DIR,
        f"m5_metr_{model_type}_ablation_{condition}_n{n_nodes}_seed{seed}.json")
    if os.path.exists(out_file):
        print(f"  SKIP: {model_type} {condition} n={n_nodes} seed={seed} (exists)")
        return None

    print(f"\n  Running: {model_type} {condition} n={n_nodes} (graph={num_nodes_graph} nodes) seed={seed}")
    t0 = time.time()

    model = build_model(model_type, num_nodes_graph, adj_norm, horizon)
    total_params = sum(p.numel() for p in model.parameters())

    # Persistence baseline
    pred_last = BaselineModels.last_value_predictor(X_te, horizon=horizon)
    Y_te_dn = Y_te * scaler_dict['std'] + scaler_dict['mean']
    pred_last_dn = pred_last * scaler_dict['std'] + scaler_dict['mean']
    metrics_last = compute_metrics(pred_last_dn, Y_te_dn)
    print(f"  Persistence: MAE={metrics_last['MAE']:.4f}")

    # Train
    model, train_losses, val_losses = train_model(
        model, X_tr, Y_tr, X_va, Y_va, adj_norm,
        lr=0.001, max_epochs=100, patience=15, device=device)
    train_time = time.time() - t0

    # Evaluate
    metrics_model, _ = evaluate_model(
        model, X_te, Y_te, adj_norm, device=device, scaler=scaler_dict)

    skill = metrics_model['skill_score']
    mae = metrics_model['MAE']
    print(f"  {model_type}: MAE={mae:.4f}, skill={skill:.4f}, time={train_time:.1f}s")

    result = {
        'granularity': f'ablation_{condition}_n{n_nodes}',
        'experiment': 'ablation',
        'condition': condition,
        'target_nodes': n_nodes,
        'graph_nodes': num_nodes_graph,
        'scene': 'all',
        'seed': seed,
        'model_type': model_type,
        'data_source': 'metr',
        'num_nodes': num_nodes_graph,
        'total_params': total_params,
        'train_time_sec': round(train_time, 1),
        model_type: {k: float(v) if v is not None and not isinstance(v, np.ndarray) else None
                     for k, v in metrics_model.items()},
        'baselines': {
            'last_value': {k: float(v) if v is not None and not isinstance(v, np.ndarray) else None
                           for k, v in metrics_last.items()},
        },
        'best_val_epoch': len(train_losses) - 15 if len(val_losses) > 15 else len(val_losses),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [SAVED] {out_file}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['dcrnn', 'graphwavenet'])
    parser.add_argument('--condition', type=str, required=True, choices=CONDITIONS)
    parser.add_argument('--n', type=int, default=None, help='Target node count (default: all)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: all)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("Loading METR-LA data...")
    X_raw, Y_raw, timestamps, sensor_ids, scaler_full = load_metr_la(DATA_PATH, normalize=True)

    with open(ADJ_PATH, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj_orig = adj_data[2].astype(np.float32)

    # Time-series split: 70/15/15
    n_total = X_raw.shape[0]
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    X_train, Y_train = X_raw[:n_train], Y_raw[:n_train]
    X_val, Y_val = X_raw[n_train:n_train+n_val], Y_raw[n_train:n_train+n_val]
    X_test, Y_test = X_raw[n_train+n_val:], Y_raw[n_train+n_val:]

    horizon = Y_train.shape[1]
    full_n = X_train.shape[2]
    print(f"Data loaded: train={X_train.shape}, full_n={full_n}, horizon={horizon}")

    # Spectral clustering labels
    print("Computing spectral clustering...")
    granularities = build_metr_la_granularities(ADJ_PATH, target_counts=TARGET_NS)

    target_ns = [args.n] if args.n is not None else TARGET_NS
    seeds = [args.seed] if args.seed is not None else SEEDS
    total_runs = 0

    for n_nodes in target_ns:
        print(f"\n{'='*60}")
        print(f"N = {n_nodes}, Condition = {args.condition}")
        print(f"{'='*60}")

        if n_nodes not in granularities:
            print(f"  WARNING: No clustering for n={n_nodes}, skipping")
            continue
        labels = granularities[n_nodes]

        for seed in seeds:
            if args.condition == 'feature_only':
                # Mean-pool features within clusters, keep full 207-node graph
                X_tr_fo, Y_tr_fo, adj_norm = build_feature_only_data(X_train, Y_train, adj_orig, labels)
                X_va_fo, Y_va_fo, _ = build_feature_only_data(X_val, Y_val, adj_orig, labels)
                X_te_fo, Y_te_fo, _ = build_feature_only_data(X_test, Y_test, adj_orig, labels)

                # Vary seed by shuffling training data order
                set_seed(seed)
                perm = np.random.permutation(X_tr_fo.shape[0])
                X_tr_fo = X_tr_fo[perm]
                Y_tr_fo = Y_tr_fo[perm]

                # Scaler: feature-only still uses original Y targets at 207 nodes
                scaler_fo = {'mean': scaler_full['mean'], 'std': scaler_full['std']}

                result = run_one(args.model, args.condition, n_nodes,
                               X_tr_fo, Y_tr_fo, X_va_fo, Y_va_fo,
                               X_te_fo, Y_te_fo, adj_norm,
                               horizon, scaler_fo, seed, args.device)

            elif args.condition == 'graph_only':
                # Select cluster centroids, original features, coarsened graph
                X_tr_go, Y_tr_go, adj_norm, centroids = build_graph_only_data(
                    X_train, Y_train, adj_orig, labels)
                X_va_go, Y_va_go, _, _ = build_graph_only_data(
                    X_val, Y_val, adj_orig, labels)
                X_te_go, Y_te_go, _, _ = build_graph_only_data(
                    X_test, Y_test, adj_orig, labels)

                set_seed(seed)
                perm = np.random.permutation(X_tr_go.shape[0])
                X_tr_go = X_tr_go[perm]
                Y_tr_go = Y_tr_go[perm]

                # Scaler: graph-only uses coarsened Y, need aggregated scaler
                # Compute per-cluster mean/std from training data
                from data_utils import aggregate_nodes
                _, _, adj_agg = aggregate_nodes(
                    np.zeros((1, 1, full_n)), np.zeros((1, 1, full_n)), adj_orig, labels)
                # Use the centroid sensors' scaler values
                scaler_go = {
                    'mean': scaler_full['mean'][centroids],
                    'std': scaler_full['std'][centroids],
                }

                result = run_one(args.model, args.condition, n_nodes,
                               X_tr_go, Y_tr_go, X_va_go, Y_va_go,
                               X_te_go, Y_te_go, adj_norm,
                               horizon, scaler_go, seed, args.device)

            if result is not None:
                total_runs += 1

    print(f"\nTotal new runs: {total_runs}")


if __name__ == '__main__':
    main()
