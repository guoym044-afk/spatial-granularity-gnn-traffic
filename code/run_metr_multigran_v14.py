#!/usr/bin/env python3
"""
METR-LA Multi-Granularity Experiments v14 (V14-T2 + V14-T3)

Extended from v13: 11 granularity levels (5-207), 10 seeds, 3 models.
Uses spectral clustering on DCRNN adjacency to create coarser granularities.

Models: DCRNN, LSTGCNN, GraphWaveNet
Granularities: 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 207
Seeds: 42, 123, 456, 789, 1024, 1337, 1999, 2024, 3141, 6283

Usage:
  # Full run (all models, all granularities, all seeds)
  python run_metr_multigran_v14.py --device cuda

  # Single model
  python run_metr_multigran_v14.py --model graphwavenet --device cuda

  # Single granularity
  python run_metr_multigran_v14.py --model dcrnn --gran 50 --seed 42 --device cuda

  # Parallel: each GPU handles a subset
  CUDA_VISIBLE_DEVICES=0 python run_metr_multigran_v14.py --model dcrnn --device cuda
  CUDA_VISIBLE_DEVICES=1 python run_metr_multigran_v14.py --model lstgcnn --device cuda
  CUDA_VISIBLE_DEVICES=2 python run_metr_multigran_v14.py --model graphwavenet --device cuda
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, '/c20250521/lane_granularity_study/code')

from data_utils import load_metr_la, build_metr_la_granularities, aggregate_nodes
from model import (LSTGCNN, STGCNPredictor, DCRNNPredictor, GraphWaveNetPredictor,
                   train_model, evaluate_model, BaselineModels, compute_metrics)
import torch
import pickle

# Config
DATA_PATH = '/c20250521/lane_granularity_study/data/metr-la.h5'
ADJ_PATH = '/c20250521/lane_granularity_study/data/adj_mx.pkl'
RESULTS_DIR = '/c20250521/lane_granularity_study/results'
SEEDS = [42, 123, 456, 789, 1024, 1337, 1999, 2024, 3141, 6283]
MODELS = ['dcrnn', 'lstgcnn', 'graphwavenet']
TARGET_GRANS = [5, 8, 10, 15, 20, 30, 50, 75, 100, 150]
# 207 = full resolution, handled separately as "all"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def build_model(model_type, num_nodes, adj, horizon):
    if model_type == 'lstgcnn':
        return LSTGCNN(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'dcrnn':
        return DCRNNPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'stgcn':
        return STGCNPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'graphwavenet':
        return GraphWaveNetPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon, hidden_dim=64)
    else:
        raise ValueError(f"Unknown model: {model_type}")

def run_experiment(model_type, n_nodes, labels, X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                   adj_norm, horizon, scaler, seed, device):
    """Run a single experiment and save results."""
    set_seed(seed)

    out_file = os.path.join(RESULTS_DIR,
        f"m5_metr_{model_type}_gran{n_nodes}_seed{seed}.json")
    if os.path.exists(out_file):
        print(f"  SKIP: {model_type} n={n_nodes} seed={seed} (exists)")
        return None

    print(f"\n  Running: {model_type} n={n_nodes} seed={seed}")

    # Build model
    model = build_model(model_type, n_nodes, adj_norm, horizon)
    total_params = sum(p.numel() for p in model.parameters())

    # Compute aggregated scaler for de-normalization
    agg_mean = np.zeros((1, 1, n_nodes), dtype=np.float32)
    agg_std = np.zeros((1, 1, n_nodes), dtype=np.float32)
    for old_n in range(len(labels)):
        new_n = labels[old_n] if n_nodes < len(labels) else old_n
        if new_n < n_nodes:
            agg_mean[0, 0, new_n] += scaler['mean'][old_n]
            agg_std[0, 0, new_n] += scaler['std'][old_n]
    for new_n in range(n_nodes):
        if n_nodes < len(labels):
            cluster_size = np.sum(labels == new_n)
        else:
            cluster_size = 1
        agg_mean[0, 0, new_n] /= max(cluster_size, 1)
        agg_std[0, 0, new_n] /= max(cluster_size, 1)

    # Baseline
    pred_last = BaselineModels.last_value_predictor(X_te, horizon=horizon)
    pred_last_dn = pred_last * agg_std + agg_mean
    Y_te_dn = Y_te * agg_std + agg_mean
    metrics_last = compute_metrics(pred_last_dn, Y_te_dn)
    print(f"  Persistence: MAE={metrics_last['MAE']:.4f}")

    # Train
    t0 = time.time()
    model, train_losses, val_losses = train_model(
        model, X_tr, Y_tr, X_va, Y_va, adj_norm,
        lr=0.001, max_epochs=100, patience=15, device=device)
    train_time = time.time() - t0

    # Evaluate
    metrics_model, _ = evaluate_model(
        model, X_te, Y_te, adj_norm, device=device,
        scaler={'mean': agg_mean.flatten(), 'std': agg_std.flatten()})

    skill = metrics_model['skill_score']
    mae = metrics_model['MAE']
    print(f"  {model_type}: MAE={mae:.4f}, skill={skill:.4f}, time={train_time:.1f}s")

    # Save
    results = {
        'granularity': f'gran{n_nodes}',
        'scene': 'all',
        'seed': seed,
        'model_type': model_type,
        'data_source': 'metr',
        'num_nodes': n_nodes,
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
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {out_file}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser(description='METR-LA Multi-Granularity v14')
    parser.add_argument('--model', type=str, default=None, choices=['dcrnn', 'lstgcnn', 'graphwavenet'],
                        help='Run only this model (default: all)')
    parser.add_argument('--gran', type=int, default=None,
                        help='Run only this granularity (default: all)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed (default: all)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds to override default (e.g., "42,123,456")')
    args = parser.parse_args()

    device = args.device
    models = [args.model] if args.model else MODELS
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS

    # Load data
    print("Loading METR-LA data...")
    X, Y, timestamps, sensor_ids, scaler = load_metr_la(DATA_PATH, normalize=True)
    print(f"  Data: {X.shape[0]} samples, {X.shape[2]} nodes")

    # Load adjacency
    with open(ADJ_PATH, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj_orig = adj_data[2].astype(np.float32)
    full_n = adj_orig.shape[0]
    print(f"  Adjacency: {adj_orig.shape}, {int(np.sum(adj_orig > 0))} edges")

    # Build granularities
    if args.gran and args.gran == full_n:
        gran_labels = {full_n: np.arange(full_n)}
    elif args.gran:
        print(f"Building single granularity k={args.gran}...")
        gran_labels = build_metr_la_granularities(ADJ_PATH, target_counts=[args.gran])
    else:
        print("Building granularities...")
        gran_labels = build_metr_la_granularities(ADJ_PATH, target_counts=TARGET_GRANS)
        gran_labels[full_n] = np.arange(full_n)  # add full resolution

    # Split data
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train+n_val], Y[n_train+n_val:]
    horizon = Y.shape[1]

    # Run experiments
    total_runs = 0
    for n_nodes, labels in sorted(gran_labels.items()):
        if args.gran and n_nodes != args.gran:
            # If --gran specified, only run that one
            actual_k = len(np.unique(labels))
            if actual_k != args.gran:
                continue

        actual_k = len(np.unique(labels))
        print(f"\n{'='*60}")
        print(f"Granularity: {n_nodes} nodes (actual clusters: {actual_k})")
        print(f"{'='*60}")

        if n_nodes < full_n:
            # Aggregate data
            X_tr, Y_tr, adj_agg = aggregate_nodes(X_train, Y_train, adj_orig, labels)
            X_va, Y_va, _ = aggregate_nodes(X_val, Y_val, adj_orig, labels)
            X_te, Y_te, _ = aggregate_nodes(X_test, Y_test, adj_orig, labels)

            # Normalize adjacency
            deg = np.sum(adj_agg, axis=1)
            deg_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0))
            adj_norm = deg_inv_sqrt @ adj_agg @ deg_inv_sqrt

            print(f"  Aggregated: {n_nodes} nodes, adj density: {np.mean(adj_agg > 0):.3f}")
        else:
            # Full resolution
            adj_safe = adj_orig + np.eye(full_n, dtype=np.float32)
            deg = np.sum(adj_safe, axis=1)
            deg_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0))
            adj_norm = deg_inv_sqrt @ adj_safe @ deg_inv_sqrt
            X_tr, Y_tr = X_train, Y_train
            X_va, Y_va = X_val, Y_val
            X_te, Y_te = X_test, Y_test

        for model_type in models:
            for seed in seeds:
                result = run_experiment(
                    model_type, n_nodes, labels, X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                    adj_norm, horizon, scaler, seed, device)
                if result is not None:
                    total_runs += 1

    print(f"\n=== METR-LA MULTI-GRANULARITY v14 COMPLETE ===")
    print(f"Total new runs: {total_runs}")

if __name__ == '__main__':
    main()
