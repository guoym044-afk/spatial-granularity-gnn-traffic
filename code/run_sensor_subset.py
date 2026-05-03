#!/usr/bin/env python3
"""
Sensor Subset Baseline Experiment
Select N random sensors from METR-LA (no averaging), train and evaluate.
Compares with coarsened results at same N to quantify smoothing artifact.

Config:
  N in {5, 10, 20, 50, 100, 150, 207}
  Models: DCRNN, GraphWaveNet
  Seeds: 42, 123, 456
  Total: 7 * 2 * 3 = 42 runs

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_sensor_subset.py --model dcrnn --device cuda
  CUDA_VISIBLE_DEVICES=1 python run_sensor_subset.py --model graphwavenet --device cuda
  python run_sensor_subset.py --model dcrnn --n 50 --seed 42 --device cuda
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, '/c20250521/lane_granularity_study/code')

from data_utils import load_metr_la, select_sensor_subset
from model import (DCRNNPredictor, GraphWaveNetPredictor,
                   train_model, evaluate_model, BaselineModels, compute_metrics)
import torch
import pickle

DATA_PATH = '/c20250521/lane_granularity_study/data/metr-la.h5'
ADJ_PATH = '/c20250521/lane_granularity_study/data/adj_mx.pkl'
RESULTS_DIR = '/c20250521/lane_granularity_study/results'
SEEDS = [42, 123, 456]
TARGET_NS = [5, 10, 20, 50, 100, 150, 207]


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


def run_one(model_type, n_nodes, X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
            adj_norm, horizon, scaler_dict, seed, device):
    """Run a single experiment."""
    set_seed(seed)

    out_file = os.path.join(RESULTS_DIR,
        f"m5_metr_{model_type}_subset{n_nodes}_seed{seed}.json")
    if os.path.exists(out_file):
        print(f"  SKIP: {model_type} subset n={n_nodes} seed={seed} (exists)")
        return None

    print(f"\n  Running: {model_type} subset n={n_nodes} seed={seed}")
    t0 = time.time()

    # Build model
    model = build_model(model_type, n_nodes, adj_norm, horizon)
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

    # Save
    result = {
        'granularity': f'subset{n_nodes}',
        'experiment': 'sensor_subset',
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
        json.dump(result, f, indent=2)
    print(f"  [SAVED] {out_file}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['dcrnn', 'graphwavenet'])
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
    X_train_full, Y_train_full = X_raw[:n_train], Y_raw[:n_train]
    X_val_full, Y_val_full = X_raw[n_train:n_train+n_val], Y_raw[n_train:n_train+n_val]
    X_test_full, Y_test_full = X_raw[n_train+n_val:], Y_raw[n_train+n_val:]

    horizon = Y_train_full.shape[1]
    print(f"Data loaded: train={X_train_full.shape}, horizon={horizon}")

    target_ns = [args.n] if args.n is not None else TARGET_NS
    seeds = [args.seed] if args.seed is not None else SEEDS
    total_runs = 0

    for n_nodes in target_ns:
        print(f"\n{'='*60}")
        print(f"N = {n_nodes}")
        print(f"{'='*60}")

        for seed in seeds:
            # Select sensor subset (same selection for train/val/test)
            X_tr, Y_tr, adj_norm, selected = select_sensor_subset(
                X_train_full, Y_train_full, adj_orig, n_nodes, seed=seed)
            X_va, Y_va, _, _ = select_sensor_subset(
                X_val_full, Y_val_full, adj_orig, n_nodes, seed=seed)
            X_te, Y_te, _, _ = select_sensor_subset(
                X_test_full, Y_test_full, adj_orig, n_nodes, seed=seed)

            # Subset scaler to selected sensors
            scaler_sub = {
                'mean': scaler_full['mean'][selected],
                'std': scaler_full['std'][selected],
            }

            result = run_one(args.model, n_nodes, X_tr, Y_tr, X_va, Y_va,
                           X_te, Y_te, adj_norm, horizon, scaler_sub, seed, args.device)
            if result is not None:
                total_runs += 1

    print(f"\nTotal new runs: {total_runs}")


if __name__ == '__main__':
    main()
