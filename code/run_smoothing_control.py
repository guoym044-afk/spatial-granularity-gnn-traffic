#!/usr/bin/env python3
"""
P1-1: Same-N Smoothing Control Experiment
Fixed N=207, Gaussian smoothing along time axis.
3 models × 3 sigma × 3 seeds = 27 runs.
"""
import sys
import os
import json
import time
import argparse
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '/c20250521/lane_granularity_study/code')
from data_utils import load_metr_la
from model import (LSTGCNN, DCRNNPredictor, GraphWaveNetPredictor,
                   train_model, evaluate_model, BaselineModels, compute_metrics)
import torch

DATA_PATH = '/c20250521/lane_granularity_study/data/metr-la.h5'
ADJ_PATH = '/c20250521/lane_granularity_study/data/adj_mx.pkl'
RESULTS_DIR = '/c20250521/lane_granularity_study/results'
SEEDS = [42, 123, 456]
SIGMAS = [0, 1, 3]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def apply_temporal_smoothing(data, sigma):
    if sigma == 0:
        return data
    smoothed = np.zeros_like(data)
    for n in range(data.shape[1]):
        for f in range(data.shape[2]):
            smoothed[:, n, f] = gaussian_filter1d(data[:, n, f], sigma=sigma)
    return smoothed

def build_model(model_type, num_nodes, adj, horizon):
    if model_type == 'lstgcnn':
        return LSTGCNN(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'dcrnn':
        return DCRNNPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon)
    elif model_type == 'graphwavenet':
        return GraphWaveNetPredictor(num_nodes=num_nodes, adj=adj, output_dim=horizon, hidden_dim=64)
    else:
        raise ValueError("Unknown model: " + model_type)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['dcrnn', 'lstgcnn', 'graphwavenet'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sigma', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    print("Loading METR-LA data...")
    X_raw, Y_raw, timestamps, sensor_ids, scaler = load_metr_la(DATA_PATH, normalize=True)

    # Load adjacency
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

    full_n = X_train.shape[2]
    horizon = Y_train.shape[1]
    print("Data: train=%s, N=%d, horizon=%d" % (str(X_train.shape), full_n, horizon))

    # Normalized adjacency
    adj_safe = adj_orig + np.eye(full_n, dtype=np.float32)
    deg = np.sum(adj_safe, axis=1)
    deg_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0))
    adj_norm = deg_inv_sqrt @ adj_safe @ deg_inv_sqrt

    sigmas = [args.sigma] if args.sigma is not None else SIGMAS
    seeds = [args.seed] if args.seed is not None else SEEDS
    total_runs = 0

    for sigma in sigmas:
        print("\n" + "="*60)
        print("Sigma=%s" % sigma)
        print("="*60)

        if sigma > 0:
            X_tr_s = apply_temporal_smoothing(X_train, sigma)
            Y_tr_s = apply_temporal_smoothing(Y_train, sigma)
            X_va_s = apply_temporal_smoothing(X_val, sigma)
            Y_va_s = apply_temporal_smoothing(Y_val, sigma)
            X_te_s = apply_temporal_smoothing(X_test, sigma)
            Y_te_s = apply_temporal_smoothing(Y_test, sigma)
        else:
            X_tr_s, Y_tr_s = X_train, Y_train
            X_va_s, Y_va_s = X_val, Y_val
            X_te_s, Y_te_s = X_test, Y_test

        for seed in seeds:
            out_file = os.path.join(RESULTS_DIR,
                "m5_metr_%s_smooth_s%s_seed%d.json" % (args.model, sigma, seed))
            if os.path.exists(out_file):
                print("  SKIP: sigma=%d seed=%d (exists)" % (sigma, seed))
                continue

            set_seed(int(seed))

            # Persistence baseline
            pred_last = BaselineModels.last_value_predictor(X_te_s, horizon=horizon)
            agg_mean = scaler['mean'].reshape(1, 1, -1).astype(np.float32)
            agg_std = scaler['std'].reshape(1, 1, -1).astype(np.float32)
            pred_last_dn = pred_last * agg_std + agg_mean
            Y_te_dn = Y_te_s * agg_std + agg_mean
            metrics_last = compute_metrics(pred_last_dn, Y_te_dn)

            # Build and train model
            model = build_model(args.model, full_n, adj_norm, horizon)
            total_params = sum(p.numel() for p in model.parameters())

            t0 = time.time()
            model, train_losses, val_losses = train_model(
                model, X_tr_s, Y_tr_s, X_va_s, Y_va_s, adj_norm,
                lr=0.001, max_epochs=100, patience=15, device=args.device)
            train_time = time.time() - t0

            metrics_model, _ = evaluate_model(
                model, X_te_s, Y_te_s, adj_norm, device=args.device,
                scaler={'mean': scaler['mean'].flatten(), 'std': scaler['std'].flatten()})

            skill = 1 - metrics_model['MAE'] / metrics_last['MAE']
            best_epoch = int(np.argmin(val_losses)) + 1

            result = {
                'granularity': 'smooth_s%s' % str(sigma).replace('.','p'),
                'scene': 'all',
                'seed': seed,
                'model_type': args.model,
                'data_source': 'metr',
                'sigma': sigma,
                'num_nodes': full_n,
                'total_params': total_params,
                'train_time_sec': round(train_time, 1),
                args.model: {
                    'MAE': float(metrics_model['MAE']),
                    'RMSE': float(metrics_model['RMSE']),
                    'MAPE': float(metrics_model['MAPE']),
                    'skill_score': float(skill),
                },
                'baselines': {
                    'last_value': {
                        'MAE': float(metrics_last['MAE']),
                        'RMSE': float(metrics_last['RMSE']),
                        'MAPE': float(metrics_last['MAPE']),
                    }
                },
                'best_val_epoch': best_epoch,
            }

            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)

            print("  %s sigma=%d seed=%d: MAE=%.4f, skill=%.4f, %.0fs, epoch=%d" % (
                args.model, sigma, seed, metrics_model['MAE'], skill, train_time, best_epoch))
            total_runs += 1

    print("\n=== DONE (%s): %d runs ===" % (args.model, total_runs))

if __name__ == '__main__':
    main()
