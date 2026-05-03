#!/usr/bin/env python3
"""PeMS-Bay smoothing control experiment.

Applies Gaussian temporal smoothing at full resolution (N=326).
Usage:
    python run_pems_smoothing_control.py --model dcrnn --sigma 1 --seed 42 --device cuda:0
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))
from model import DCRNNPredictor, STGCNPredictor, train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['dcrnn', 'stgcn'])
    parser.add_argument('--sigma', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load pre-split PeMS-Bay data at full resolution
    X_train = np.load('/c20250521/lane_granularity_study/data/processed/il/X_train.npy')
    Y_train = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_train.npy')
    X_val = np.load('/c20250521/lane_granularity_study/data/processed/il/X_val.npy')
    Y_val = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_val.npy')
    X_test = np.load('/c20250521/lane_granularity_study/data/processed/il/X_test.npy')
    Y_test = np.load('/c20250521/lane_granularity_study/data/processed/il/Y_test.npy')
    adj = np.load('/c20250521/lane_granularity_study/data/processed/il/adjacency.npy')

    # Apply Gaussian smoothing along time axis
    if args.sigma > 0:
        X_train = gaussian_filter1d(X_train, sigma=args.sigma, axis=1)
        Y_train = gaussian_filter1d(Y_train, sigma=args.sigma, axis=1)
        X_val = gaussian_filter1d(X_val, sigma=args.sigma, axis=1)
        Y_val = gaussian_filter1d(Y_val, sigma=args.sigma, axis=1)
        X_test = gaussian_filter1d(X_test, sigma=args.sigma, axis=1)
        Y_test = gaussian_filter1d(Y_test, sigma=args.sigma, axis=1)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    n_nodes = X_train.shape[2]

    # Build model
    if args.model == 'dcrnn':
        model = DCRNNPredictor(num_nodes=n_nodes, input_dim=1,
                               hidden_dim=64, output_dim=3, adj=adj).to(device)
    else:
        model = STGCNPredictor(num_nodes=n_nodes, input_dim=1,
                               hidden_dim=64, output_dim=3, adj=adj).to(device)

    # Train
    train_model(model, X_train, Y_train, X_val, Y_val, adj,
                lr=0.001, max_epochs=100, patience=15, device=device)

    # Evaluate
    metrics, _ = evaluate_model(model, X_test, Y_test, adj, device=device)

    # Save
    result = {
        'dataset': 'pems', 'model': args.model, 'sigma': args.sigma,
        'n_nodes': 326, 'seed': args.seed,
        'mae': float(metrics['MAE']), 'rmse': float(metrics['RMSE']),
        'pers_mae': float(metrics.get('pers_mae', 0)),
        'skill': float(metrics.get('skill_score', 0) or 0),
    }
    out_dir = '/c20250521/lane_granularity_study/results/pems_smoothing'
    os.makedirs(out_dir, exist_ok=True)
    out_file = f'{out_dir}/pems_{args.model}_sigma{args.sigma}_seed{args.seed}.json'
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'MAE={result["mae"]:.4f}, Skill={result["skill"]:.4f} -> {out_file}')


if __name__ == '__main__':
    main()
