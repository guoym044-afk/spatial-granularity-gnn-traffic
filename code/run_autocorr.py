#!/usr/bin/env python3
"""
Compute lag-1 autocorrelation for METR-LA and PeMS-Bay raw speed data.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/c20250521/lane_granularity_study/code')

DATA_DIR = '/c20250521/lane_granularity_study/data'

def load_raw_speed(data_path):
    import tables
    f = tables.open_file(data_path, 'r')
    speed = f.root.df.block0_values[:]  # [T, N]
    f.close()
    return speed

def lag1_acf(data):
    """Compute lag-1 autocorrelation for each sensor (column)."""
    x = data[:-1]
    y = data[1:]
    # Per-sensor correlation
    acfs = []
    for j in range(data.shape[1]):
        col_x = x[:, j]
        col_y = y[:, j]
        mask = ~(np.isnan(col_x) | np.isnan(col_y))
        if mask.sum() < 10:
            continue
        cx = col_x[mask] - np.mean(col_x[mask])
        cy = col_y[mask] - np.mean(col_y[mask])
        denom = np.sqrt(np.sum(cx**2) * np.sum(cy**2))
        if denom < 1e-10:
            continue
        acf = np.sum(cx * cy) / denom
        acfs.append(acf)
    acfs = np.array(acfs)
    return np.mean(acfs), np.std(acfs), len(acfs)

def main():
    datasets = [
        ('METR-LA', os.path.join(DATA_DIR, 'metr-la.h5')),
        ('PeMS-Bay', os.path.join(DATA_DIR, 'pems-bay.h5')),
    ]

    for name, path in datasets:
        print(f"\n{name}:")
        data = load_raw_speed(path)
        print(f"  Shape: {data.shape}")
        mean_acf, std_acf, n_valid = lag1_acf(data)
        print(f"  Mean lag-1 ACF: {mean_acf:.4f} +/- {std_acf:.4f} ({n_valid} sensors)")

if __name__ == '__main__':
    main()
