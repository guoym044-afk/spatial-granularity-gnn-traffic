#!/usr/bin/env python3
"""
Historical Average baseline for METR-LA and PeMS-Bay.
Fast CPU-only baseline — ARIMA would take too long on all sensors.
"""
import sys
import os
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/c20250521/lane_granularity_study/code')

DATA_DIR = '/c20250521/lane_granularity_study/data'
RESULTS_DIR = '/c20250521/lane_granularity_study/results'

def load_raw_speed(data_path):
    """Load raw speed data using PyTables."""
    import tables
    f = tables.open_file(data_path, 'r')
    speed = f.root.df.block0_values[:]  # [T, N]
    timestamps_unix = f.root.df.axis1[:]
    sensor_ids_raw = f.root.df.axis0[:]
    sensor_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in sensor_ids_raw]
    f.close()
    return speed, timestamps_unix, sensor_ids

def run_historical_average(train_data, test_data, horizon=3):
    """
    Historical Average: for each test timestep t, predict using
    the average of the same time-of-week from training data.
    """
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    n_sensors = train_data.shape[1]

    # 5-min intervals: 288 per day, 2016 per week
    slots_per_week = 7 * 24 * 12  # 2016

    # Compute slot index for training data
    train_slots = np.arange(n_train) % slots_per_week

    # Build HA lookup: slot -> mean speed across all occurrences in training
    ha_lookup = np.zeros((slots_per_week, n_sensors))
    for slot in range(slots_per_week):
        mask = train_slots == slot
        if mask.sum() > 0:
            ha_lookup[slot] = np.mean(train_data[mask], axis=0)
        else:
            ha_lookup[slot] = np.mean(train_data, axis=0)

    # Predict test set: use slot index starting from where training ended
    test_slots = (n_train + np.arange(n_test)) % slots_per_week
    preds = ha_lookup[test_slots]

    # Compute metrics
    mae = np.mean(np.abs(preds - test_data))
    rmse = np.sqrt(np.mean((preds - test_data) ** 2))

    # Per-step MAE (1-step, 2-step, 3-step)
    step_maes = []
    for h in range(horizon):
        step_idx = np.arange(h, n_test, horizon)
        if len(step_idx) > 0:
            step_mae = np.mean(np.abs(preds[step_idx] - test_data[step_idx]))
            step_maes.append(float(step_mae))

    return mae, rmse, step_maes

def run_persistence(test_data, horizon=3):
    """Persistence baseline: predict last observed value."""
    n_test = test_data.shape[0]
    # Simple: just repeat the last value
    # For fair comparison, use same horizon logic
    preds = np.roll(test_data, 1, axis=0)
    preds[0] = test_data[0]  # first step uses itself

    mae = np.mean(np.abs(preds - test_data))
    rmse = np.sqrt(np.mean((preds - test_data) ** 2))
    return mae, rmse

def main():
    datasets = [
        ('metr', os.path.join(DATA_DIR, 'metr-la.h5')),
        ('pems', os.path.join(DATA_DIR, 'pems-bay.h5')),
    ]

    for dataset_name, data_path in datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset_name.upper()} Baselines")
        print(f"{'='*60}")

        out_file = os.path.join(RESULTS_DIR, f'{dataset_name}_ha_baseline.json')
        if os.path.exists(out_file):
            print(f"  SKIP: {out_file} exists")
            with open(out_file) as f:
                result = json.load(f)
            print(f"  HA: MAE={result['historical_average']['MAE']:.4f}")
            print(f"  Persistence: MAE={result['persistence']['MAE']:.4f}")
            continue

        print("Loading data...")
        data, timestamps, sensor_ids = load_raw_speed(data_path)
        print(f"  Shape: {data.shape}")

        # Split: 70/15/15
        n_total = data.shape[0]
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+n_val:]

        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

        result = {
            'dataset': dataset_name,
            'num_sensors': data.shape[1],
            'train_samples': int(train_data.shape[0]),
            'test_samples': int(test_data.shape[0]),
        }

        # Historical Average
        print("Running Historical Average...")
        t0 = time.time()
        mae_ha, rmse_ha, step_maes = run_historical_average(train_data, test_data, horizon=3)
        t_ha = time.time() - t0
        result['historical_average'] = {
            'MAE': float(mae_ha), 'RMSE': float(rmse_ha),
            'step_maes': step_maes, 'time_sec': round(t_ha, 1)
        }
        print(f"  HA: MAE={mae_ha:.4f}, RMSE={rmse_ha:.4f}, {t_ha:.0f}s")
        print(f"  HA per-step: {[f'{m:.3f}' for m in step_maes]}")

        # Persistence
        print("Running Persistence...")
        mae_per, rmse_per = run_persistence(test_data, horizon=3)
        result['persistence'] = {'MAE': float(mae_per), 'RMSE': float(rmse_per)}
        print(f"  Persistence: MAE={mae_per:.4f}, RMSE={rmse_per:.4f}")

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {out_file}")

if __name__ == '__main__':
    np.random.seed(42)
    main()
