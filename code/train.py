"""
Main training script for Idea 11: Lane Granularity Diagnostic Study (v2).

v2 changes:
- hidden_dim parameter support (for capacity study M3)
- skill_score and NMAE metrics
- per_sample_mae saved for statistical tests
- improvement_over_mean_pct retained for backward compat
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import LSTGCNN, BaselineModels, train_model, evaluate_model, compute_metrics
from config import MODEL_CONFIG, RESULTS_DIR, PROCESSED_DATA_DIR, LOG_DIR


def run_single_experiment(granularity, scene, seed, device="cuda", hidden_dim=None):
    """
    Run a single experiment: one granularity x one scene x one seed.

    Args:
        hidden_dim: override MODEL_CONFIG["hidden_dim"] (for M3 capacity study)

    Returns:
        results: dict with metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if hidden_dim is None:
        hidden_dim = MODEL_CONFIG["hidden_dim"]

    # Load data
    data_dir = os.path.join(PROCESSED_DATA_DIR, granularity)
    adj_path = os.path.join(data_dir, "adjacency.npy")

    if os.path.exists(os.path.join(data_dir, "X_train.npy")):
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
        scene_mask = np.load(os.path.join(data_dir, "scene_mask.npy"), allow_pickle=True)
        node_list = np.load(os.path.join(data_dir, "node_list.npy"), allow_pickle=True)
    elif os.path.exists(os.path.join(data_dir, "data.npz")):
        data = np.load(os.path.join(data_dir, "data.npz"), allow_pickle=True)
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_val = data["X_val"]
        Y_val = data["Y_val"]
        X_test = data["X_test"]
        Y_test = data["Y_test"]
        scene_mask = data["scene_mask"]
        node_list = data["node_list"]
    else:
        print(f"[ERROR] No data found in {data_dir}")
        return None

    if not os.path.exists(adj_path):
        print(f"[ERROR] Adjacency file not found: {adj_path}")
        return None
    adj = np.load(adj_path)

    # Filter to specific scene if requested
    if scene != "all":
        scene_indices = [i for i, s in enumerate(scene_mask) if s == scene]
        if not scene_indices:
            print(f"[WARN] No nodes for scene {scene} in granularity {granularity}")
            return None
        X_train_s = X_train[:, :, scene_indices]
        Y_train_s = Y_train[:, :, scene_indices]
        X_val_s = X_val[:, :, scene_indices]
        Y_val_s = Y_val[:, :, scene_indices]
        X_test_s = X_test[:, :, scene_indices]
        Y_test_s = Y_test[:, :, scene_indices]
        adj_s = adj[np.ix_(scene_indices, scene_indices)]
        scene_mask_s = [scene_mask[i] for i in scene_indices]
    else:
        X_train_s, Y_train_s = X_train, Y_train
        X_val_s, Y_val_s = X_val, Y_val
        X_test_s, Y_test_s = X_test, Y_test
        adj_s = adj
        scene_mask_s = list(scene_mask)

    num_nodes = X_train_s.shape[2]
    lookback = X_train_s.shape[1]
    horizon = Y_train_s.shape[1]

    print(f"\n[EXP] Granularity={granularity}, Scene={scene}, Seed={seed}, hidden_dim={hidden_dim}")
    print(f"  Nodes: {num_nodes}, Lookback: {lookback}, Horizon: {horizon}")
    print(f"  Train: {X_train_s.shape[0]}, Val: {X_val_s.shape[0]}, Test: {X_test_s.shape[0]}")

    # === Baseline 1: Mean predictor ===
    pred_mean = BaselineModels.mean_predictor(X_test_s, horizon=horizon)
    metrics_mean = compute_metrics(pred_mean, Y_test_s)
    print(f"  Mean predictor: MAE={metrics_mean['MAE']:.4f}, MAPE={metrics_mean['MAPE']:.2f}%")

    # === Baseline 2: Last value predictor ===
    pred_last = BaselineModels.last_value_predictor(X_test_s, horizon=horizon)
    metrics_last = compute_metrics(pred_last, Y_test_s)
    print(f"  Last value:     MAE={metrics_last['MAE']:.4f}, MAPE={metrics_last['MAPE']:.2f}%")

    # === Baseline 3: Linear trend ===
    pred_linear = BaselineModels.linear_trend_predictor(X_test_s, horizon=horizon)
    metrics_linear = compute_metrics(pred_linear, Y_test_s)
    print(f"  Linear trend:   MAE={metrics_linear['MAE']:.4f}, MAPE={metrics_linear['MAPE']:.2f}%")

    # === LSTGCNN ===
    model = LSTGCNN(
        num_nodes=num_nodes,
        input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=horizon,
        num_gc_layers=MODEL_CONFIG["num_layers"],
        num_lstm_layers=MODEL_CONFIG["lstm_layers"],
        dropout=MODEL_CONFIG["dropout"],
        adj=adj_s,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LSTGCNN params: {total_params:,} (hidden_dim={hidden_dim})")

    # Train
    t0 = time.time()
    model, train_losses, val_losses = train_model(
        model, X_train_s, Y_train_s, X_val_s, Y_val_s, adj_s,
        lr=MODEL_CONFIG["learning_rate"],
        max_epochs=MODEL_CONFIG["max_epochs"],
        patience=MODEL_CONFIG["early_stopping_patience"],
        device=device,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate (v2: returns skill_score via baseline_pred)
    metrics_lstm, scene_metrics = evaluate_model(
        model, X_test_s, Y_test_s, adj_s, device=device, scene_mask=scene_mask_s
    )
    print(f"  LSTGCNN: MAE={metrics_lstm['MAE']:.4f}, MAPE={metrics_lstm['MAPE']:.2f}%, "
          f"NMAE={metrics_lstm['NMAE']:.4f}, skill={metrics_lstm['skill_score']:.4f}")

    # Legacy metric: improvement over mean
    improvement = (metrics_mean["MAE"] - metrics_lstm["MAE"]) / metrics_mean["MAE"] * 100
    print(f"  Improvement over mean: {improvement:.1f}%")

    # Serialize per_sample_mae as list (for JSON)
    per_sample = metrics_lstm.pop("per_sample_mae", None)
    per_sample_list = per_sample.tolist() if per_sample is not None else None

    # Also extract per_scene per_sample_mae
    scene_metrics_serializable = {}
    for sc, sm in scene_metrics.items():
        sm_copy = dict(sm)
        ps = sm_copy.pop("per_sample_mae", None)
        scene_metrics_serializable[sc] = {k: float(v) if v is not None else None
                                           for k, v in sm_copy.items()}

    results = {
        "granularity": granularity,
        "scene": scene,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "num_nodes": num_nodes,
        "total_params": total_params,
        "train_time_sec": round(train_time, 1),
        "baselines": {
            "mean": {k: (float(v) if v is not None and not isinstance(v, np.ndarray) else None)
                     for k, v in metrics_mean.items()},
            "last_value": {k: (float(v) if v is not None and not isinstance(v, np.ndarray) else None)
                           for k, v in metrics_last.items()},
            "linear_trend": {k: (float(v) if v is not None and not isinstance(v, np.ndarray) else None)
                             for k, v in metrics_linear.items()},
        },
        "lstgcnn": {k: float(v) if v is not None else None
                     for k, v in metrics_lstm.items()},
        "improvement_over_mean_pct": round(improvement, 2),
        "per_sample_mae": per_sample_list,
        "best_val_epoch": len(train_losses) - MODEL_CONFIG["early_stopping_patience"]
            if len(val_losses) > MODEL_CONFIG["early_stopping_patience"] else len(val_losses),
    }

    if scene_metrics_serializable:
        results["per_scene"] = scene_metrics_serializable

    return results


def main():
    parser = argparse.ArgumentParser(description="Idea 11: Lane Granularity Experiment (v2)")
    parser.add_argument("--granularity", type=str, required=True, choices=["seg", "lg", "il"],
                        help="Granularity level: seg (segment), lg (lane-group), il (individual-lane)")
    parser.add_argument("--scene", type=str, default="all",
                        help="Scene type: S1_merge, S2_straight, S3_intersection, S4_urban, all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Override hidden_dim (for M3 capacity study)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Idea 11 Experiment (v2): {args.granularity} / {args.scene} / seed={args.seed}")
    print("=" * 60)

    results = run_single_experiment(
        args.granularity, args.scene, args.seed, args.device,
        hidden_dim=args.hidden_dim
    )

    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Include hidden_dim in filename if non-default
        hd_suffix = f"_hd{args.hidden_dim}" if args.hidden_dim else ""
        out_file = os.path.join(
            RESULTS_DIR,
            f"m11_{args.granularity}_{args.scene}_seed{args.seed}{hd_suffix}.json"
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Results: {out_file}")
    else:
        print("\n[FAILED] Experiment produced no results")


if __name__ == "__main__":
    main()
