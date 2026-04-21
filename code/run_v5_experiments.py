"""
v5 Experiment Runner: STGCN + DCRNN on PeMS + METR-LA.

Runs:
  PeMS il:  STGCN x 4 scenes x 3 seeds = 12 runs
  PeMS il:  DCRNN x 4 scenes x 3 seeds = 12 runs
  PeMS lg:  STGCN x 1 scene x 3 seeds = 3 runs
  PeMS lg:  DCRNN x 1 scene x 3 seeds = 3 runs
  PeMS seg: STGCN x 1 scene x 3 seeds = 3 runs
  PeMS seg: DCRNN x 1 scene x 3 seeds = 3 runs
  METR-LA:  LSTGCNN x 1 scene x 3 seeds = 3 runs
  METR-LA:  STGCN x 1 scene x 3 seeds = 3 runs
  METR-LA:  DCRNN x 1 scene x 3 seeds = 3 runs
  TOTAL: 45 runs

Usage:
  python run_v5_experiments.py --experiment pems_stgcn --scene S1_merge --seed 42 --device cuda
  python run_v5_experiments.py --experiment pems_stgcn_lg --scene all --seed 42 --device cuda
  python run_v5_experiments.py --experiment pems_dcrnn_seg --scene all --seed 42 --device cuda
"""
import os
import sys
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (LSTGCNN, STGCNPredictor, DCRNNPredictor, GraphWaveNetPredictor,
                    BaselineModels, compute_metrics, train_model, evaluate_model)
from data_utils import (load_real_pems, build_real_graph, prepare_pems_training,
                         split_data, load_metr_la, build_metr_la_graph, load_metr_la_scenes)
from config import (MODEL_CONFIG, RESULTS_DIR, RAW_DATA_DIR, METR_LA_RAW, PROCESSED_DATA_DIR)


V5_PREFIX = "m5_"
PEMS_SEEDS = [42, 123, 456]
METR_SEEDS = [42, 123, 456]
PEMS_SCENES = ["S1_merge", "S2_straight", "S3_intersection", "S4_urban"]

# Experiments that use lg/seg granularity (scene=all only)
LG_SEG_EXPERIMENTS = {"pems_stgcn_lg", "pems_stgcn_seg", "pems_dcrnn_lg", "pems_dcrnn_seg", "pems_stgcn_lg2", "pems_stgcn_il2", "pems_dcrnn_lg2", "pems_dcrnn_il2"}


def load_pems_preprocessed(granularity):
    """Load preprocessed PeMS data for lg2/il2 granularities (already split)."""
    import os
    data_dir = os.path.join(PROCESSED_DATA_DIR, granularity)
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    scene_mask = list(np.load(os.path.join(data_dir, "scene_mask.npy"), allow_pickle=True))
    node_list = list(np.load(os.path.join(data_dir, "node_list.npy"), allow_pickle=True))
    adj = np.load(os.path.join(data_dir, "adjacency.npy"))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj, scene_mask, node_list, None


def load_pems_granularity(granularity):
    """Load PeMS data at a specific granularity (il/lg/seg/lg2/il2)."""
    if granularity in ("lg2", "il2"):
        return load_pems_preprocessed(granularity)
    speed_3d, positions, zones, timestamps = load_real_pems(RAW_DATA_DIR)
    adj, node_mapping, node_list = build_real_graph(positions, zones, granularity)
    X, Y, scene_mask, node_list = prepare_pems_training(
        speed_3d, node_mapping, node_list, zones, lookback=12, horizon=3
    )
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X, Y)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj, scene_mask, node_list, None


def load_metr_la_data():
    """Load METR-LA data with train/val/test split."""
    X, Y, timestamps, sensor_ids, scaler = load_metr_la(METR_LA_RAW)
    adj, node_list = build_metr_la_graph(METR_LA_RAW, adj_pkl_path="/c20250521/lane_granularity_study/data/adj_mx.pkl")

    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    scene_mask = ["all"] * X_train.shape[2]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj, scene_mask, node_list, None


def filter_scene(X, Y, scene_mask, adj, scene):
    """Filter data to a specific scene."""
    if scene == "all":
        return X, Y, adj, scene_mask

    scene_indices = [i for i, s in enumerate(scene_mask) if s == scene]
    if not scene_indices:
        return None, None, None, None

    X_s = X[:, :, scene_indices]
    Y_s = Y[:, :, scene_indices]
    adj_s = adj[np.ix_(scene_indices, scene_indices)]
    mask_s = [scene_mask[i] for i in scene_indices]
    return X_s, Y_s, adj_s, mask_s


def build_model(model_type, num_nodes, adj, output_dim, hidden_dim=64, dropout=0.1):
    """Build a model by type."""
    if model_type == "lstgcnn":
        return LSTGCNN(
            num_nodes=num_nodes, input_dim=1, hidden_dim=hidden_dim,
            output_dim=output_dim, num_gc_layers=2, num_lstm_layers=2,
            dropout=dropout, adj=adj,
        )
    elif model_type == "stgcn":
        return STGCNPredictor(
            num_nodes=num_nodes, input_dim=1, hidden_dim=hidden_dim,
            output_dim=output_dim, num_blocks=2, K=3, dropout=dropout, adj=adj,
        )
    elif model_type == "dcrnn":
        return DCRNNPredictor(
            num_nodes=num_nodes, input_dim=1, hidden_dim=hidden_dim,
            output_dim=output_dim, num_rnn_layers=2, K=2, dropout=dropout, adj=adj,
        )
    elif model_type == "graphwavenet":
        return GraphWaveNetPredictor(
            num_nodes=num_nodes, input_dim=1, hidden_dim=hidden_dim,
            output_dim=output_dim, dropout=dropout, adj=adj,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_single(experiment, scene, seed, device="cuda"):
    """
    Run a single v5 experiment.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Parse experiment
    is_lg_seg = experiment in LG_SEG_EXPERIMENTS

    if is_lg_seg:
        # e.g. pems_stgcn_lg -> data_source=pems, model_type=stgcn, granularity=lg
        parts = experiment.split("_")
        data_source = parts[0]  # pems
        model_type = parts[1]   # stgcn or dcrnn
        granularity = parts[2]  # lg or seg
    elif experiment.startswith("pems_"):
        data_source = "pems"
        model_type = experiment.replace("pems_", "")
        granularity = "il"
    elif experiment.startswith("metr_"):
        data_source = "metr"
        model_type = experiment.replace("metr_", "")
        granularity = "il"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print(f"\n{'='*60}")
    print(f"v5: {experiment} / {scene} / seed={seed} / gran={granularity}")
    print(f"{'='*60}")

    # Load data
    if data_source == "pems":
        X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_full, scene_mask, node_list, _ = load_pems_granularity(granularity)
        metr_scaler = None
        X_train_s, Y_train_s, adj_s, scene_mask_s = filter_scene(X_train, Y_train, scene_mask, adj_full, scene)
        X_val_s, Y_val_s, _, _ = filter_scene(X_val, Y_val, scene_mask, adj_full, scene)
        X_test_s, Y_test_s, _, _ = filter_scene(X_test, Y_test, scene_mask, adj_full, scene)
    else:  # metr
        X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_full, scene_mask, node_list, scaler = load_metr_la_data()
        metr_scaler = scaler
        X_train_s, Y_train_s = X_train, Y_train
        X_val_s, Y_val_s = X_val, Y_val
        X_test_s, Y_test_s = X_test, Y_test
        adj_s = adj_full
        scene_mask_s = scene_mask

    if X_train_s is None:
        print(f"  [SKIP] No nodes for scene {scene}")
        return None

    num_nodes = X_train_s.shape[2]
    horizon = Y_train_s.shape[1]

    print(f"  Nodes: {num_nodes}, Train: {len(X_train_s)}, Val: {len(X_val_s)}, Test: {len(X_test_s)}")

    # Baselines (de-normalize for correct MAE in mph)
    pred_last = BaselineModels.last_value_predictor(X_test_s, horizon=horizon)
    if metr_scaler is not None:
        mean = np.array(metr_scaler['mean']).reshape(1, 1, -1)
        std = np.array(metr_scaler['std']).reshape(1, 1, -1)
        pred_last_dn = pred_last * std + mean
        Y_test_dn = Y_test_s * std + mean
        metrics_last = compute_metrics(pred_last_dn, Y_test_dn)
    else:
        metrics_last = compute_metrics(pred_last, Y_test_s)
    print(f"  Persistence: MAE={metrics_last['MAE']:.4f}")

    # Build and train
    model = build_model(model_type, num_nodes, adj_s, horizon)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_type} params: {total_params:,}")

    t0 = time.time()
    model, train_losses, val_losses = train_model(
        model, X_train_s, Y_train_s, X_val_s, Y_val_s, adj_s,
        lr=MODEL_CONFIG["learning_rate"],
        max_epochs=MODEL_CONFIG["max_epochs"],
        patience=MODEL_CONFIG["early_stopping_patience"],
        device=device,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s, epochs: {len(train_losses)}")

    # Evaluate
    metrics_model, _ = evaluate_model(
        model, X_test_s, Y_test_s, adj_s, device=device, scene_mask=None, scaler=metr_scaler
    )
    print(f"  {model_type}: MAE={metrics_model['MAE']:.4f}, skill={metrics_model['skill_score']:.4f}")

    # Serialize
    per_sample = metrics_model.pop("per_sample_mae", None)
    results = {
        "granularity": granularity,
        "scene": scene,
        "seed": seed,
        "model_type": model_type,
        "data_source": data_source,
        "dropout": 0.1,
        "num_nodes": num_nodes,
        "total_params": total_params,
        "train_time_sec": round(train_time, 1),
        "lstgcnn": {k: float(v) if v is not None else None for k, v in metrics_model.items()},
        "baselines": {
            "last_value": {k: float(v) if v is not None and not isinstance(v, np.ndarray) else None
                           for k, v in metrics_last.items()},
        },
        "best_val_epoch": len(train_losses) - MODEL_CONFIG["early_stopping_patience"]
            if len(val_losses) > MODEL_CONFIG["early_stopping_patience"] else len(val_losses),
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"{V5_PREFIX}{experiment}_{scene}_seed{seed}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="v5 Experiment Runner")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["pems_stgcn", "pems_dcrnn", "pems_lstgcnn", "pems_graphwavenet",
                                 "pems_stgcn_lg", "pems_stgcn_seg",
                                 "pems_dcrnn_lg", "pems_dcrnn_seg",
                                 "pems_lstgcnn_lg", "pems_lstgcnn_seg",
                                 "pems_graphwavenet_lg", "pems_graphwavenet_seg",
                                 "pems_stgcn_lg2", "pems_stgcn_il2",
                                 "pems_dcrnn_lg2", "pems_dcrnn_il2",
                                 "metr_lstgcnn", "metr_stgcn", "metr_dcrnn", "metr_graphwavenet"])
    parser.add_argument("--scene", type=str, default="S1_merge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_single(args.experiment, args.scene, args.seed, args.device)


if __name__ == "__main__":
    main()
