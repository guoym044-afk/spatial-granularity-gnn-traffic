"""
Configuration for Lane Granularity Diagnostic Study (Idea 11)
车道级粒度对交通预测重要性的诊断研究
"""
import os

# === Project Paths (all on bucket: /c20250521/) ===
# Local: for code editing only; Remote: all data and results go to bucket
BUCKET_MOUNT = "/c20250521"
PROJECT_DIR = os.path.join(BUCKET_MOUNT, "lane_granularity_study")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
METR_LA_RAW = os.path.join(DATA_DIR, "metr-la.h5")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
CODE_DIR = os.path.join(PROJECT_DIR, "code")

# === Dataset Config ===
PEMS_URL = "https://pems.dot.ca.gov/PeMS_Dataset/PeMS-Bay.zip"
PEMS_RAW_FILE = os.path.join(RAW_DATA_DIR, "pems_bay.h5")

# NGSIM for lane-level ground truth
NGSIM_RAW_DIR = os.path.join(RAW_DATA_DIR, "ngsim")

# === Scene Classification ===
SCENARIOS = {
    "S1_merge": {
        "name": "高速公路合流/分流区",
        "description": "有匝道汇入或车道减少的路段",
        "expected_gain": "HIGH (15-25%)",
    },
    "S2_straight": {
        "name": "高速公路多车道直线路段",
        "description": "3-5车道、无匝道的直线路段",
        "expected_gain": "LOW (0-5%)",
    },
    "S3_intersection": {
        "name": "城市主干道交叉口",
        "description": "信号灯控制、有转向车道",
        "expected_gain": "MEDIUM (5-15%)",
    },
    "S4_urban": {
        "name": "城市普通路段",
        "description": "2-3车道、无复杂几何",
        "expected_gain": "LOW (0-5%)",
    },
}

# === Granularity Levels ===
GRANULARITIES = {
    "seg": {"name": "路段级", "description": "每条道路段一个节点"},
    "lg":  {"name": "车道组级", "description": "同方向车道合并为一个节点"},
    "il":  {"name": "单车道级", "description": "每条车道一个节点"},
}

# === Task Config ===
TASKS = {
    "bottleneck": {"name": "瓶颈位置预测", "type": "classification", "metric": "F1"},
    "travel_time": {"name": "行程时间估计", "type": "regression", "metric": "MAE"},
    "incident": {"name": "事故影响范围预测", "type": "regression", "metric": "MAE"},
}

# === Model Config ===
MODEL_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "lstm_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "early_stopping_patience": 15,
    "batch_size": 32,
    "seeds": [42, 123, 456, 789, 1024],
    "hidden_dim_sweep": [64, 128, 192],
}

# === Training Split ===
# PeMS-Bay: 6 months of data (Jan-Jun 2017)
SPLIT_CONFIG = {
    "train_months": 4,   # first 4 months
    "val_months": 1,     # 5th month
    "test_months": 1,    # 6th month
    "interval_minutes": 5,
}

# === v4 Additions ===
V4_RESULT_PREFIX = "m4_"

# Model configs for GAT and PureLSTM
MODEL_CONFIG_V4 = {
    "gat_heads": 4,
    "gat_hidden_dim": 64,
    "gat_dropout": 0.1,  # aligned with MODEL_CONFIG dropout
    "lstm_only_hidden": 64,
    "lstm_only_layers": 2,
}

# Fixed-node granularity experiment
FIXED_NODE_TARGET = 15  # target node count for fixed-node experiment
FIXED_NODE_STRATEGIES = ["spatial_merge", "zone_merge", "random_merge"]

# === Remote Server (all data on bucket) ===
REMOTE_CONFIG = {
    "ssh_alias": "volc-aris-gpu",
    "bucket_dir": "/c20250521/lane_granularity_study",
    "code_dir": "/c20250521/lane_granularity_study/code",
    "data_dir": "/c20250521/lane_granularity_study/data",
    "results_dir": "/c20250521/lane_granularity_study/results",
    "conda_cmd": 'eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate base',
    "num_gpus": 4,
}
