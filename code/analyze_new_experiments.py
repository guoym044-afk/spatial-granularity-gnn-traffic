#!/usr/bin/env python3
"""
Analyze sensor-subset and feature-vs-graph ablation results.
Produces summary tables for the paper.

Usage:
  python analyze_new_experiments.py [--server]
"""
import json
import glob
import os
import sys
import numpy as np
from collections import defaultdict

RESULTS_DIR = '/c20250521/lane_granularity_study/results'
SEEDS = [42, 123, 456]
COARSENED_SEEDS = [42, 123, 456, 789, 1024, 1337, 1999, 2024, 3141, 6283]
MODELS = ['dcrnn', 'graphwavenet']
SUBSET_NS = [5, 10, 20, 50, 100, 150, 207]
ABLATION_NS = [50, 100, 150]
CONDITIONS = ['coarsened', 'feature_only', 'graph_only', 'sensor_subset']


def load_results(pattern, model):
    """Load all result files matching pattern, return list of dicts."""
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    results = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
            if model in d:
                results.append(d)
    return results


def get_mae_skill(results, model):
    """Extract MAE and skill_score from results."""
    maes = []
    skills = []
    for r in results:
        m = r[model]
        if m.get('MAE') is not None:
            maes.append(m['MAE'])
        if m.get('skill_score') is not None:
            skills.append(m['skill_score'])
    return maes, skills


def mean_std(vals):
    if not vals:
        return float('nan'), float('nan')
    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0


def fit_power_law(ns, maes):
    """Fit MAE = a * N^b via log-linear regression. Returns (a, b, r_squared)."""
    log_ns = np.log(ns)
    log_maes = np.log(maes)
    coeffs = np.polyfit(log_ns, log_maes, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    # R-squared
    log_pred = np.polyval(coeffs, log_ns)
    ss_res = np.sum((log_maes - log_pred) ** 2)
    ss_tot = np.sum((log_maes - np.mean(log_maes)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def analyze_subset():
    """Analyze sensor-subset results."""
    print("=" * 70)
    print("EXPERIMENT 1: Sensor Subset Baseline")
    print("=" * 70)

    for model in MODELS:
        print(f"\n--- {model.upper()} ---")
        print(f"{'N':>5} {'MAE':>10} {'±':>6} {'Skill':>10} {'±':>6} {'n_seeds':>7}")

        ns_for_fit = []
        maes_for_fit = []

        for n in SUBSET_NS:
            pattern = f"m5_metr_{model}_subset{n}_seed*.json"
            results = load_results(pattern, model)
            maes, skills = get_mae_skill(results, model)
            mae_mean, mae_std = mean_std(maes)
            skill_mean, skill_std = mean_std(skills)
            print(f"{n:>5} {mae_mean:>10.4f} {mae_std:>6.4f} {skill_mean:>10.4f} {skill_std:>6.4f} {len(maes):>7}")
            if len(maes) > 0:
                ns_for_fit.append(n)
                maes_for_fit.append(mae_mean)

        # Power-law fit
        if len(ns_for_fit) >= 3:
            a, b, r2 = fit_power_law(np.array(ns_for_fit), np.array(maes_for_fit))
            print(f"\n  Power-law fit: MAE = {a:.3f} * N^{b:.3f}  (R² = {r2:.4f})")

        # Compare with coarsened at same N
        print(f"\n  Comparison with coarsened (3 seeds):")
        print(f"  {'N':>5} {'Subset':>10} {'Coarsened':>10} {'ΔMAE':>8} {'Δ%':>6}")
        for n in SUBSET_NS:
            # Subset
            pattern_sub = f"m5_metr_{model}_subset{n}_seed*.json"
            results_sub = load_results(pattern_sub, model)
            maes_sub, _ = get_mae_skill(results_sub, model)
            mae_sub = np.mean(maes_sub) if maes_sub else float('nan')

            # Coarsened
            pattern_coarse = f"m5_metr_{model}_gran{n}_seed*.json"
            results_coarse = load_results(pattern_coarse, model)
            maes_coarse, _ = get_mae_skill(results_coarse, model)
            mae_coarse = np.mean(maes_coarse) if maes_coarse else float('nan')

            if not np.isnan(mae_sub) and not np.isnan(mae_coarse):
                delta = mae_sub - mae_coarse
                pct = delta / mae_coarse * 100
                print(f"  {n:>5} {mae_sub:>10.4f} {mae_coarse:>10.4f} {delta:>+8.4f} {pct:>+5.1f}%")


def analyze_ablation():
    """Analyze feature-vs-graph ablation results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Feature-vs-Graph Ablation")
    print("=" * 70)

    for model in MODELS:
        print(f"\n--- {model.upper()} ---")
        print(f"{'N':>5} {'Condition':<15} {'MAE':>10} {'±':>6} {'Skill':>10} {'±':>6} {'n':>3}")

        for n in ABLATION_NS:
            for cond in CONDITIONS:
                if cond == 'coarsened':
                    pattern = f"m5_metr_{model}_gran{n}_seed*.json"
                elif cond == 'sensor_subset':
                    pattern = f"m5_metr_{model}_subset{n}_seed*.json"
                else:
                    pattern = f"m5_metr_{model}_ablation_{cond}_n{n}_seed*.json"

                results = load_results(pattern, model)
                maes, skills = get_mae_skill(results, model)
                mae_mean, mae_std = mean_std(maes)
                skill_mean, skill_std = mean_std(skills)
                print(f"{n:>5} {cond:<15} {mae_mean:>10.4f} {mae_std:>6.4f} {skill_mean:>10.4f} {skill_std:>6.4f} {len(maes):>3}")
            print()


def analyze_ablation_paper_table():
    """Generate the ablation table for the paper (LaTeX format)."""
    print("\n" + "=" * 70)
    print("ABLATION TABLE (LaTeX)")
    print("=" * 70)

    for model in MODELS:
        print(f"\n% {model.upper()}")
        print(f"\\begin{{tabular}}{{lccc}}")
        print(f"\\toprule")
        print(f"Condition & N=50 & N=100 & N=150 \\\\")
        print(f"\\midrule")

        for cond in CONDITIONS:
            row = cond.replace('_', ' ').title()
            if cond == 'coarsened':
                row = 'Coarsened'
            elif cond == 'sensor_subset':
                row = 'Sensor Subset'

            for n in ABLATION_NS:
                if cond == 'coarsened':
                    pattern = f"m5_metr_{model}_gran{n}_seed*.json"
                elif cond == 'sensor_subset':
                    pattern = f"m5_metr_{model}_subset{n}_seed*.json"
                else:
                    pattern = f"m5_metr_{model}_ablation_{cond}_n{n}_seed*.json"

                results = load_results(pattern, model)
                maes, _ = get_mae_skill(results, model)
                mae_mean, mae_std = mean_std(maes)

                if n == ABLATION_NS[0]:
                    row += f" & {mae_mean:.2f}{{\\tiny $\\pm${mae_std:.2f}}}"
                else:
                    row += f" & {mae_mean:.2f}{{\\tiny $\\pm${mae_std:.2f}}}"

            row += " \\\\"
            print(row)

        print(f"\\bottomrule")
        print(f"\\end{{tabular}}")


def analyze_power_law_comparison():
    """Compare power-law fits between subset and coarsened."""
    print("\n" + "=" * 70)
    print("POWER-LAW FIT COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<15} {'Condition':<15} {'a':>8} {'b':>8} {'R²':>8}")
    print("-" * 55)

    for model in MODELS:
        for cond_name, n_values in [('Coarsened', SUBSET_NS), ('Subset', SUBSET_NS)]:
            ns = []
            maes = []
            for n in n_values:
                if cond_name == 'Coarsened':
                    pattern = f"m5_metr_{model}_gran{n}_seed*.json"
                else:
                    pattern = f"m5_metr_{model}_subset{n}_seed*.json"
                results = load_results(pattern, model)
                mae_list, _ = get_mae_skill(results, model)
                if mae_list:
                    ns.append(n)
                    maes.append(np.mean(mae_list))

            if len(ns) >= 3:
                a, b, r2 = fit_power_law(np.array(ns), np.array(maes))
                print(f"{model:<15} {cond_name:<15} {a:>8.3f} {b:>8.3f} {r2:>8.4f}")


def generate_summary():
    """Generate overall summary for paper."""
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    # Key comparison at N=50
    print("\nKey comparison at N=50 (mean across seeds):")
    print(f"{'Model':<15} {'Coarsened':>10} {'Subset':>10} {'Feature':>10} {'Graph':>10}")
    print("-" * 55)

    for model in MODELS:
        vals = {}
        for cond in CONDITIONS:
            if cond == 'coarsened':
                pattern = f"m5_metr_{model}_gran50_seed*.json"
            elif cond == 'sensor_subset':
                pattern = f"m5_metr_{model}_subset50_seed*.json"
            else:
                pattern = f"m5_metr_{model}_ablation_{cond}_n50_seed*.json"
            results = load_results(pattern, model)
            maes, _ = get_mae_skill(results, model)
            vals[cond] = np.mean(maes) if maes else float('nan')

        print(f"{model:<15} {vals['coarsened']:>10.3f} {vals['sensor_subset']:>10.3f} {vals['feature_only']:>10.3f} {vals['graph_only']:>10.3f}")

    # Scaling exponent comparison
    print("\nScaling exponents (b in MAE = a * N^b):")
    for model in MODELS:
        for cond_name in ['Coarsened', 'Subset']:
            ns, maes = [], []
            for n in SUBSET_NS:
                if cond_name == 'Coarsened':
                    pattern = f"m5_metr_{model}_gran{n}_seed*.json"
                else:
                    pattern = f"m5_metr_{model}_subset{n}_seed*.json"
                results = load_results(pattern, model)
                mae_list, _ = get_mae_skill(results, model)
                if mae_list:
                    ns.append(n)
                    maes.append(np.mean(mae_list))
            if len(ns) >= 3:
                _, b, _ = fit_power_law(np.array(ns), np.array(maes))
                print(f"  {model} {cond_name}: b = {b:.3f}")


def main():
    analyze_subset()
    analyze_ablation()
    analyze_ablation_paper_table()
    analyze_power_law_comparison()
    generate_summary()


if __name__ == '__main__':
    main()
