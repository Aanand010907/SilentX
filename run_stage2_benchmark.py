"""Stage 2 Comprehensive Monitoring Benchmark Runner for SilentX.

Executes:
1. Multi-seed synthetic benchmark (seeds: 42, 123, 999) across all 6 shift regimes.
2. Real-world OpenML Electricity (ID 151) chronological deployment stream benchmark.
3. Evaluates all baseline monitors:
   - Drift: KS_max, PSI_max, JSD_max
   - Confidence / Transport Estimators: AC, DoC, ATC, CBPE, COTT
   - TrustGuard: S_attr, S_dis, S_conf, S_grad, S_d3m, Risk_TG, alarm
4. Sequestered Oracle scoring: Accuracy MAE, AUROC, AUPRC, FAR_benign, Spearman rho_s.
5. Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR adjustment.
6. Generates publication-grade diagnostic plots in results/stage2_monitor_comparison.png.
7. Saves results/stage2_synthetic_summary.csv, results/stage2_electricity_windows.csv,
   results/stage2_electricity_aggregate.csv, and results/stage2_statistical_tests.csv.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.dataset_registry import DatasetRegistry
from src.data.real_datasets import OpenMLElectricityLoader
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.baselines import (
    KSMaxDriftDetector,
    PopulationStabilityIndexDetector,
    JensenShannonDivergenceDetector,
    AverageConfidenceEstimator,
    DifferenceOfConfidencesEstimator,
    AverageThresholdedConfidenceEstimator,
    CBPEPerformanceEstimator,
    ConfidenceOptimalTransportEstimator,
)
from src.evaluation.oracle import EvaluationOracle


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) procedure."""
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    adjusted_p = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (n / rank) * sorted_p[i]
        cum_min = min(cum_min, adj)
        adjusted_p[i] = min(1.0, cum_min)

    # Restore original order
    rev_idx = np.empty(n, dtype=int)
    rev_idx[sorted_idx] = np.arange(n)
    final_adj_p = adjusted_p[rev_idx]
    reject = (final_adj_p <= alpha)
    return reject, final_adj_p


def run_stage2_benchmarks():
    print("=" * 80)
    print("SILENTX: STAGE 2 COMPREHENSIVE MONITORING BENCHMARK")
    print("=" * 80)
    start_total = time.time()
    os.makedirs("results", exist_ok=True)

    seeds = [42, 123, 999]
    oracle = EvaluationOracle(degradation_threshold=0.05, benign_margin=0.02)

    # =====================================================================
    # PART 1: MULTI-SEED SYNTHETIC BENCHMARK
    # =====================================================================
    print("\n--- PART 1: Multi-Seed Synthetic Benchmark (Seeds 42, 123, 999) ---")
    synth_records = []

    for seed in seeds:
        print(f"[*] Running synthetic benchmark with seed {seed}...")
        bundle = DatasetRegistry.load_dataset(
            name="synthetic_stream",
            source_train_ratio=0.75,
            window_size=300,
            n_source_samples=3000,
            seed=seed,
        )

        pipeline = train_and_freeze_pipeline(
            X_train=bundle.X_source_train,
            y_train=bundle.y_source_train,
            X_cal=bundle.X_source_cal,
            y_cal=bundle.y_source_cal,
            feature_names=bundle.feature_names,
            seed=seed,
        )

        # Baseline monitors
        ks_det = KSMaxDriftDetector(bundle.X_source_cal)
        psi_det = PopulationStabilityIndexDetector(bundle.X_source_cal, n_bins=10)
        jsd_det = JensenShannonDivergenceDetector(bundle.X_source_cal, n_bins=10)
        atc_est = AverageThresholdedConfidenceEstimator(pipeline.atc_threshold)
        doc_est = DifferenceOfConfidencesEstimator(pipeline.source_cal_acc, pipeline.source_cal_mean_conf)
        cbpe_est = CBPEPerformanceEstimator()

        cal_probs = pipeline.predict_primary_proba(bundle.X_source_cal)
        cott_est = ConfidenceOptimalTransportEstimator.fit(
            cal_probs=cal_probs,
            cal_labels=bundle.y_source_cal.to_numpy(),
            cal_acc=pipeline.source_cal_acc,
            max_cal_samples=80,
            seed=seed,
        )

        # TrustGuard monitor
        tg_mon = TrustGuardCompositeMonitor(pipeline, bundle.X_source_cal, seed=seed)

        for w in bundle.deployment_windows:
            X_unlab = w.unlabeled_features
            probs_prim = pipeline.predict_primary_proba(X_unlab)

            # Evaluate drift
            ks_val = ks_det.evaluate_window(X_unlab)
            psi_val = psi_det.evaluate_window(X_unlab)
            jsd_val = jsd_det.evaluate_window(X_unlab)

            # Evaluate confidence & transport estimators
            est_ac = AverageConfidenceEstimator.estimate_accuracy(probs_prim)
            est_doc = doc_est.estimate_accuracy(probs_prim)
            est_atc = atc_est.estimate_accuracy(probs_prim)
            est_cbpe = cbpe_est.estimate_accuracy(probs_prim)
            est_cott = cott_est.estimate_accuracy(probs_prim, max_target_samples=80)

            # Evaluate TrustGuard
            tg_out = tg_mon.evaluate_window(X_unlab)

            # Oracle scoring
            eval_res = oracle.score_window(
                window_id=w.window_id,
                regime=w.regime,
                severity=w.severity,
                y_sequestered=w.y,
                predicted_probs=probs_prim,
                source_cal_acc=pipeline.source_cal_acc,
                monitor_estimated_acc=est_cbpe,
                monitor_risk_score=tg_out["risk_tg"],
            )

            synth_records.append({
                "seed": seed,
                "window_id": w.window_id,
                "regime": w.regime,
                "severity": w.severity,
                "true_accuracy": eval_res.true_accuracy,
                "true_degradation": eval_res.true_degradation,
                "is_degraded": eval_res.is_degraded,
                "est_ac": est_ac,
                "est_doc": est_doc,
                "est_atc": est_atc,
                "est_cbpe": est_cbpe,
                "est_cott": est_cott,
                "ks_max": ks_val,
                "psi_max": psi_val,
                "jsd_max": jsd_val,
                "tg_risk": tg_out["risk_tg"],
                "tg_alarm": tg_out["is_alarm"],
                "s_attr": tg_out["s_attr"],
                "s_dis": tg_out["s_dis"],
                "s_conf": tg_out["s_conf"],
                "s_grad": tg_out["s_grad"],
                "s_d3m": tg_out["s_d3m"],
            })

    synth_df = pd.DataFrame(synth_records)
    synth_path = os.path.join("results", "stage2_synthetic_summary.csv")
    synth_df.to_csv(synth_path, index=False)
    print(f"[+] Saved multi-seed synthetic summary to: {synth_path}")

    # =====================================================================
    # PART 2: REAL OPENML ELECTRICITY (ID 151) TEMPORAL BENCHMARK
    # =====================================================================
    print("\n--- PART 2: Real OpenML Electricity (ID 151) Temporal Stream ---")
    elec_data = OpenMLElectricityLoader.load(
        source_ratio=0.25,
        source_train_ratio=0.75,
        manifest_path="results/electricity_manifest.json",
    )
    print(f"[*] Loaded OpenML Electricity: {elec_data['n_total']} rows total, real download: {elec_data['is_real']}")

    # Train base model on early training split (first 8496 instances)
    elec_pipeline = train_and_freeze_pipeline(
        X_train=elec_data["X_train"],
        y_train=elec_data["y_train"],
        X_cal=elec_data["X_cal"],
        y_cal=elec_data["y_cal"],
        feature_names=elec_data["feature_names"],
        seed=42,
    )
    print(f"    Electricity Source Cal Acc: {elec_pipeline.source_cal_acc * 100:.2f}%")

    # Set up monitors on source calibration split
    elec_ks = KSMaxDriftDetector(elec_data["X_cal"])
    elec_psi = PopulationStabilityIndexDetector(elec_data["X_cal"], n_bins=10)
    elec_jsd = JensenShannonDivergenceDetector(elec_data["X_cal"], n_bins=10)
    elec_atc = AverageThresholdedConfidenceEstimator(elec_pipeline.atc_threshold)
    elec_doc = DifferenceOfConfidencesEstimator(elec_pipeline.source_cal_acc, elec_pipeline.source_cal_mean_conf)
    elec_cbpe = CBPEPerformanceEstimator()

    cal_probs_elec = elec_pipeline.predict_primary_proba(elec_data["X_cal"])
    elec_cott = ConfidenceOptimalTransportEstimator.fit(
        cal_probs=cal_probs_elec,
        cal_labels=elec_data["y_cal"].to_numpy(),
        cal_acc=elec_pipeline.source_cal_acc,
        max_cal_samples=100,
        seed=42,
    )

    elec_tg = TrustGuardCompositeMonitor(elec_pipeline, elec_data["X_cal"], seed=42)

    # Partition deployment stream into chronological windows of m = 500
    window_size = 500
    X_deploy = elec_data["X_deploy"]
    y_deploy = elec_data["y_deploy"]
    n_windows = min(25, len(X_deploy) // window_size)  # 25 chronological deployment windows

    elec_records = []
    elec_oracle_results = []

    print(f"[*] Streaming {n_windows} sequential deployment windows (m={window_size} each)...")
    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = start + window_size
        X_w = X_deploy.iloc[start:end].reset_index(drop=True)
        y_w = y_deploy.iloc[start:end].reset_index(drop=True)

        probs_w = elec_pipeline.predict_primary_proba(X_w)

        # Baseline estimates
        est_ac = AverageConfidenceEstimator.estimate_accuracy(probs_w)
        est_doc = elec_doc.estimate_accuracy(probs_w)
        est_atc = elec_atc.estimate_accuracy(probs_w)
        est_cbpe = elec_cbpe.estimate_accuracy(probs_w)
        est_cott = elec_cott.estimate_accuracy(probs_w, max_target_samples=100)

        ks_val = elec_ks.evaluate_window(X_w)
        psi_val = elec_psi.evaluate_window(X_w)
        jsd_val = elec_jsd.evaluate_window(X_w)

        # TrustGuard evaluation
        tg_out = elec_tg.evaluate_window(X_w)

        # Oracle evaluation
        eval_res = oracle.score_window(
            window_id=w_idx,
            regime="temporal_electricity",
            severity=float(w_idx) / n_windows,
            y_sequestered=y_w,
            predicted_probs=probs_w,
            source_cal_acc=elec_pipeline.source_cal_acc,
            monitor_estimated_acc=est_cbpe,
            monitor_risk_score=tg_out["risk_tg"],
        )
        elec_oracle_results.append(eval_res)

        elec_records.append({
            "window_id": w_idx,
            "start_sample": start,
            "end_sample": end,
            "true_accuracy": eval_res.true_accuracy,
            "true_degradation": eval_res.true_degradation,
            "is_degraded": eval_res.is_degraded,
            "est_ac": est_ac,
            "est_doc": est_doc,
            "est_atc": est_atc,
            "est_cbpe": est_cbpe,
            "est_cott": est_cott,
            "ks_max": ks_val,
            "psi_max": psi_val,
            "jsd_max": jsd_val,
            "tg_risk": tg_out["risk_tg"],
            "tg_alarm": tg_out["is_alarm"],
            "s_attr": tg_out["s_attr"],
            "s_dis": tg_out["s_dis"],
            "s_conf": tg_out["s_conf"],
            "s_grad": tg_out["s_grad"],
            "s_d3m": tg_out["s_d3m"],
        })

    elec_df = pd.DataFrame(elec_records)
    elec_win_path = os.path.join("results", "stage2_electricity_windows.csv")
    elec_df.to_csv(elec_win_path, index=False)
    print(f"[+] Saved Electricity window records to: {elec_win_path}")

    # Electricity aggregate metrics
    elec_eval = oracle.evaluate_stream(elec_oracle_results, risk_alarm_threshold=elec_tg.alarm_threshold)

    # MAE per estimator
    mae_ac = float(np.mean(np.abs(elec_df["est_ac"] - elec_df["true_accuracy"])))
    mae_doc = float(np.mean(np.abs(elec_df["est_doc"] - elec_df["true_accuracy"])))
    mae_atc = float(np.mean(np.abs(elec_df["est_atc"] - elec_df["true_accuracy"])))
    mae_cbpe = float(np.mean(np.abs(elec_df["est_cbpe"] - elec_df["true_accuracy"])))
    mae_cott = float(np.mean(np.abs(elec_df["est_cott"] - elec_df["true_accuracy"])))

    elec_agg_df = pd.DataFrame([{
        "dataset": "OpenML_Electricity_151",
        "n_windows": n_windows,
        "window_size": window_size,
        "n_degraded_windows": elec_eval["n_degraded_windows"],
        "n_benign_windows": elec_eval["n_benign_windows"],
        "detection_auroc": elec_eval["detection_auroc"],
        "detection_auprc": elec_eval["detection_auprc"],
        "far_benign": elec_eval["far_benign"],
        "spearman_rho": elec_eval["spearman_rank_correlation"],
        "spearman_p": elec_eval["spearman_p_value"],
        "mae_ac": mae_ac,
        "mae_doc": mae_doc,
        "mae_atc": mae_atc,
        "mae_cbpe": mae_cbpe,
        "mae_cott": mae_cott,
    }])
    elec_agg_path = os.path.join("results", "stage2_electricity_aggregate.csv")
    elec_agg_df.to_csv(elec_agg_path, index=False)
    print(f"[+] Saved Electricity aggregate summary to: {elec_agg_path}")

    # =====================================================================
    # PART 3: PAIRED STATISTICAL SIGNIFICANCE TESTS
    # =====================================================================
    print("\n--- PART 3: Paired Statistical Significance Testing (Wilcoxon + FDR) ---")
    stat_records = []

    # Paired comparisons of absolute error across Electricity deployment windows
    err_ac = np.abs(elec_df["est_ac"] - elec_df["true_accuracy"])
    err_doc = np.abs(elec_df["est_doc"] - elec_df["true_accuracy"])
    err_atc = np.abs(elec_df["est_atc"] - elec_df["true_accuracy"])
    err_cbpe = np.abs(elec_df["est_cbpe"] - elec_df["true_accuracy"])
    err_cott = np.abs(elec_df["est_cott"] - elec_df["true_accuracy"])

    pairs = [
        ("CBPE_vs_ATC", err_cbpe, err_atc),
        ("CBPE_vs_COTT", err_cbpe, err_cott),
        ("ATC_vs_COTT", err_atc, err_cott),
        ("CBPE_vs_DoC", err_cbpe, err_doc),
        ("CBPE_vs_AC", err_cbpe, err_ac),
    ]

    raw_p_values = []
    for pair_name, d1, d2 in pairs:
        diff = d1 - d2
        if np.all(diff == 0.0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                res = wilcoxon(d1, d2, zero_method="wilcox")
                stat, p_val = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, p_val = 0.0, 1.0
        raw_p_values.append(p_val)

        stat_records.append({
            "test_pair": pair_name,
            "unit": "Deployment Window (m=500)",
            "n_units": len(d1),
            "wilcoxon_stat": stat,
            "raw_p_value": p_val,
            "mean_error_diff": float(np.mean(diff)),
        })

    # Apply Benjamini-Hochberg FDR correction
    rejects, adj_p = benjamini_hochberg(np.array(raw_p_values), alpha=0.05)
    for i in range(len(stat_records)):
        stat_records[i]["fdr_adjusted_p_value"] = float(adj_p[i])
        stat_records[i]["is_significant_05"] = bool(rejects[i])

    stat_df = pd.DataFrame(stat_records)
    stat_path = os.path.join("results", "stage2_statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)
    print(f"[+] Saved statistical test results to: {stat_path}")

    # =====================================================================
    # PART 4: DIAGNOSTIC PLOTS
    # =====================================================================
    print("\n--- PART 4: Diagnostic Comparison Figures ---")
    from src.evaluation.plots import generate_stage2_plots
    plot_path = os.path.join("results", "stage2_monitor_comparison.png")
    generate_stage2_plots(
        elec_windows_csv=elec_win_path,
        elec_agg_csv=elec_agg_path,
        synth_summary_csv=synth_path,
        output_png=plot_path,
    )

    print("\n" + "=" * 80)
    print("STAGE 2 BENCHMARKING COMPLETE")
    print(f"Total Execution Time: {time.time() - start_total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_stage2_benchmarks()
