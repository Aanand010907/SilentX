"""Stage 2 Comprehensive Monitoring Benchmark Runner for SilentX.

Scientific Audit, Algorithm Verification, and Empirical Revalidation.

Executes:
1. Multi-seed synthetic benchmark (seeds: 42, 123, 999) across all 6 shift regimes.
2. Full Real-world OpenML Electricity (ID 151) chronological deployment stream benchmark:
   - 15,000 source training instances.
   - 2,500 source calibration fitting instances (probability calibration).
   - 2,500 source calibration reference instances (threshold setting & out-of-fold evaluation).
   - 25,312 deployment instances evaluated in 51 sequential windows (50 full m=500 + 1 final m=312).
3. Evaluates all baseline monitors:
   - Drift: KS_max, PSI_max, JSD_max
   - Confidence / Transport Estimators: AC, DoC, ATC, CBPE, Published COTT, Scalar OT Heuristic
   - TrustGuard: S_attr, S_dis, S_conf, S_grad, S_d3m, Risk_TG, alarm
4. Sequestered Oracle scoring: Accuracy MAE, AUROC, AUPRC, FAR_benign, Spearman rho_s.
5. Time-Series Moving-Block Bootstrap (MBB) (Kunsch 1989) with block lengths b in {2, 3, 5, 8}
   and Benjamini-Hochberg FDR adjustment, alongside paired Wilcoxon signed-rank tests.
6. Generates publication-grade diagnostic plots in results/stage2_monitor_comparison.png.
7. Saves all validated artifacts:
   - results/stage2_synthetic_summary.csv
   - results/electricity_manifest.json
   - results/stage2_electricity_windows.csv
   - results/stage2_electricity_aggregate.csv
   - results/stage2_statistical_tests.csv
   - results/stage2_monitor_comparison.png
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
    ScalarConfidenceOTHeuristic,
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

    rev_idx = np.empty(n, dtype=int)
    rev_idx[sorted_idx] = np.arange(n)
    final_adj_p = adjusted_p[rev_idx]
    reject = (final_adj_p <= alpha)
    return reject, final_adj_p


def moving_block_bootstrap(
    diff_series: np.ndarray,
    block_length: int = 3,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Moving-Block Bootstrap (MBB) (Kunsch 1989) for autocorrelated time series.

    Returns:
        (observed_mean, ci_lower, ci_upper, two_sided_p_value)
    """
    diff = np.asarray(diff_series, dtype=float).ravel()
    n = len(diff)
    if n < block_length:
        block_length = max(1, n)

    n_blocks = n - block_length + 1
    blocks = [diff[i : i + block_length] for i in range(n_blocks)]
    rng = np.random.default_rng(seed)

    k = int(np.ceil(n / block_length))
    boot_means = []
    for _ in range(n_bootstraps):
        chosen_idx = rng.choice(n_blocks, size=k, replace=True)
        sample = np.concatenate([blocks[i] for i in chosen_idx])[:n]
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    obs_mean = float(np.mean(diff))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    # Two-sided empirical p-value testing H0: mean(diff) <= 0
    p_val = 2.0 * min(float(np.mean(boot_means <= 0.0)), float(np.mean(boot_means >= 0.0)))
    return obs_mean, ci_low, ci_high, min(1.0, max(0.0, p_val))


def run_stage2_benchmarks():
    print("=" * 80)
    print("SILENTX: STAGE 2 COMPREHENSIVE REVALIDATED BENCHMARK (COTT & ELECTRICITY)")
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
        y_cal_arr = bundle.y_source_cal.to_numpy()

        # Published COTT Estimator
        cott_est = ConfidenceOptimalTransportEstimator.fit(
            cal_probs=cal_probs,
            cal_labels=y_cal_arr,
            cal_acc=pipeline.source_cal_acc,
            max_cal_samples=100,
            seed=seed,
        )

        # Experimental Scalar OT Heuristic
        scalar_ot_est = ScalarConfidenceOTHeuristic.fit(
            cal_probs=cal_probs,
            cal_labels=y_cal_arr,
            max_samples=100,
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
            est_cott = cott_est.estimate_accuracy(probs_prim, batch_size=100, seed=seed)
            est_scalar_ot = scalar_ot_est.estimate_accuracy(probs_prim, max_target_samples=100)

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
                monitor_estimated_acc=est_cott,
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
                "est_scalar_ot": est_scalar_ot,
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
        n_train_samples=15000,
        n_cal_samples=5000,
        manifest_path="results/electricity_manifest.json",
    )
    print(f"[*] Loaded OpenML Electricity: {elec_data['n_total']} rows total, real download: {elec_data['is_real']}")

    # Train base model on early training split (first 15,000 instances)
    # Using separated calibration-fit (2,500) and calibration-ref (2,500)
    elec_pipeline = train_and_freeze_pipeline(
        X_train=elec_data["X_train"],
        y_train=elec_data["y_train"],
        X_cal=elec_data["X_cal_fit"],
        y_cal=elec_data["y_cal_fit"],
        X_cal_ref=elec_data["X_cal_ref"],
        y_cal_ref=elec_data["y_cal_ref"],
        feature_names=elec_data["feature_names"],
        seed=42,
    )
    print(f"    Electricity Source Cal Acc: {elec_pipeline.source_cal_acc * 100:.2f}%")
    print(f"    Electricity Source Cal Mean Conf: {elec_pipeline.source_cal_mean_conf * 100:.2f}%")
    print(f"    Electricity DoC Offset: {elec_pipeline.doc_offset:.4f}")
    print(f"    Electricity ATC Threshold: {elec_pipeline.atc_threshold:.4f}")

    # Set up monitors on source reference split
    elec_ks = KSMaxDriftDetector(elec_data["X_cal_ref"])
    elec_psi = PopulationStabilityIndexDetector(elec_data["X_cal_ref"], n_bins=10)
    elec_jsd = JensenShannonDivergenceDetector(elec_data["X_cal_ref"], n_bins=10)
    elec_atc = AverageThresholdedConfidenceEstimator(elec_pipeline.atc_threshold)
    elec_doc = DifferenceOfConfidencesEstimator(elec_pipeline.source_cal_acc, elec_pipeline.source_cal_mean_conf)
    elec_cbpe = CBPEPerformanceEstimator()

    cal_ref_probs = elec_pipeline.predict_primary_proba(elec_data["X_cal_ref"])
    y_cal_ref = elec_data["y_cal_ref"].to_numpy()

    # Published COTT
    elec_cott = ConfidenceOptimalTransportEstimator.fit(
        cal_probs=cal_ref_probs,
        cal_labels=y_cal_ref,
        cal_acc=elec_pipeline.source_cal_acc,
        max_cal_samples=150,
        seed=42,
    )
    print(f"    Electricity COTT Cost Threshold t*: {elec_cott.cost_threshold:.4f}")

    # Scalar OT Heuristic
    elec_scalar_ot = ScalarConfidenceOTHeuristic.fit(
        cal_probs=cal_ref_probs,
        cal_labels=y_cal_ref,
        max_samples=150,
        seed=42,
    )

    elec_tg = TrustGuardCompositeMonitor(elec_pipeline, elec_data["X_cal_ref"], seed=42)

    # Partition full deployment stream (25,312 instances) into windows
    window_size = 500
    X_deploy = elec_data["X_deploy"]
    y_deploy = elec_data["y_deploy"]
    total_deploy = len(X_deploy)
    n_windows = int(np.ceil(total_deploy / window_size))

    elec_records = []
    elec_oracle_results = []

    print(f"[*] Streaming all {total_deploy} deployment instances into {n_windows} sequential windows...")
    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = min(total_deploy, start + window_size)
        X_w = X_deploy.iloc[start:end].reset_index(drop=True)
        y_w = y_deploy.iloc[start:end].reset_index(drop=True)
        win_size = len(X_w)
        is_partial = (win_size < window_size)

        probs_w = elec_pipeline.predict_primary_proba(X_w)

        # Baseline estimates
        est_ac = AverageConfidenceEstimator.estimate_accuracy(probs_w)
        est_doc = elec_doc.estimate_accuracy(probs_w)
        est_atc = elec_atc.estimate_accuracy(probs_w)
        est_cbpe = elec_cbpe.estimate_accuracy(probs_w)
        est_cott = elec_cott.estimate_accuracy(probs_w, batch_size=150, seed=42)
        est_scalar_ot = elec_scalar_ot.estimate_accuracy(probs_w, max_target_samples=150)

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
            monitor_estimated_acc=est_cott,
            monitor_risk_score=tg_out["risk_tg"],
        )
        elec_oracle_results.append(eval_res)

        elec_records.append({
            "window_id": w_idx,
            "start_sample": start,
            "end_sample": end,
            "n_samples": win_size,
            "is_partial_window": is_partial,
            "true_accuracy": eval_res.true_accuracy,
            "true_degradation": eval_res.true_degradation,
            "is_degraded": eval_res.is_degraded,
            "est_ac": est_ac,
            "est_doc": est_doc,
            "est_atc": est_atc,
            "est_cbpe": est_cbpe,
            "est_cott": est_cott,
            "est_scalar_ot": est_scalar_ot,
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
    print(f"[+] Saved Electricity window records ({len(elec_df)} windows) to: {elec_win_path}")

    # Electricity aggregate metrics
    elec_eval = oracle.evaluate_stream(elec_oracle_results, risk_alarm_threshold=elec_tg.alarm_threshold)

    # MAE per estimator across all 51 windows
    mae_ac = float(np.mean(np.abs(elec_df["est_ac"] - elec_df["true_accuracy"])))
    mae_doc = float(np.mean(np.abs(elec_df["est_doc"] - elec_df["true_accuracy"])))
    mae_atc = float(np.mean(np.abs(elec_df["est_atc"] - elec_df["true_accuracy"])))
    mae_cbpe = float(np.mean(np.abs(elec_df["est_cbpe"] - elec_df["true_accuracy"])))
    mae_cott = float(np.mean(np.abs(elec_df["est_cott"] - elec_df["true_accuracy"])))
    mae_scalar_ot = float(np.mean(np.abs(elec_df["est_scalar_ot"] - elec_df["true_accuracy"])))

    elec_agg_df = pd.DataFrame([{
        "dataset": "OpenML_Electricity_151",
        "n_total_instances": elec_data["n_total"],
        "n_deployment_instances": total_deploy,
        "n_windows": n_windows,
        "window_size_nominal": window_size,
        "n_degraded_windows": elec_eval["n_degraded_windows"],
        "n_benign_windows": elec_eval["n_benign_windows"],
        "detection_auroc": elec_eval["detection_auroc"],
        "detection_auprc": elec_eval["detection_auprc"],
        "far_benign": elec_eval["far_benign"],
        "spearman_rho": elec_eval["spearman_rank_correlation"],
        "spearman_p": elec_eval["spearman_p_value"],
        "mae_cott": mae_cott,
        "mae_scalar_ot": mae_scalar_ot,
        "mae_cbpe": mae_cbpe,
        "mae_atc": mae_atc,
        "mae_doc": mae_doc,
        "mae_ac": mae_ac,
    }])
    elec_agg_path = os.path.join("results", "stage2_electricity_aggregate.csv")
    elec_agg_df.to_csv(elec_agg_path, index=False)
    print(f"[+] Saved Electricity aggregate summary to: {elec_agg_path}")

    # =====================================================================
    # PART 3: TIME-SERIES STATISTICAL SIGNIFICANCE TESTS (MBB + FDR)
    # =====================================================================
    print("\n--- PART 3: Time-Series Statistical Testing (Moving-Block Bootstrap + FDR) ---")
    stat_records = []

    err_ac = np.abs(elec_df["est_ac"] - elec_df["true_accuracy"])
    err_doc = np.abs(elec_df["est_doc"] - elec_df["true_accuracy"])
    err_atc = np.abs(elec_df["est_atc"] - elec_df["true_accuracy"])
    err_cbpe = np.abs(elec_df["est_cbpe"] - elec_df["true_accuracy"])
    err_cott = np.abs(elec_df["est_cott"] - elec_df["true_accuracy"])
    err_scalar_ot = np.abs(elec_df["est_scalar_ot"] - elec_df["true_accuracy"])

    pairs = [
        ("CBPE_vs_COTT", err_cbpe, err_cott),
        ("ATC_vs_COTT", err_atc, err_cott),
        ("DoC_vs_COTT", err_doc, err_cott),
        ("AC_vs_COTT", err_ac, err_cott),
        ("ScalarOT_vs_COTT", err_scalar_ot, err_cott),
        ("CBPE_vs_ATC", err_cbpe, err_atc),
    ]

    mbb_p_values = []
    wilcox_p_values = []

    for pair_name, d_base, d_comp in pairs:
        # diff > 0 implies d_comp has lower error than d_base
        diff = d_base - d_comp

        # Moving-Block Bootstrap with block length b = 3
        obs_m, ci_l, ci_h, p_mbb_b3 = moving_block_bootstrap(diff, block_length=3, n_bootstraps=2000, seed=42)
        mbb_p_values.append(p_mbb_b3)

        # Sensitivity across block lengths b in {2, 5, 8}
        _, _, _, p_b2 = moving_block_bootstrap(diff, block_length=2, n_bootstraps=1000, seed=42)
        _, _, _, p_b5 = moving_block_bootstrap(diff, block_length=5, n_bootstraps=1000, seed=42)
        _, _, _, p_b8 = moving_block_bootstrap(diff, block_length=8, n_bootstraps=1000, seed=42)

        # Supplemental Wilcoxon signed-rank test
        if np.all(diff == 0.0):
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                w_res = wilcoxon(d_base, d_comp, zero_method="wilcox")
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = 0.0, 1.0
        wilcox_p_values.append(w_pval)

        stat_records.append({
            "test_pair": pair_name,
            "statistical_unit": "Chronological Deployment Window (m=500)",
            "n_windows": len(diff),
            "mean_error_diff": obs_m,
            "mbb_b3_ci_lower": ci_l,
            "mbb_b3_ci_upper": ci_h,
            "mbb_b3_raw_p": p_mbb_b3,
            "mbb_sensitivity_p_b2": p_b2,
            "mbb_sensitivity_p_b5": p_b5,
            "mbb_sensitivity_p_b8": p_b8,
            "wilcoxon_stat": w_stat,
            "wilcoxon_raw_p": w_pval,
        })

    # Benjamini-Hochberg FDR correction on primary MBB p-values (b=3)
    rejects_mbb, adj_p_mbb = benjamini_hochberg(np.array(mbb_p_values), alpha=0.05)
    _, adj_p_wilcox = benjamini_hochberg(np.array(wilcox_p_values), alpha=0.05)

    for i in range(len(stat_records)):
        stat_records[i]["mbb_fdr_adj_p"] = float(adj_p_mbb[i])
        stat_records[i]["mbb_significant_05"] = bool(rejects_mbb[i])
        stat_records[i]["wilcoxon_fdr_adj_p"] = float(adj_p_wilcox[i])

    stat_df = pd.DataFrame(stat_records)
    stat_path = os.path.join("results", "stage2_statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)
    print(f"[+] Saved time-series statistical test results to: {stat_path}")

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
    print("STAGE 2 REVALIDATED BENCHMARK COMPLETE")
    print(f"Total Execution Time: {time.time() - start_total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_stage2_benchmarks()
