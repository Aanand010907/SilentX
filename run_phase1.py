"""Stage 1 Execution Runner for SilentX / TrustGuard.

Executes:
1. Environment and schema verification.
2. Source data loading and 75/25 stratified source partitioning.
3. Multi-architecture model training (Primary Tree, Secondary MLP, VBLL Head, Diagnostic LogReg).
4. Source-only calibration evaluation (3-fold CV ECE check).
5. Mondrian split-conformal quantile locking (alpha = 0.10, finite-sample corrected).
6. D3M bootstrap reference disagreement calibration across query sizes m in {10, 20, 50}.
7. TreeSHAP reliance weight computation.
8. Unlabeled deployment window evaluation and post-hoc Oracle scoring.
9. Serialization of Stage 1 calibration and benchmark artifacts to results/.
"""

import os
import json
import time
import yaml
import numpy as np
import pandas as pd

from src.data.dataset_registry import DatasetRegistry
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.baselines import (
    KSMaxDriftDetector,
    AverageConfidenceEstimator,
    AverageThresholdedConfidenceEstimator,
    DifferenceOfConfidencesEstimator,
    CBPEPerformanceEstimator,
)
from src.evaluation.oracle import EvaluationOracle


def run_stage1_pipeline(config_path: str = "configs/experiment.yaml"):
    print("=" * 80)
    print("SILENTX: STAGE 1 RESEARCH PIPELINE & CALIBRATION LOCK")
    print("=" * 80)

    start_time = time.time()
    os.makedirs("results", exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["random_seeds"][0]
    print(f"[*] Initializing with random seed: {seed}")

    # 1. Load Dataset via unified data contract
    print("[*] Loading synthetic benchmark dataset with 75/25 stratified source split...")
    bundle = DatasetRegistry.load_dataset(
        name="synthetic_stream",
        source_train_ratio=config["experiment"]["source_train_ratio"],
        window_size=config["calibration"]["d3m"]["query_sizes"][-1] * 10,  # e.g., 500
        n_source_samples=config["data"]["synthetic"]["n_samples"],
        seed=seed,
    )

    print(f"    - Source Train size: {len(bundle.X_source_train)} samples")
    print(f"    - Source Calibration size: {len(bundle.X_source_cal)} samples")
    print(f"    - Total Deployment Windows: {len(bundle.deployment_windows)} windows")

    # 2. Train Base Learners and Lock Frozen Inference Pipeline
    print("[*] Training base learners & computing source calibration statistics...")
    pipeline = train_and_freeze_pipeline(
        X_train=bundle.X_source_train,
        y_train=bundle.y_source_train,
        X_cal=bundle.X_source_cal,
        y_cal=bundle.y_source_cal,
        feature_names=bundle.feature_names,
        tree_family=config["models"]["primary"]["family"],
        seed=seed,
    )

    print(f"    - Source Cal Accuracy: {pipeline.source_cal_acc * 100:.2f}%")
    print(f"    - Source Cal Mean Confidence: {pipeline.source_cal_mean_conf * 100:.2f}%")
    print(f"    - Isotonic Calibration Applied: {pipeline.source_calibrator.is_calibrated}")
    print(f"    - 3-Fold Raw ECE: {pipeline.source_calibrator.cv_results_.get('mean_raw_ece', 0.0):.4f}")
    print(f"    - 3-Fold Cal ECE: {pipeline.source_calibrator.cv_results_.get('mean_cal_ece', 0.0):.4f}")
    print(f"    - ATC Error Threshold (t*): {pipeline.atc_threshold:.4f}")
    print(f"    - DoC Offset: {pipeline.doc_offset:.4f}")
    print(f"    - Mondrian Conformal Quantiles: q_0 = {pipeline.conformal_calibrator.q_0:.4f}, q_1 = {pipeline.conformal_calibrator.q_1:.4f}")
    print(f"    - Source Baseline GdScore: {pipeline.source_cal_gdscore:.4f}")
    print(f"    - D3M Locked Disagreement Thresholds: {pipeline.d3m_calibrator.thresholds_}")

    # 3. Initialize Monitors (Monitors receive ONLY unlabeled features & frozen model states)
    print("[*] Initializing monitoring engine (Strict Ground-Truth Isolation enforced)...")
    ks_monitor = KSMaxDriftDetector(X_cal=bundle.X_source_cal)
    atc_estimator = AverageThresholdedConfidenceEstimator(atc_threshold=pipeline.atc_threshold)
    doc_estimator = DifferenceOfConfidencesEstimator(
        source_cal_acc=pipeline.source_cal_acc,
        source_cal_mean_conf=pipeline.source_cal_mean_conf,
    )
    trustguard_monitor = TrustGuardCompositeMonitor(
        frozen_pipeline=pipeline, X_cal=bundle.X_source_cal
    )

    oracle = EvaluationOracle(
        degradation_threshold=config["evaluation"]["degradation_threshold"],
        benign_margin=config["evaluation"]["benign_margin"],
    )

    # 4. Stream Deployment Windows Evaluation
    print("[*] Streaming unlabeled deployment windows through monitors and scoring via Oracle...")
    window_results = []
    monitor_audit_records = []

    for window in bundle.deployment_windows:
        # A. Monitoring Phase: UNLABELED INPUT ONLY
        X_unlabeled = window.unlabeled_features
        tg_out = trustguard_monitor.evaluate_window(X_unlabeled)
        probs_primary = tg_out["probs_primary"]

        # Baseline estimates
        est_ac = AverageConfidenceEstimator.estimate_accuracy(probs_primary)
        est_atc = atc_estimator.estimate_accuracy(probs_primary)
        est_doc = doc_estimator.estimate_accuracy(probs_primary)
        est_cbpe = CBPEPerformanceEstimator.estimate_accuracy(probs_primary)
        ks_val = ks_monitor.evaluate_window(X_unlabeled)

        # B. Evaluation Phase: SEQUESTERED GROUND-TRUTH VAULT ONLY
        oracle_eval = oracle.score_window(
            window_id=window.window_id,
            regime=window.regime,
            severity=window.severity,
            y_sequestered=window.y,
            predicted_probs=probs_primary,
            source_cal_acc=pipeline.source_cal_acc,
            monitor_estimated_acc=est_cbpe,
            monitor_risk_score=tg_out["risk_tg"],
        )
        window_results.append(oracle_eval)

        monitor_audit_records.append({
            "window_id": window.window_id,
            "regime": window.regime,
            "severity": window.severity,
            "true_accuracy": oracle_eval.true_accuracy,
            "true_degradation": oracle_eval.true_degradation,
            "is_degraded": oracle_eval.is_degraded,
            "est_ac": est_ac,
            "est_atc": est_atc,
            "est_doc": est_doc,
            "est_cbpe": est_cbpe,
            "ks_max": ks_val,
            "tg_risk": tg_out["risk_tg"],
            "tg_s_attr": tg_out["s_attr"],
            "tg_s_dis": tg_out["s_dis"],
            "tg_s_conf": tg_out["s_conf"],
            "tg_s_grad": tg_out["s_grad"],
        })

    # 5. Aggregate Oracle Evaluation Metrics
    stream_summary = oracle.evaluate_stream(window_results, risk_alarm_threshold=0.45)
    audit_df = pd.DataFrame(monitor_audit_records)

    # 6. Save Artifacts to results/
    calibration_artifact = {
        "source_cal_acc": pipeline.source_cal_acc,
        "source_cal_mean_conf": pipeline.source_cal_mean_conf,
        "isotonic_calibration_applied": pipeline.source_calibrator.is_calibrated,
        "cv_results": pipeline.source_calibrator.cv_results_,
        "atc_threshold": pipeline.atc_threshold,
        "doc_offset": pipeline.doc_offset,
        "mondrian_quantiles": {
            "q_0": pipeline.conformal_calibrator.q_0,
            "q_1": pipeline.conformal_calibrator.q_1,
        },
        "d3m_thresholds": pipeline.d3m_calibrator.thresholds_,
        "source_cal_gdscore": pipeline.source_cal_gdscore,
        "shap_weights": pipeline.shap_weights.tolist(),
        "stream_evaluation_summary": {
            "n_windows": stream_summary["n_windows"],
            "n_degraded_windows": stream_summary["n_degraded_windows"],
            "n_benign_windows": stream_summary["n_benign_windows"],
            "mean_mae_accuracy": stream_summary["mean_mae_accuracy"],
            "detection_auroc": stream_summary["detection_auroc"],
            "detection_auprc": stream_summary["detection_auprc"],
            "far_benign": stream_summary["far_benign"],
            "spearman_rank_correlation": stream_summary["spearman_rank_correlation"],
            "spearman_p_value": stream_summary["spearman_p_value"],
        },
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    cal_path = os.path.join("results", "stage1_calibration_summary.json")
    with open(cal_path, "w") as f:
        json.dump(calibration_artifact, f, indent=2)

    eval_path = os.path.join("results", "stage1_evaluation.csv")
    audit_df.to_csv(eval_path, index=False)

    print("\n" + "=" * 80)
    print("STAGE 1 EXECUTION COMPLETE")
    print("=" * 80)
    print(f"[+] Total Runtime: {time.time() - start_time:.2f}s")
    print(f"[+] Degradation Detection AUROC: {stream_summary['detection_auroc']:.4f}")
    print(f"[+] Benign-Shift False Alarm Rate (FAR_benign): {stream_summary['far_benign']:.4f}")
    print(f"[+] Spearman Rank Correlation (rho_s): {stream_summary['spearman_rank_correlation']:.4f}")
    print(f"[+] Calibration Summary saved to: {cal_path}")
    print(f"[+] Window Evaluation saved to: {eval_path}")

    return calibration_artifact, audit_df


if __name__ == "__main__":
    run_stage1_pipeline()
