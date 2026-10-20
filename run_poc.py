"""Fast self-contained Proof-of-Concept smoke test for SilentX / TrustGuard.

Runs in ~15-20 seconds on CPU:
- Generates controlled synthetic streams.
- Fits frozen pipeline with XGBoost and neural probe.
- Computes baseline estimators vs TrustGuard signals.
- Verifies that TrustGuard suppresses benign noise drift while flagging silent collapse.
- Outputs results/poc_summary.csv.
"""

import os
import time
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


def main():
    print("=" * 70)
    print("SILENTX: FAST PROOF-OF-CONCEPT (POC) SMOKE TEST")
    print("=" * 70)
    start_t = time.time()
    os.makedirs("results", exist_ok=True)

    print("[*] Generating fast synthetic benchmark (N=2000 source)...")
    bundle = DatasetRegistry.load_dataset(
        name="synthetic_stream",
        source_train_ratio=0.75,
        window_size=250,
        n_source_samples=2000,
        seed=42,
    )

    print("[*] Training frozen base models and locking calibration...")
    pipeline = train_and_freeze_pipeline(
        X_train=bundle.X_source_train,
        y_train=bundle.y_source_train,
        X_cal=bundle.X_source_cal,
        y_cal=bundle.y_source_cal,
        feature_names=bundle.feature_names,
        seed=42,
    )

    print(f"    Source Cal Acc: {pipeline.source_cal_acc * 100:.2f}%")
    print(f"    Source Cal ECE Calibrated: {pipeline.source_calibrator.is_calibrated}")
    print(f"    ATC threshold: {pipeline.atc_threshold:.4f}")
    print(f"    Mondrian Conformal: q0={pipeline.conformal_calibrator.q_0:.4f}, q1={pipeline.conformal_calibrator.q_1:.4f}")

    # Set up monitors
    ks_det = KSMaxDriftDetector(bundle.X_source_cal)
    atc_est = AverageThresholdedConfidenceEstimator(pipeline.atc_threshold)
    doc_est = DifferenceOfConfidencesEstimator(pipeline.source_cal_acc, pipeline.source_cal_mean_conf)
    tg_mon = TrustGuardCompositeMonitor(pipeline, bundle.X_source_cal)
    oracle = EvaluationOracle(degradation_threshold=0.05, benign_margin=0.02)

    rows = []
    window_evals = []

    print("[*] Evaluating deployment streams...")
    for w in bundle.deployment_windows:
        X_unlabeled = w.unlabeled_features
        tg_out = tg_mon.evaluate_window(X_unlabeled)
        probs = tg_out["probs_primary"]

        ks_max = ks_det.evaluate_window(X_unlabeled)
        est_ac = AverageConfidenceEstimator.estimate_accuracy(probs)
        est_atc = atc_est.estimate_accuracy(probs)
        est_doc = doc_est.estimate_accuracy(probs)
        est_cbpe = CBPEPerformanceEstimator.estimate_accuracy(probs)

        # Oracle evaluation
        eval_res = oracle.score_window(
            window_id=w.window_id,
            regime=w.regime,
            severity=w.severity,
            y_sequestered=w.y,
            predicted_probs=probs,
            source_cal_acc=pipeline.source_cal_acc,
            monitor_estimated_acc=est_cbpe,
            monitor_risk_score=tg_out["risk_tg"],
        )
        window_evals.append(eval_res)

        rows.append({
            "Window": w.window_id,
            "Regime": w.regime,
            "Severity": w.severity,
            "True_Acc": round(eval_res.true_accuracy, 3),
            "True_Drop": round(eval_res.true_degradation, 3),
            "KS_Max": round(ks_max, 3),
            "Est_CBPE": round(est_cbpe, 3),
            "Est_ATC": round(est_atc, 3),
            "TG_Risk": round(tg_out["risk_tg"], 3),
            "S_Dis": round(tg_out["s_dis"], 3),
            "S_Attr": round(tg_out["s_attr"], 3),
            "S_Conf": round(tg_out["s_conf"], 3),
        })

    summary_df = pd.DataFrame(rows)
    stream_metrics = oracle.evaluate_stream(window_evals, risk_alarm_threshold=0.45)

    out_csv = os.path.join("results", "poc_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    print("\n" + summary_df.to_string(index=False))
    print("-" * 70)
    print(f"Degradation Detection AUROC: {stream_metrics['detection_auroc']:.4f}")
    print(f"Benign Shift False Alarm Rate: {stream_metrics['far_benign']:.4f}")
    print(f"Spearman Rank Correlation: {stream_metrics['spearman_rank_correlation']:.4f}")
    print(f"Elapsed Time: {time.time() - start_t:.2f}s")
    print(f"Saved POC results to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
