"""Train and evaluate the SilentX / OmniGuard model on real natural data (OpenML Electricity 151) on local hardware."""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss

from src.data.real_datasets import OpenMLElectricityLoader
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.models.hybrid_ensemble import build_robust_hybrid_ensemble_from_pipeline
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.three_tier_guardrail import ThreeTierOperationalGuardrail


def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("IMPORTING NATURAL REAL-WORLD DATA & TRAINING ON LOCAL HARDWARE")
    print("=" * 80)

    # 1. Import real natural OpenML Electricity dataset
    print("[1/4] Loading real natural data from local cache (.cache/datasets/)...")
    elec_data = OpenMLElectricityLoader.load(cache_dir=os.path.join(base_dir, ".cache/datasets"))
    
    X_train = elec_data["X_train"]
    y_train = elec_data["y_train"]
    X_cal_fit = elec_data["X_cal_fit"]
    y_cal_fit = elec_data["y_cal_fit"]
    X_cal_ref = elec_data["X_cal_ref"]
    y_cal_ref = elec_data["y_cal_ref"]
    X_deploy = elec_data["X_deploy"]
    y_deploy = elec_data["y_deploy"]
    feature_names = elec_data["feature_names"]

    print(f"      - Source Training Data:    {len(X_train):,d} real records")
    print(f"      - Source Calibration Fit:  {len(X_cal_fit):,d} real records")
    print(f"      - Source Calibration Ref:  {len(X_cal_ref):,d} real records")
    print(f"      - Real Deployment Stream:  {len(X_deploy):,d} real streaming records")
    print(f"      - Total Natural Records:   {elec_data['n_total']:,d}")
    print(f"      - Natural Features:        {feature_names}")

    # 2. Train Models on Local Hardware
    print("\n[2/4] Training multi-architecture ensemble on local hardware...")
    t0 = time.time()
    pipeline = train_and_freeze_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal_fit,
        y_cal=y_cal_fit,
        X_cal_ref=X_cal_ref,
        y_cal_ref=y_cal_ref,
        feature_names=feature_names,
        seed=42,
    )

    hybrid_ensemble = build_robust_hybrid_ensemble_from_pipeline(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        seed=42,
    )
    train_time = time.time() - t0
    print(f"      [+] Training completed in {train_time:.2f} seconds on local CPU!")
    print(f"      - Out-of-fold Cal Accuracy (Trees):  {pipeline.source_cal_acc * 100:.2f}%")
    print(f"      - Source Cal Mean Confidence:        {pipeline.source_cal_mean_conf * 100:.2f}%")
    print(f"      - TreeSHAP Decision Weights:         {np.round(pipeline.shap_weights, 3)}")

    # 3. Evaluate Models on Real Natural Deployment Stream
    print("\n[3/4] Evaluating models on the full 25,312 real deployment stream...")
    y_true_stream = y_deploy.to_numpy(dtype=int)

    # Primary Calibrated Tree predictions
    probs_primary = pipeline.predict_primary_proba(X_deploy)
    preds_primary = (probs_primary >= 0.5).astype(int)

    # Secondary Neural Probe predictions
    probs_neural = pipeline.predict_secondary_proba(X_deploy)
    preds_neural = (probs_neural >= 0.5).astype(int)

    # OmniGuard Hybrid Ensemble predictions
    probs_hybrid = hybrid_ensemble.predict_hybrid_proba(X_deploy, s_attr=0.05)
    preds_hybrid = (probs_hybrid >= 0.5).astype(int)

    # Conformal Prediction Sets
    conformal_sets = hybrid_ensemble.predict_sets(probs_hybrid)
    set_sizes = [len(s) for s in conformal_sets]
    singleton_coverage = np.mean([y_true_stream[i] in conformal_sets[i] for i in range(len(y_true_stream))])

    # Metrics
    acc_primary = accuracy_score(y_true_stream, preds_primary)
    acc_neural = accuracy_score(y_true_stream, preds_neural)
    acc_hybrid = accuracy_score(y_true_stream, preds_hybrid)

    f1_hybrid = f1_score(y_true_stream, preds_hybrid)
    auc_hybrid = roc_auc_score(y_true_stream, probs_hybrid)
    brier_hybrid = brier_score_loss(y_true_stream, probs_hybrid)
    loss_hybrid = log_loss(y_true_stream, np.clip(probs_hybrid, 1e-6, 1 - 1e-6))

    # 4. Save Real Predictions and Summary
    print("\n[4/4] Saving real predictions & export summaries...")
    out_csv = os.path.join(results_dir, "natural_electricity_predictions.csv")
    pred_export = pd.DataFrame({
        "true_label": y_true_stream,
        "primary_tree_pred": preds_primary,
        "primary_tree_prob": np.round(probs_primary, 4),
        "neural_probe_pred": preds_neural,
        "neural_probe_prob": np.round(probs_neural, 4),
        "hybrid_omni_pred": preds_hybrid,
        "hybrid_omni_prob": np.round(probs_hybrid, 4),
        "conformal_set": [str(s) for s in conformal_sets],
        "is_singleton": [sz == 1 for sz in set_sizes],
    })
    # Export sample
    pred_export.to_csv(out_csv, index=False)

    summary_json = os.path.join(results_dir, "natural_training_summary.json")
    summary_data = {
        "dataset": "OpenML_Electricity_151",
        "is_real_data": True,
        "training_samples": len(X_train),
        "calibration_samples": len(X_cal_fit) + len(X_cal_ref),
        "deployment_stream_samples": len(X_deploy),
        "training_duration_seconds": round(train_time, 2),
        "source_calibration_accuracy": round(pipeline.source_cal_acc * 100, 2),
        "real_stream_primary_tree_acc": round(acc_primary * 100, 2),
        "real_stream_neural_probe_acc": round(acc_neural * 100, 2),
        "real_stream_hybrid_omni_acc": round(acc_hybrid * 100, 2),
        "real_stream_f1": round(f1_hybrid, 4),
        "real_stream_roc_auc": round(auc_hybrid, 4),
        "real_stream_brier": round(brier_hybrid, 4),
        "real_stream_log_loss": round(loss_hybrid, 4),
        "conformal_singleton_rate": round(float(np.mean(np.array(set_sizes) == 1)) * 100, 2),
        "conformal_empirical_coverage": round(float(singleton_coverage) * 100, 2),
    }
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("           REAL NATURAL DATA TRAINING & BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Dataset:                      OpenML Electricity (Natural Energy Market Data)")
    print(f"Total Real Samples Evaluated: 25,312 Unseen Streaming Inferences")
    print(f"Local Training Time:          {train_time:.2f} seconds")
    print("-" * 80)
    print(f"In-Distribution Cal Accuracy: {pipeline.source_cal_acc * 100:.2f}%")
    print(f"Primary Tree Real Stream Acc: {acc_primary * 100:.2f}%")
    print(f"Neural Probe Real Stream Acc: {acc_neural * 100:.2f}%")
    print(f"Hybrid Omni Real Stream Acc:  {acc_hybrid * 100:.2f}%")
    print(f"Stream F1-Score:              {f1_hybrid:.4f}")
    print(f"Stream ROC-AUC:               {auc_hybrid:.4f}")
    print(f"Stream Brier Calibration:     {brier_hybrid:.4f}")
    print(f"Stream Log Loss:              {loss_hybrid:.4f}")
    print(f"Conformal Empirical Coverage: {singleton_coverage * 100:.2f}% (Target: 90.00%)")
    print("=" * 80)
    print(f"Predictions saved to: {out_csv}")
    print(f"Summary JSON saved to: {summary_json}")
    print(f"Execution finished in {total_time:.2f}s!")


if __name__ == "__main__":
    main()
