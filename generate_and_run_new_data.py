"""Generate fresh unseen test data (new_data.csv), run SilentX / TrustGuard on it, and output predictions."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data.dataset_registry import DatasetRegistry
from src.data.synthetic_streams import SyntheticStreamGenerator
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.baselines import (
    KSMaxDriftDetector,
    AverageConfidenceEstimator,
    AverageThresholdedConfidenceEstimator,
    DifferenceOfConfidencesEstimator,
    CBPEPerformanceEstimator,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    new_data_path = os.path.join(base_dir, "new_data.csv")
    predictions_path = os.path.join(base_dir, "new_data_predictions.csv")

    print("=" * 80)
    print("STEP 1: GENERATING BRAND-NEW TEST DATA (new_data.csv)")
    print("=" * 80)

    # Use a fresh, completely unseen random seed (2026)
    gen = SyntheticStreamGenerator(seed=2026)
    
    # Create a realistic test stream of 1,200 instances:
    # 400 clean In-Distribution + 400 Harmless Noise Shift + 400 Concept Drift
    df_id, y_id, _ = gen.generate_regime(n_samples=400, regime="in_distribution", seed=20261)
    df_id["regime"] = "in_distribution"
    df_id["true_label"] = y_id

    df_noise, y_noise, _ = gen.generate_regime(n_samples=400, regime="harmless_drift", severity=1.0, seed=20262)
    df_noise["regime"] = "harmless_noise"
    df_noise["true_label"] = y_noise

    df_drift, y_drift, _ = gen.generate_regime(n_samples=400, regime="concept_shift", severity=0.75, seed=20263)
    df_drift["regime"] = "concept_drift"
    df_drift["true_label"] = y_drift

    new_data = pd.concat([df_id, df_noise, df_drift], ignore_index=True)
    new_data.to_csv(new_data_path, index=False)

    print(f"[+] Successfully generated new testing dataset!")
    print(f"    File path: {new_data_path}")
    print(f"    Total test samples: {len(new_data)}")
    print(f"    Features: {[col for col in new_data.columns if col.startswith('x_')]}")
    print(f"    Sub-regimes included:")
    for regime, count in new_data["regime"].value_counts().items():
        print(f"      - {regime}: {count} samples")

    print("\n" + "=" * 80)
    print("STEP 2: LOADING & TRAINING FROZEN MODEL PIPELINE ON SOURCE DATA")
    print("=" * 80)

    bundle = DatasetRegistry.load_dataset(
        name="synthetic_stream",
        source_train_ratio=0.75,
        window_size=400,
        n_source_samples=2500,
        seed=42,
    )

    pipeline = train_and_freeze_pipeline(
        X_train=bundle.X_source_train,
        y_train=bundle.y_source_train,
        X_cal=bundle.X_source_cal,
        y_cal=bundle.y_source_cal,
        feature_names=bundle.feature_names,
        seed=42,
    )
    print(f"[+] Models trained on source data:")
    print(f"    - Primary Model: Calibrated XGBoost Decision Tree")
    print(f"    - Secondary Probe: Deep Neural Network (MLP) with Variational Bayesian Last Layer")
    print(f"    - Source Calibration Baseline Accuracy: {pipeline.source_cal_acc * 100:.2f}%")

    # Initialize TrustGuard Composite Monitor
    tg_monitor = TrustGuardCompositeMonitor(
        frozen_pipeline=pipeline,
        X_cal=bundle.X_source_cal,
        alpha_alarm=0.05,
    )
    print(f"    - TrustGuard Alarm Threshold: {tg_monitor.alarm_threshold:.4f}")

    print("\n" + "=" * 80)
    print("STEP 3: RUNNING OUR MODELS ON THE NEW TEST DATA")
    print("=" * 80)

    features_df = new_data[[c for c in new_data.columns if c.startswith("x_")]]
    y_true = new_data["true_label"].to_numpy(dtype=int)

    # 1. Primary Model predictions
    probs_primary = pipeline.predict_primary_proba(features_df)
    preds_primary = (probs_primary >= 0.5).astype(int)

    # 2. Secondary Neural MLP predictions
    probs_secondary = pipeline.predict_secondary_proba(features_df)
    preds_secondary = (probs_secondary >= 0.5).astype(int)

    # 3. Conformal prediction sets
    conformal_sets = pipeline.conformal_calibrator.predict_prediction_sets(probs_primary)
    set_strings = [str(s) for s in conformal_sets]
    set_sizes = [len(s) for s in conformal_sets]

    # Save detailed prediction outputs
    output_df = new_data.copy()
    output_df["pred_primary_label"] = preds_primary
    output_df["pred_primary_prob"] = np.round(probs_primary, 4)
    output_df["pred_secondary_label"] = preds_secondary
    output_df["pred_secondary_prob"] = np.round(probs_secondary, 4)
    output_df["models_agree"] = preds_primary == preds_secondary
    output_df["conformal_set"] = set_strings
    output_df["conformal_set_size"] = set_sizes
    output_df["primary_correct"] = preds_primary == y_true

    output_df.to_csv(predictions_path, index=False)
    print(f"[+] Model predictions saved to: {predictions_path}")

    # 4. Overall Classifier Performance Summary
    print("\n" + "-" * 60)
    print("CLASSIFIER PERFORMANCE ON NEW TEST DATA (Across all 1,200 samples)")
    print("-" * 60)
    acc_primary = accuracy_score(y_true, preds_primary)
    f1_primary = f1_score(y_true, preds_primary)
    roc_primary = roc_auc_score(y_true, probs_primary)

    acc_secondary = accuracy_score(y_true, preds_secondary)
    f1_secondary = f1_score(y_true, preds_secondary)
    roc_secondary = roc_auc_score(y_true, probs_secondary)

    print(f"Primary Tree (XGBoost):")
    print(f"   Accuracy: {acc_primary * 100:.2f}% | F1-Score: {f1_primary:.4f} | ROC-AUC: {roc_primary:.4f}")
    print(f"Secondary Neural Probe (MLP):")
    print(f"   Accuracy: {acc_secondary * 100:.2f}% | F1-Score: {f1_secondary:.4f} | ROC-AUC: {roc_secondary:.4f}")
    print(f"Inter-Model Disagreement Rate: {np.mean(preds_primary != preds_secondary) * 100:.2f}%")

    # 5. Window-by-Window TrustGuard Monitoring Breakdown
    print("\n" + "=" * 80)
    print("STEP 4: TRUSTGUARD POST-DEPLOYMENT MONITORING OUTPUT BY BATCH")
    print("=" * 80)

    regimes = ["in_distribution", "harmless_noise", "concept_drift"]
    summary_rows = []

    for reg in regimes:
        sub_mask = new_data["regime"] == reg
        sub_X = features_df[sub_mask].reset_index(drop=True)
        sub_y = y_true[sub_mask]

        # Run TrustGuard evaluation
        tg_out = tg_monitor.evaluate_window(sub_X)

        sub_preds = preds_primary[sub_mask]
        true_acc = accuracy_score(sub_y, sub_preds)

        summary_rows.append({
            "Batch / Regime": reg,
            "Samples": len(sub_X),
            "True Accuracy": f"{true_acc * 100:.1f}%",
            "S_attr (Drift)": f"{tg_out['s_attr']:.3f}",
            "S_dis (Disagreement)": f"{tg_out['s_dis']:.3f}",
            "S_conf (Ambiguity)": f"{tg_out['s_conf']:.3f}",
            "S_grad (Gradient)": f"{tg_out['s_grad']:.3f}",
            "TrustGuard Risk": f"{tg_out['risk_tg']:.4f}",
            "Alarm Threshold": f"{tg_monitor.alarm_threshold:.4f}",
            "Verdict": "🚨 ALARM" if tg_out["is_alarm"] else "✅ CLEAR",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
