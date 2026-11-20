"""Benchmark the New Combined Optimal Approach (OmniGuard) against original models on new_data.csv."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss

from src.data.dataset_registry import DatasetRegistry
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.models.hybrid_ensemble import build_robust_hybrid_ensemble_from_pipeline
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.three_tier_guardrail import (
    ThreeTierOperationalGuardrail,
    OperationalVerdict,
)
from src.monitors.baselines import KSMaxDriftDetector


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "new_data.csv")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("RUNNING THE NEW COMBINED OPTIMAL APPROACH (OMNIGUARD BENCHMARK)")
    print("=" * 80)

    # 1. Load fresh test data
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("x_")]
    X_test = df[feature_cols]
    y_test = df["true_label"].to_numpy(dtype=int)
    regimes = df["regime"].to_numpy()

    # 2. Train and assemble the new Combined Architecture
    print("[*] Training source base models & assembling GBDT + Bayesian Hybrid Ensemble...")
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

    hybrid_ensemble = build_robust_hybrid_ensemble_from_pipeline(
        pipeline=pipeline,
        X_train=bundle.X_source_train,
        y_train=bundle.y_source_train,
        seed=42,
    )

    tg_monitor = TrustGuardCompositeMonitor(
        frozen_pipeline=pipeline,
        X_cal=bundle.X_source_cal,
        alpha_alarm=0.05,
    )

    guardrail = ThreeTierOperationalGuardrail(
        hybrid_ensemble=hybrid_ensemble,
        trustguard_monitor=tg_monitor,
        input_drift_threshold=0.20,
    )
    ks_detector = KSMaxDriftDetector(bundle.X_source_cal)

    print(f"[+] Optimal System Initialized:")
    print(f"    - Multi-Tree GBDT Ensemble (HGB + XGB + RF): Loaded")
    print(f"    - Variational Bayesian Last Layer Probe: Loaded")
    print(f"    - Dynamic SHAP Attribution Gating: Active")
    print(f"    - Three-Tier Operational Guardrail: Active")

    # 3. Evaluate by regime batches
    all_omni_probs = []
    all_omni_sets = []
    batch_records = []

    unique_regimes = ["in_distribution", "harmless_noise", "concept_drift"]

    for b_idx, reg in enumerate(unique_regimes):
        mask = regimes == reg
        sub_X = X_test[mask].reset_index(drop=True)
        sub_y = y_test[mask]

        ks_val = ks_detector.evaluate_window(sub_X)
        record, batch_probs, batch_sets = guardrail.evaluate_batch(
            X_batch=sub_X,
            batch_id=b_idx,
            ks_max=ks_val,
        )

        all_omni_probs.extend(batch_probs)
        all_omni_sets.extend(batch_sets)

        sub_preds = (batch_probs >= 0.5).astype(int)
        acc = accuracy_score(sub_y, sub_preds)

        batch_records.append({
            "Batch / Regime": reg,
            "Samples": len(sub_X),
            "Omni Accuracy": f"{acc * 100:.2f}%",
            "Verdict": record.verdict.value,
            "Risk_TG": f"{record.risk_tg:.4f}",
            "S_attr": f"{record.signals['s_attr']:.3f}",
            "S_dis": f"{record.signals['s_dis']:.3f}",
            "Abstention Rate": f"{record.tier3_abstention_rate * 100:.1f}%",
            "Escalated to Humans": record.tier3_escalated_count,
        })

    omni_probs_arr = np.array(all_omni_probs)
    omni_preds_arr = (omni_probs_arr >= 0.5).astype(int)

    # 4. Load friend predictions for comparison
    friend_csv = os.path.join(base_dir, "predictions.csv")
    f_df = pd.read_csv(friend_csv)
    fr_preds = f_df["pred_label"].to_numpy()
    fr_probs = f_df["prob_class_1"].to_numpy()

    # Original model predictions
    orig_csv = os.path.join(base_dir, "new_data_predictions.csv")
    o_df = pd.read_csv(orig_csv)
    orig_preds = o_df["pred_primary_label"].to_numpy()
    orig_probs = o_df["pred_primary_prob"].to_numpy()

    # 5. Build Comprehensive Scoreboard
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARATIVE SCOREBOARD (Across All 1,200 Samples)")
    print("=" * 80)

    competitors = {
        "1. Original SilentX (Primary Tree)": (orig_preds, orig_probs),
        "2. Friend's Approach (Candidate B)": (fr_preds, fr_probs),
        "3. NEW OPTIMAL APPROACH (OmniGuard)": (omni_preds_arr, omni_probs_arr),
    }

    comp_rows = []
    for name, (p_lbl, p_prb) in competitors.items():
        acc = accuracy_score(y_test, p_lbl)
        f1 = f1_score(y_test, p_lbl, zero_division=0)
        auc = roc_auc_score(y_test, p_prb)
        brier = brier_score_loss(y_test, p_prb)
        loss = log_loss(y_test, np.clip(p_prb, 1e-6, 1 - 1e-6))

        comp_rows.append({
            "System": name,
            "Overall Accuracy (%)": f"{acc * 100:.2f}%",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{auc:.4f}",
            "Brier Score (Lower=Better)": f"{brier:.4f}",
            "Log Loss (Lower=Better)": f"{loss:.4f}",
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # 6. Regime Breakdown
    print("\n" + "=" * 80)
    print("ACCURACY BY REGIME COMPARISON")
    print("=" * 80)

    reg_rows = []
    for reg in unique_regimes:
        mask = regimes == reg
        sub_y = y_test[mask]
        reg_rows.append({
            "Regime": reg,
            "Original SilentX": f"{accuracy_score(sub_y, orig_preds[mask]) * 100:.2f}%",
            "Friend's Approach": f"{accuracy_score(sub_y, fr_preds[mask]) * 100:.2f}%",
            "NEW OPTIMAL APPROACH": f"{accuracy_score(sub_y, omni_preds_arr[mask]) * 100:.2f}%",
        })

    reg_df = pd.DataFrame(reg_rows)
    print(reg_df.to_string(index=False))

    # 7. Operational Decisions Breakdown
    print("\n" + "=" * 80)
    print("THREE-TIER OPERATIONAL GUARDRAIL AUDIT TRAIL")
    print("=" * 80)
    batch_df = pd.DataFrame(batch_records)
    print(batch_df.to_string(index=False))

    # 8. Human Analyst Queue
    queue_summary = guardrail.escalation_queue.get_summary()
    print("\n" + "=" * 80)
    print("TIER 3 HUMAN ANALYST ESCALATION QUEUE SUMMARY")
    print("=" * 80)
    print(f"Total Ambiguous Samples Escalated: {queue_summary['total_escalated']}")
    print(f"Severe Out-of-Distribution Cases: {queue_summary['severe_ood_count']}")
    print(f"Multi-label Ambiguous Cases: {queue_summary['multi_label_count']}")
    print("=" * 80)

    # Save artifacts
    summary_path = os.path.join(results_dir, "omni_benchmark_summary.csv")
    comp_df.to_csv(summary_path, index=False)
    
    # Save combined predictions
    output_pred_path = os.path.join(base_dir, "omni_optimal_predictions.csv")
    df_out = df.copy()
    df_out["omni_pred_label"] = omni_preds_arr
    df_out["omni_prob"] = np.round(omni_probs_arr, 4)
    df_out["omni_conformal_set"] = [str(s) for s in all_omni_sets]
    df_out.to_csv(output_pred_path, index=False)
    print(f"\n[+] Detailed predictions saved to: {output_pred_path}")
    print(f"[+] Summary report saved to: {summary_path}")


if __name__ == "__main__":
    main()
