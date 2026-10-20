"""Unit and integration tests for strict ground-truth isolation and oracle evaluation."""

import pytest
import numpy as np
import pandas as pd

from src.data.dataset_registry import DatasetRegistry, DeploymentWindow
from src.evaluation.oracle import EvaluationOracle, WindowEvaluationResult
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.baselines import (
    AverageConfidenceEstimator,
    AverageThresholdedConfidenceEstimator,
    CBPEPerformanceEstimator,
)
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.data.synthetic_streams import generate_synthetic_stream


def test_monitor_interface_prohibits_label_input():
    """Verify that monitor evaluate_window signature accepts only unlabeled feature dataframes."""
    X_train, y_train, _ = generate_synthetic_stream(n_samples=400, seed=42)
    X_cal, y_cal, _ = generate_synthetic_stream(n_samples=200, seed=43)

    pipeline = train_and_freeze_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        feature_names=list(X_train.columns),
        seed=42,
    )

    monitor = TrustGuardCompositeMonitor(frozen_pipeline=pipeline, X_cal=X_cal)

    # Deployment window with sequestered ground truth
    X_deploy, y_deploy, _ = generate_synthetic_stream(n_samples=100, seed=44)

    # Monitor receives ONLY X_deploy (unlabeled)
    result = monitor.evaluate_window(X_deploy)

    assert "risk_tg" in result
    assert "probs_primary" in result
    # Result must not contain ground truth labels
    assert "target" not in result
    assert "y" not in result
    assert "y_true" not in result


def test_deployment_window_unlabeled_features_view():
    """Verify that DeploymentWindow provides a safe unlabeled view without target columns."""
    X = pd.DataFrame({"feat_1": [1, 2], "feat_2": [3, 4]})
    y = pd.Series([0, 1], name="target")

    window = DeploymentWindow(
        window_id=0, regime="in_distribution", severity=0.0, X=X, y=y, metadata={}
    )

    unlabeled = window.unlabeled_features
    assert "target" not in unlabeled.columns
    assert "feat_1" in unlabeled.columns
    assert "feat_2" in unlabeled.columns


def test_evaluation_oracle_computes_exact_metrics_and_handles_edge_cases():
    """Verify that Oracle correctly scores windows and handles edge cases without fabricating results."""
    oracle = EvaluationOracle(degradation_threshold=0.05, benign_margin=0.02)

    # Scenario: 2 windows
    # Window 0: Clean ID, perfect accuracy
    y_0 = pd.Series([0, 0, 1, 1])
    probs_0 = np.array([0.1, 0.2, 0.8, 0.9])
    res_0 = oracle.score_window(
        window_id=0,
        regime="in_distribution",
        severity=0.0,
        y_sequestered=y_0,
        predicted_probs=probs_0,
        source_cal_acc=1.0,
        monitor_estimated_acc=1.0,
        monitor_risk_score=0.05,
    )

    assert res_0.true_accuracy == 1.0
    assert res_0.true_degradation == 0.0
    assert not res_0.is_degraded
    assert res_0.estimation_error == 0.0

    # Window 1: Degraded concept shift, 0 accuracy
    y_1 = pd.Series([1, 1, 0, 0])  # Inverted
    probs_1 = np.array([0.1, 0.2, 0.8, 0.9])
    res_1 = oracle.score_window(
        window_id=1,
        regime="concept_shift",
        severity=1.0,
        y_sequestered=y_1,
        predicted_probs=probs_1,
        source_cal_acc=1.0,
        monitor_estimated_acc=1.0,  # Blind CBPE estimate
        monitor_risk_score=0.90,    # Model-aware alert
    )

    assert res_1.true_accuracy == 0.0
    assert res_1.true_degradation == 1.0
    assert res_1.is_degraded
    assert res_1.estimation_error == 1.0

    # Evaluate aggregate stream metrics
    stream_eval = oracle.evaluate_stream([res_0, res_1], risk_alarm_threshold=0.5)

    assert stream_eval["n_windows"] == 2
    assert stream_eval["n_degraded_windows"] == 1
    assert stream_eval["n_benign_windows"] == 1
    assert stream_eval["detection_auroc"] == pytest.approx(1.0)
    assert stream_eval["far_benign"] == pytest.approx(0.0)
    assert stream_eval["spearman_rank_correlation"] == pytest.approx(1.0)


def test_oracle_handles_zero_positive_degradation_without_faking_auroc():
    """Verify that when no degradation events occur, AUROC is explicitly NaN, not 1.0."""
    oracle = EvaluationOracle(degradation_threshold=0.05)

    y_seq = pd.Series([0, 1])
    probs = np.array([0.1, 0.9])
    res_0 = oracle.score_window(0, "in_distribution", 0.0, y_seq, probs, 1.0, monitor_risk_score=0.1)
    res_1 = oracle.score_window(1, "in_distribution", 0.0, y_seq, probs, 1.0, monitor_risk_score=0.2)

    stream_eval = oracle.evaluate_stream([res_0, res_1])
    # Mathematical limitation: undefined AUROC must be NaN, never faked!
    assert np.isnan(stream_eval["detection_auroc"])
    assert np.isnan(stream_eval["detection_auprc"])
