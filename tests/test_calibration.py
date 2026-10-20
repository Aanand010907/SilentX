"""Unit and integration tests for calibration, ECE, and Mondrian conformal prediction."""

import numpy as np
import pytest

from src.models.calibration import (
    compute_expected_calibration_error,
    SourceProbabilityCalibrator,
    MondrianConformalCalibrator,
)
from src.models.vbll_model import D3MNeuralClassifier, D3MReferenceCalibrator


def test_ece_computation_properties():
    """Verify Expected Calibration Error properties on perfectly calibrated vs uncalibrated sets."""
    # Perfectly calibrated case: prob = label
    probs_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    labels_perfect = np.array([0, 0, 1, 1])
    ece_perfect = compute_expected_calibration_error(probs_perfect, labels_perfect, n_bins=5)
    assert ece_perfect == 0.0

    # Perfectly miscalibrated: high confidence in wrong class
    probs_bad = np.array([0.99, 0.99, 0.01, 0.01])
    labels_bad = np.array([0, 0, 1, 1])
    ece_bad = compute_expected_calibration_error(probs_bad, labels_bad, n_bins=5)
    assert ece_bad > 0.90


def test_source_probability_calibrator_ece_decision():
    """Verify that calibrator correctly runs 3-fold CV and stores reduction metrics."""
    rng = np.random.default_rng(42)
    # Generate synthetic probabilities with monotonic distortion
    true_probs = rng.uniform(0.1, 0.9, size=300)
    labels = (rng.random(300) < true_probs).astype(int)
    # Distort raw probs via power transformation (miscalibration)
    raw_probs = true_probs ** 2.5

    cal = SourceProbabilityCalibrator(n_bins=10, n_cv_splits=3, random_state=42)
    cal.fit(raw_probs, labels)

    assert "mean_raw_ece" in cal.cv_results_
    assert "mean_cal_ece" in cal.cv_results_
    assert isinstance(cal.is_calibrated, bool)

    # Transform probabilities
    calibrated_output = cal.predict_proba(raw_probs)
    assert len(calibrated_output) == len(raw_probs)
    assert np.all((calibrated_output >= 0.0) & (calibrated_output <= 1.0))


def test_mondrian_conformal_finite_sample_quantiles():
    """Verify class-conditional conformal prediction sets and cardinality metrics."""
    rng = np.random.default_rng(42)
    cal_probs = np.concatenate([rng.uniform(0.0, 0.4, size=150), rng.uniform(0.6, 1.0, size=150)])
    cal_labels = np.array([0] * 150 + [1] * 150)

    conf_cal = MondrianConformalCalibrator(alpha=0.10)
    conf_cal.fit(cal_probs, cal_labels)

    assert conf_cal.q_0 is not None
    assert conf_cal.q_1 is not None
    assert 0.0 < conf_cal.q_0 < 1.0
    assert 0.0 < conf_cal.q_1 < 1.0

    # Test prediction sets on clean ID probabilities
    test_probs = np.array([0.05, 0.95, 0.50])
    prediction_sets = conf_cal.predict_prediction_sets(test_probs)
    assert len(prediction_sets) == 3

    # Confident 0 sample should contain [0]
    assert 0 in prediction_sets[0]
    # Confident 1 sample should contain [1]
    assert 1 in prediction_sets[1]

    metrics = conf_cal.compute_cardinality_metrics(test_probs)
    assert "singleton_rate" in metrics
    assert "abstention_rate" in metrics
    assert 0.0 <= metrics["singleton_rate"] <= 1.0
    assert 0.0 <= metrics["abstention_rate"] <= 1.0


def test_d3m_reference_calibration_distribution():
    """Verify D3M bootstrap reference calibrator locks 95th-percentile disagreement thresholds."""
    model = D3MNeuralClassifier(in_features=5, hidden_dims=[16], prior_var=1.0)
    X_cal = np.random.randn(200, 5)

    d3m = D3MReferenceCalibrator(
        query_sizes=[10, 20], num_bootstrap_rounds=20, posterior_samples_k=10, alpha=0.05, seed=42
    )
    d3m.fit(model, X_cal)

    assert 10 in d3m.thresholds_
    assert 20 in d3m.thresholds_
    assert len(d3m.reference_distributions_[10]) == 20
    assert 0.0 <= d3m.thresholds_[10] <= 1.0

    # Evaluate unlabeled test batch
    X_test = np.random.randn(10, 5)
    res = d3m.evaluate_batch(model, X_test, query_size=10)
    assert "disagreement_rate" in res
    assert "is_alert" in res
    assert isinstance(res["is_alert"], bool)
