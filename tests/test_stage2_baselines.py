"""Stage 2 behavioral tests for expanded baselines, PSI, JSD, COTT, and temporal evaluation."""

import pytest
import numpy as np
import pandas as pd

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
from src.data.real_datasets import OpenMLElectricityLoader
from src.evaluation.oracle import EvaluationOracle


# =====================================================================
# 1. PSI TESTS
# =====================================================================

def test_psi_identical_distributions():
    """PSI must be approximately 0.0 when source and target distributions are identical."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 3))
    df_cal = pd.DataFrame(data, columns=["a", "b", "c"])
    df_t = pd.DataFrame(data.copy(), columns=["a", "b", "c"])

    psi_det = PopulationStabilityIndexDetector(df_cal, n_bins=10)
    psi_val = psi_det.evaluate_window(df_t)

    assert psi_val == pytest.approx(0.0, abs=1e-4)


def test_psi_shifted_distributions_and_zero_count_bins():
    """PSI must be significantly > 0.25 when distributions shift, handling zero-count bins."""
    rng = np.random.default_rng(42)
    df_cal = pd.DataFrame(rng.standard_normal((500, 2)), columns=["x1", "x2"])
    # Shifted: mean shift by +3 standard deviations
    df_shifted = pd.DataFrame(rng.standard_normal((500, 2)) + 3.0, columns=["x1", "x2"])

    psi_det = PopulationStabilityIndexDetector(df_cal, n_bins=10)
    psi_val = psi_det.evaluate_window(df_shifted)

    # Standard industry rule: PSI > 0.25 indicates significant shift
    assert psi_val > 0.50


# =====================================================================
# 2. JSD TESTS
# =====================================================================

def test_jsd_mathematical_conventions_and_boundaries():
    """Verify JSD returns true divergence in [0, 1] using base 2 log, 0 for identical, and handles constants."""
    df_cal = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "const": [10.0, 10.0, 10.0, 10.0, 10.0]})
    df_same = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "const": [10.0, 10.0, 10.0, 10.0, 10.0]})

    jsd_det = JensenShannonDivergenceDetector(df_cal, n_bins=5)
    jsd_identical = jsd_det.evaluate_window(df_same)
    assert jsd_identical == pytest.approx(0.0, abs=1e-6)

    # Completely disjoint distributions
    df_disjoint = pd.DataFrame({"a": [200.0, 300.0, 400.0, 500.0, 600.0], "const": [10.0, 10.0, 10.0, 10.0, 10.0]})
    jsd_disjoint = jsd_det.evaluate_window(df_disjoint)
    # For completely disjoint non-overlapping distributions, JSD (base 2) is 1.0
    assert jsd_disjoint == pytest.approx(1.0, abs=1e-5)


# =====================================================================
# 3. COTT TESTS
# =====================================================================

def test_cott_tractable_transport_example():
    """Verify COTT on an analytically solvable transport problem with HiGHS LP solver."""
    # Source calibration: 4 points, 2 for class 0, 2 for class 1
    # Points with ground-truth one-hot vectors
    cal_probs = np.array([0.1, 0.2, 0.8, 0.9])
    cal_labels = np.array([0, 0, 1, 1])
    cal_acc = 1.0

    cott = ConfidenceOptimalTransportEstimator.fit(
        cal_probs=cal_probs, cal_labels=cal_labels, cal_acc=cal_acc
    )

    # Clean target with matching high-confidence predictions
    target_clean = np.array([0.1, 0.2, 0.8, 0.9])
    est_acc_clean = cott.estimate_accuracy(target_clean)
    assert est_acc_clean >= 0.75

    # Completely inverted target probabilities
    target_inverted = np.array([0.9, 0.9, 0.1, 0.1])
    est_acc_inv = cott.estimate_accuracy(target_inverted)
    assert 0.0 <= est_acc_inv <= 1.0


# =====================================================================
# 4. CONFIDENCE ESTIMATORS AND ATC TIE-BREAKING
# =====================================================================

def test_atc_tie_breaking_behavior():
    """ATC must handle exact confidence ties without crashing or generating invalid thresholds."""
    # Uniform tied confidence of exactly 0.8 across all samples
    cal_probs = np.array([0.8] * 100)
    cal_acc = 0.80

    atc_mon = AverageThresholdedConfidenceEstimator.fit_threshold(cal_probs, cal_acc)
    assert atc_mon.atc_threshold == pytest.approx(0.8, abs=1e-5)

    # Test target with ties
    target_probs = np.array([0.8] * 50)
    est_acc = atc_mon.estimate_accuracy(target_probs)
    # Since confidence (0.8) >= threshold (0.8), fraction is 1.0
    assert est_acc == 1.0


def test_cbpe_multiclass_confusion_matrix():
    """Verify CBPE multiclass Poisson-binomial expected confusion matrix formulation."""
    # 3-class predicted probability simplex matrix
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
    ])

    cm_out = CBPEPerformanceEstimator.estimate_confusion_matrix(probs, num_classes=3)
    assert "overall_accuracy" in cm_out
    assert "macro_f1" in cm_out
    assert len(cm_out["class_metrics"]) == 3
    assert 0.0 <= cm_out["overall_accuracy"] <= 1.0
    assert 0.0 <= cm_out["macro_f1"] <= 1.0


# =====================================================================
# 5. CHRONOLOGICAL DATASET INTEGRITY
# =====================================================================

def test_chronological_electricity_boundaries():
    """Verify chronological dataset partitioning without temporal leakage."""
    data = OpenMLElectricityLoader.load()

    assert data["n_total"] > 0
    assert len(data["X_train"]) > 0
    assert len(data["X_cal"]) > 0
    assert len(data["X_deploy"]) > 0

    total_len = len(data["X_train"]) + len(data["X_cal"]) + len(data["X_deploy"])
    assert total_len == data["n_total"]

    # Verify chronological sequence order via 'date' column
    # The minimum date of deploy should be >= maximum date of train
    max_train_date = data["X_train"]["date"].max()
    min_deploy_date = data["X_deploy"]["date"].min()
    assert min_deploy_date >= max_train_date
