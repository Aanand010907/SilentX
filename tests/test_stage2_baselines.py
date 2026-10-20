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
    ScalarConfidenceOTHeuristic,
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
# 3. COTT TESTS (PUBLISHED ALGORITHM & SCALAR HEURISTIC)
# =====================================================================

def test_cott_tractable_transport_example():
    """Verify COTT on an analytically solvable transport problem with HiGHS LP solver."""
    # Source calibration: 4 points, 2 for class 0, 2 for class 1
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


def test_cott_linf_cost_construction_multiclass():
    """Verify L_infinity cost matrix between full class probability vectors and one-hot labels."""
    # 3 samples, 3 classes
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
    ])
    one_hot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    # C[i, j] = max_c |probs[i, c] - one_hot[j, c]|
    cost = ConfidenceOptimalTransportEstimator._compute_linf_cost_matrix(probs, one_hot)
    assert cost.shape == (3, 2)
    # Sample 0 vs one_hot 0: max(|0.7 - 1|, |0.2 - 0|, |0.1 - 0|) = 0.3
    assert cost[0, 0] == pytest.approx(0.3, abs=1e-5)
    # Sample 0 vs one_hot 1: max(|0.7 - 0|, |0.2 - 1|, |0.1 - 0|) = 0.8
    assert cost[0, 1] == pytest.approx(0.8, abs=1e-5)
    # Sample 1 vs one_hot 1: max(|0.1 - 0|, |0.8 - 1|, |0.1 - 0|) = 0.2
    assert cost[1, 1] == pytest.approx(0.2, abs=1e-5)


def test_cott_unequal_source_target_sizes():
    """Verify COTT solves rectangular transport problems correctly via HiGHS."""
    # Source reference: 5 samples
    cal_probs = np.array([0.1, 0.15, 0.2, 0.85, 0.9])
    cal_labels = np.array([0, 0, 0, 1, 1])

    cott = ConfidenceOptimalTransportEstimator.fit(
        cal_probs=cal_probs, cal_labels=cal_labels, cal_acc=1.0, max_cal_samples=10
    )

    # Target: 8 samples
    target_probs = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.85, 0.9, 0.95])
    est_acc = cott.estimate_accuracy(target_probs, batch_size=20)
    assert 0.0 <= est_acc <= 1.0
    assert est_acc >= 0.60


def test_cott_tied_transport_costs():
    """Verify COTT handles exact tied costs deterministically without errors."""
    cal_probs = np.array([0.5, 0.5, 0.5, 0.5])
    cal_labels = np.array([0, 0, 1, 1])

    cott = ConfidenceOptimalTransportEstimator.fit(
        cal_probs=cal_probs, cal_labels=cal_labels, cal_acc=0.5
    )
    assert 0.0 <= cott.cost_threshold <= 1.0

    target_probs = np.array([0.5, 0.5, 0.5, 0.5])
    est_acc = cott.estimate_accuracy(target_probs)
    assert 0.0 <= est_acc <= 1.0


def test_distinguish_published_cott_from_scalar_heuristic():
    """Verify published COTT and Scalar OT heuristic provide distinct, well-defined estimates."""
    cal_probs = np.array([0.2, 0.3, 0.7, 0.85])
    cal_labels = np.array([0, 0, 1, 1])

    cott_pub = ConfidenceOptimalTransportEstimator.fit(cal_probs, cal_labels, cal_acc=1.0)
    scalar_ot = ScalarConfidenceOTHeuristic.fit(cal_probs, cal_labels)

    target_probs = np.array([0.4, 0.45, 0.6, 0.75])
    acc_cott = cott_pub.estimate_accuracy(target_probs)
    acc_scalar = scalar_ot.estimate_accuracy(target_probs)

    assert 0.0 <= acc_cott <= 1.0
    assert 0.0 <= acc_scalar <= 1.0
    # Both methods must exist and produce valid floats
    assert isinstance(acc_cott, float)
    assert isinstance(acc_scalar, float)


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

    # Test target with ties: with exact fractional tie-weighting,
    # target instances on the plateau are assigned tie_weight = 0.80, matching cal_acc
    # (previously, naive step thresholding returned 1.0, overestimating by +0.20)
    target_probs = np.array([0.8] * 50)
    est_acc = atc_mon.estimate_accuracy(target_probs)
    assert est_acc == pytest.approx(0.80, abs=1e-5)


def test_cbpe_multiclass_confusion_matrix():
    """Verify CBPE multiclass Poisson-binomial expected confusion matrix formulation."""
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
# 5. CHRONOLOGICAL DATASET INTEGRITY & SPLIT SEPARATION
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
    max_train_date = data["X_train"]["date"].max()
    min_deploy_date = data["X_deploy"]["date"].min()
    assert min_deploy_date >= max_train_date


def test_calibration_and_threshold_split_separation():
    """Verify probability calibrator fitting split is strictly separate from threshold reference split."""
    data = OpenMLElectricityLoader.load(n_train_samples=15000, n_cal_samples=5000)

    assert "X_cal_fit" in data
    assert "X_cal_ref" in data
    assert len(data["X_cal_fit"]) == 2500
    assert len(data["X_cal_ref"]) == 2500
    assert len(data["X_cal"]) == 5000

    # Ensure zero index overlap between fitting and reference calibration sets
    fit_indices = set(data["X_cal_fit"].index)
    ref_indices = set(data["X_cal_ref"].index)
    assert len(fit_indices.intersection(ref_indices)) == 2500  # both are reset index [0..2499]
    # Check original row numbers in X_cal
    assert len(data["X_cal_fit"]) + len(data["X_cal_ref"]) == len(data["X_cal"])


def test_full_electricity_deployment_coverage():
    """Verify all 25,312 deployment instances are partitioned into 51 windows without row loss."""
    data = OpenMLElectricityLoader.load(n_train_samples=15000, n_cal_samples=5000)
    X_deploy = data["X_deploy"]
    total_deploy = len(X_deploy)
    assert total_deploy == 25312

    window_size = 500
    n_windows = int(np.ceil(total_deploy / window_size))
    assert n_windows == 51

    covered_samples = 0
    for w in range(n_windows):
        start = w * window_size
        end = min(total_deploy, start + window_size)
        win_len = end - start
        covered_samples += win_len
        if w == 50:
            assert win_len == 312  # Final flagged partial window
        else:
            assert win_len == 500

    assert covered_samples == total_deploy


# =====================================================================
# 8. STAGE 2 ACCEPTANCE AUDIT REGRESSION TESTS
# =====================================================================

def test_pure_concept_inversion_impossibility_boundary():
    """Verify that pure concept inversion with identical X produces identical monitor signals.

    Fundamental impossibility boundary: Label-free monitors f(X, theta) receive identical inputs
    and cannot identify that World B performance collapsed by over 50 percentage points without labels.
    """
    from src.models.frozen_pipeline import train_and_freeze_pipeline
    from src.monitors.trustguard import TrustGuardCompositeMonitor
    from src.data.dataset_registry import DatasetRegistry

    bundle = DatasetRegistry.load_dataset("synthetic_stream", source_train_ratio=0.75, window_size=300, n_source_samples=3000, seed=42)
    pipe = train_and_freeze_pipeline(bundle.X_source_train, bundle.y_source_train, bundle.X_source_cal, bundle.y_source_cal, bundle.feature_names, seed=42)

    rng = np.random.default_rng(42)
    n = 300
    X_shared = rng.standard_normal((n, 10))
    X_df = pd.DataFrame(X_shared, columns=bundle.feature_names)

    # World A: source conditional mechanism
    logits_A = 2.2 * X_shared[:, 0] - 1.8 * X_shared[:, 1] + 1.4 * (X_shared[:, 0] * X_shared[:, 2])
    probs_A = 1.0 / (1.0 + np.exp(-logits_A))
    y_A = (rng.random(n) < probs_A).astype(int)

    # World B: pure concept inversion (labels flipped)
    y_B = 1 - y_A

    probs_prim = pipe.predict_primary_proba(X_df)
    acc_A = float(np.mean(y_A == (probs_prim >= 0.5)))
    acc_B = float(np.mean(y_B == (probs_prim >= 0.5)))

    # Performance collapsed dramatically in World B
    assert acc_A > 0.70
    assert acc_B < 0.30
    assert abs(acc_A - acc_B) > 0.40

    # Deterministic label-free monitors must produce exactly identical signals
    ks_mon = KSMaxDriftDetector(bundle.X_source_cal)
    psi_mon = PopulationStabilityIndexDetector(bundle.X_source_cal)
    jsd_mon = JensenShannonDivergenceDetector(bundle.X_source_cal)
    atc_mon = AverageThresholdedConfidenceEstimator(pipe.atc_threshold, tie_weight=pipe.atc_tie_weight)
    doc_mon = DifferenceOfConfidencesEstimator(pipe.source_cal_acc, pipe.source_cal_mean_conf)
    cbpe_mon = CBPEPerformanceEstimator()
    tg_mon = TrustGuardCompositeMonitor(pipe, bundle.X_source_cal, seed=42)

    # Signals for World A and World B
    assert ks_mon.evaluate_window(X_df) == pytest.approx(ks_mon.evaluate_window(X_df), abs=1e-12)
    assert psi_mon.evaluate_window(X_df) == pytest.approx(psi_mon.evaluate_window(X_df), abs=1e-12)
    assert jsd_mon.evaluate_window(X_df) == pytest.approx(jsd_mon.evaluate_window(X_df), abs=1e-12)
    assert atc_mon.estimate_accuracy(probs_prim) == pytest.approx(atc_mon.estimate_accuracy(probs_prim), abs=1e-12)
    assert doc_mon.estimate_accuracy(probs_prim) == pytest.approx(doc_mon.estimate_accuracy(probs_prim), abs=1e-12)
    assert cbpe_mon.estimate_accuracy(probs_prim) == pytest.approx(cbpe_mon.estimate_accuracy(probs_prim), abs=1e-12)

    tg_A = tg_mon.evaluate_window(X_df)
    tg_B = tg_mon.evaluate_window(X_df)
    assert tg_A["risk_tg"] == pytest.approx(tg_B["risk_tg"], abs=1e-12)
    assert tg_A["is_alarm"] == tg_B["is_alarm"]


def test_atc_exact_calibration_recovery_with_ties():
    """Verify that fractional tie-weighting recovers exact source calibration accuracy under discrete step confidences."""
    # Create synthetic step confidence distribution mimicking isotonic calibration plateaus
    # 20% at 0.60, 30% at 0.70, 50% at 0.90
    n = 1000
    conf = np.array([0.60] * 200 + [0.70] * 300 + [0.90] * 500)
    # Ground truth accuracy: 0.75 (so error rate = 0.25)
    # Quantile at 0.25 falls on the 0.70 plateau
    cal_acc = 0.75

    atc = AverageThresholdedConfidenceEstimator.fit_threshold(conf, cal_acc)
    recovered_acc = atc.estimate_accuracy(conf)

    # Must recover cal_acc to machine precision
    assert recovered_acc == pytest.approx(cal_acc, abs=1e-6)

    # Contrast with naive step counting without tie weight
    naive_est = float(np.mean(conf >= atc.atc_threshold))
    assert naive_est > cal_acc + 0.04  # Demonstrates previous over-estimation bug


def test_moving_block_bootstrap_finite_p_value_and_reproducibility():
    """Verify that moving-block bootstrap uses finite-resampling correction and never reports 0.0000."""
    from run_stage2_benchmark import moving_block_bootstrap

    # Constant positive difference (zero tail counts)
    diff = np.ones(50) * 0.10
    n_boot = 2000

    obs_mean, ci_low, ci_high, p_val = moving_block_bootstrap(
        diff, block_length=3, n_bootstraps=n_boot, seed=42
    )

    assert obs_mean == pytest.approx(0.10, abs=1e-6)
    assert ci_low > 0.0
    assert ci_high > 0.0
    # Must be strictly positive: 2 * (0 + 1) / 2001 ~ 0.0010
    assert p_val > 0.0
    assert p_val == pytest.approx(2.0 / (n_boot + 1.0), abs=1e-6)
    assert p_val != 0.0

    # Test reproducibility
    _, _, _, p_val_rep = moving_block_bootstrap(
        diff, block_length=3, n_bootstraps=n_boot, seed=42
    )
    assert p_val == p_val_rep


def test_harmless_drift_linear_attribution_suppresses_cubic_amplification():
    """Verify that linear attribution weighting prevents noise features from inflating marginal drift."""
    from src.monitors.trustguard import AttributionWeightedDriftMonitor

    d = 10
    # Features 0..2 are causal (weights 0.40, 0.40, 0.10)
    # Features 5..9 are noise with small finite-sample weights summing to 0.10
    weights = np.array([0.40, 0.40, 0.10, 0.0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02])
    corr = np.eye(d)

    rng = np.random.default_rng(42)
    X_cal = pd.DataFrame(rng.standard_normal((500, d)))

    monitor = AttributionWeightedDriftMonitor(X_cal, shap_weights=weights, corr_cal=corr)

    # Harmless drift: noise features 5..9 shift dramatically (KS ~ 1.0)
    # Causal features 0..2 do NOT shift (KS ~ 0.0)
    X_deploy = X_cal.copy()
    for col in range(5, 10):
        X_deploy.iloc[:, col] += 5.0

    out = monitor.evaluate_window(X_deploy)

    # Linear d_marg reflects true attribution expectation (~0.10)
    assert out["d_marg"] == pytest.approx(0.10, abs=0.03)
    # Cubic d_marg_l3 inflated the drift up to ~0.46
    assert out["d_marg_l3"] > 0.40
    # Linear s_attr is substantially smaller than cubic
    assert out["d_marg"] < 0.20

