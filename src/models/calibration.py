"""Calibration, ECE estimation, and Mondrian conformal prediction sets for SilentX."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width probability bins.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        probabilities: Predicted probabilities for positive class (in [0, 1]).
        labels: Binary ground-truth labels (0 or 1).
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        Scalar ECE in [0, 1].
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(labels, dtype=int).ravel()

    if len(probs) != len(y):
        raise ValueError(f"Length mismatch: probs ({len(probs)}) vs labels ({len(y)})")

    # Bins between 0.0 and 1.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into the bin
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


class SourceProbabilityCalibrator:
    """Evaluates and fits probability calibration strictly on the source calibration split.

    Adheres to the NannyML protocol:
    1. Evaluates raw probabilities vs. 3-fold stratified Isotonic Regression via ECE.
    2. Retains raw probabilities if isotonic calibration does not strictly lower ECE.
    3. Locks all calibrator parameters exclusively on source data.
    """

    def __init__(self, n_bins: int = 10, n_cv_splits: int = 3, random_state: int = 42):
        self.n_bins = n_bins
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.is_calibrated: bool = False
        self.calibrator: Optional[IsotonicRegression] = None
        self.cv_results_: Dict[str, float] = {}

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "SourceProbabilityCalibrator":
        """Evaluate and fit the calibrator on source calibration data."""
        p = np.asarray(raw_probs, dtype=float).ravel()
        y = np.asarray(labels, dtype=int).ravel()

        skf = StratifiedKFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state)
        raw_eces = []
        cal_eces = []

        for train_idx, val_idx in skf.split(p, y):
            p_train, p_val = p[train_idx], p[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit isotonic regression on fold train
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p_train, y_train)

            raw_ece = compute_expected_calibration_error(p_val, y_val, n_bins=self.n_bins)
            cal_ece = compute_expected_calibration_error(iso.predict(p_val), y_val, n_bins=self.n_bins)

            raw_eces.append(raw_ece)
            cal_eces.append(cal_ece)

        mean_raw_ece = float(np.mean(raw_eces))
        mean_cal_ece = float(np.mean(cal_eces))

        self.cv_results_ = {
            "mean_raw_ece": mean_raw_ece,
            "mean_cal_ece": mean_cal_ece,
            "ece_reduction": mean_raw_ece - mean_cal_ece,
        }

        # Apply calibration only if it strictly reduces Expected Calibration Error
        if mean_cal_ece < mean_raw_ece:
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(p, y)
            self.is_calibrated = True
        else:
            self.calibrator = None
            self.is_calibrated = False

        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities using the fitted calibrator (or identity if uncalibrated)."""
        p = np.asarray(raw_probs, dtype=float).ravel()
        if self.is_calibrated and self.calibrator is not None:
            return np.clip(self.calibrator.predict(p), 0.0, 1.0)
        return p


class MondrianConformalCalibrator:
    """Mondrian (class-conditional) split-conformal prediction calibrator.

    Computes class-specific nonconformity quantiles on the source calibration set
    with exact finite-sample correction at nominal coverage level (1 - alpha).

    Conformal quantile formula:
    q_y = Quantile_{ (ceil((N_y + 1)(1 - alpha))) / N_y }( {r_i}_{y_i = y} )
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_0: Optional[float] = None
        self.q_1: Optional[float] = None
        self.n_cal_0: int = 0
        self.n_cal_1: int = 0

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> "MondrianConformalCalibrator":
        """Lock Mondrian nonconformity quantiles on source calibration split."""
        p = np.asarray(cal_probs, dtype=float).ravel()
        y = np.asarray(cal_labels, dtype=int).ravel()

        mask_0 = (y == 0)
        mask_1 = (y == 1)

        self.n_cal_0 = int(np.sum(mask_0))
        self.n_cal_1 = int(np.sum(mask_1))

        if self.n_cal_0 == 0 or self.n_cal_1 == 0:
            raise ValueError(f"Both classes must be present in calibration split. Found 0: {self.n_cal_0}, 1: {self.n_cal_1}")

        # Nonconformity score: r_y(x) is the predicted probability of the *other* class
        # For class 0: nonconformity is p (confidence in 1)
        # For class 1: nonconformity is 1 - p (confidence in 0)
        r_0 = p[mask_0]
        r_1 = 1.0 - p[mask_1]

        # Finite-sample quantile level: ceil((N + 1)(1 - alpha)) / N
        level_0 = min(1.0, np.ceil((self.n_cal_0 + 1) * (1.0 - self.alpha)) / self.n_cal_0)
        level_1 = min(1.0, np.ceil((self.n_cal_1 + 1) * (1.0 - self.alpha)) / self.n_cal_1)

        self.q_0 = float(np.quantile(r_0, level_0, method="higher"))
        self.q_1 = float(np.quantile(r_1, level_1, method="higher"))

        return self

    def predict_prediction_sets(self, probs: np.ndarray) -> List[List[int]]:
        """Compute conformal prediction set C(x_i) for each sample."""
        if self.q_0 is None or self.q_1 is None:
            raise RuntimeError("MondrianConformalCalibrator must be fitted before prediction.")

        p = np.asarray(probs, dtype=float).ravel()
        sets: List[List[int]] = []

        for p_i in p:
            c_set = []
            if p_i <= self.q_0:
                c_set.append(0)
            if (1.0 - p_i) <= self.q_1:
                c_set.append(1)
            sets.append(c_set)

        return sets

    def compute_cardinality_metrics(self, probs: np.ndarray) -> Dict[str, float]:
        """Compute batch-level conformal cardinality and abstention rates."""
        prediction_sets = self.predict_prediction_sets(probs)
        n = len(prediction_sets)
        if n == 0:
            return {"singleton_rate": 0.0, "empty_set_rate": 0.0, "multi_label_rate": 0.0, "abstention_rate": 0.0}

        sizes = np.array([len(s) for s in prediction_sets])
        singleton = np.mean(sizes == 1)
        empty = np.mean(sizes == 0)
        multi = np.mean(sizes > 1)
        abstention = np.mean(sizes != 1)

        return {
            "singleton_rate": float(singleton),
            "empty_set_rate": float(empty),
            "multi_label_rate": float(multi),
            "abstention_rate": float(abstention),
        }
