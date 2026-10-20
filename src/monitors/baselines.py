"""Baseline label-free estimators and input drift detectors for SilentX.

Covers:
1. Univariate Input Drift: Kolmogorov-Smirnov (KS_max), Population Stability Index (PSI_max),
   and Jensen-Shannon Divergence (JSD).
2. Softmax Confidence Proxies:
   - Average Confidence (AC / MC)
   - Difference of Confidences (DoC)
   - Average Thresholded Confidence (ATC)
   - Confidence-Based Performance Estimation (CBPE)
   - Confidence Optimal Transport with Thresholding (COTT stub interface)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


class KSMaxDriftDetector:
    """Maximum univariate Kolmogorov-Smirnov input drift detector."""

    def __init__(self, X_cal: pd.DataFrame):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.feature_names = list(X_cal.columns)

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> float:
        """Compute KS_max between source calibration and unlabeled target window."""
        X_t = X_unlabeled.to_numpy(dtype=float)
        n_features = X_t.shape[1]
        ks_stats = []
        for j in range(n_features):
            res = ks_2samp(self.X_cal[:, j], X_t[:, j])
            ks_stats.append(res.statistic)
        return float(np.max(ks_stats))


class JensenShannonDivergenceDetector:
    """Univariate Jensen-Shannon Divergence detector using empirical histograms.

    Note: Computes true JSD (square of Jensen-Shannon distance, bounded in [0, 1] using base 2).
    """

    def __init__(self, X_cal: pd.DataFrame, n_bins: int = 10):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.n_bins = n_bins

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> float:
        X_t = X_unlabeled.to_numpy(dtype=float)
        n_features = X_t.shape[1]
        jsd_values = []

        for j in range(n_features):
            # Compute common bin edges
            min_val = min(np.min(self.X_cal[:, j]), np.min(X_t[:, j]))
            max_val = max(np.max(self.X_cal[:, j]), np.max(X_t[:, j]))
            if min_val == max_val:
                jsd_values.append(0.0)
                continue

            bins = np.linspace(min_val, max_val + 1e-6, self.n_bins + 1)
            hist_cal, _ = np.histogram(self.X_cal[:, j], bins=bins, density=True)
            hist_t, _ = np.histogram(X_t[:, j], bins=bins, density=True)

            p = hist_cal / np.sum(hist_cal)
            q = hist_t / np.sum(hist_t)

            # jensenshannon returns distance; square it to obtain divergence
            js_dist = jensenshannon(p, q, base=2.0)
            jsd_values.append(float(js_dist**2))

        return float(np.max(jsd_values))


class AverageConfidenceEstimator:
    """Average Confidence with Maximum Confidence (AC-MC) accuracy estimator."""

    @staticmethod
    def estimate_accuracy(predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float).ravel()
        conf = np.maximum(p, 1.0 - p)
        return float(np.mean(conf))


class DifferenceOfConfidencesEstimator:
    """Difference of Confidences (DoC) accuracy estimator (Guillory et al., 2021).

    Acc_DoC(W_t) = Acc(D_cal) - (mean(Conf(D_cal)) - mean(Conf(W_t)))
    """

    def __init__(self, source_cal_acc: float, source_cal_mean_conf: float):
        self.source_cal_acc = float(source_cal_acc)
        self.source_cal_mean_conf = float(source_cal_mean_conf)

    def estimate_accuracy(self, predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float).ravel()
        conf_t = np.mean(np.maximum(p, 1.0 - p))
        est = self.source_cal_acc - (self.source_cal_mean_conf - conf_t)
        return float(np.clip(est, 0.0, 1.0))


class AverageThresholdedConfidenceEstimator:
    """Average Thresholded Confidence (ATC) accuracy estimator (Garg et al., 2022).

    Learns threshold t* such that mean(Conf(D_cal) < t*) = 1 - Acc(D_cal).
    Acc_ATC(W_t) = (1/m) sum I(Conf(x_i) >= t*).
    """

    def __init__(self, atc_threshold: float):
        self.atc_threshold = float(atc_threshold)

    def estimate_accuracy(self, predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float).ravel()
        conf_t = np.maximum(p, 1.0 - p)
        return float(np.mean(conf_t >= self.atc_threshold))


class CBPEPerformanceEstimator:
    """Confidence-Based Performance Estimation (NannyML CBPE formulation).

    Reconstructs expected accuracy from point-level calibrated posterior probabilities:
    P(y_hat = y)_i = 1 - |y_hat_i - p_i|.
    """

    @staticmethod
    def estimate_accuracy(predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float).ravel()
        pseudo_y = (p >= 0.5).astype(float)
        # Expected accuracy per instance
        p_correct = 1.0 - np.abs(pseudo_y - p)
        return float(np.mean(p_correct))
