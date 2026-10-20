"""Comprehensive baseline label-free monitors, drift detectors, and optimal transport estimators.

Covers:
1. Traditional Univariate Drift Baselines:
   - Maximum Kolmogorov-Smirnov Statistic (KS_max)
   - Population Stability Index (PSI_max)
   - Jensen-Shannon Divergence (JSD_max)

2. Confidence-Based Performance Estimators:
   - Average Confidence / Maximum Confidence (AC-MC)
   - Difference of Confidences (DoC)
   - Average Thresholded Confidence (ATC)
   - Confidence-Based Performance Estimation (CBPE) - binary and multiclass Poisson-binomial formulation

3. Confidence Optimal Transport with Thresholding (COTT):
   - Discrete optimal transport via scipy.optimize.linprog (HiGHS solver)
   - L_infinity cost matrix between target probability vectors and source one-hot labels
   - Validation-calibrated cost thresholding
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linprog


# =====================================================================
# 1. TRADITIONAL INPUT DRIFT BASELINES
# =====================================================================

class KSMaxDriftDetector:
    """Maximum univariate two-sample Kolmogorov-Smirnov input drift detector."""

    def __init__(self, X_cal: pd.DataFrame):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.feature_names = list(X_cal.columns)

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> float:
        """Compute KS_max between source calibration and unlabeled target window."""
        X_t = X_unlabeled.to_numpy(dtype=float)
        n_features = X_t.shape[1]
        ks_stats = []
        for j in range(n_features):
            col_cal = self.X_cal[:, j]
            col_t = X_t[:, j]
            # Handle constant features
            if np.all(col_cal == col_cal[0]) and np.all(col_t == col_t[0]):
                ks_stats.append(0.0)
                continue
            res = ks_2samp(col_cal, col_t)
            ks_stats.append(float(res.statistic))
        return float(np.max(ks_stats)) if ks_stats else 0.0


class PopulationStabilityIndexDetector:
    """Population Stability Index (PSI) drift detector across all features.

    Binning strategy:
    - 10 empirical decile bins derived from the source calibration distribution.
    - Smoothing constant epsilon = 1e-4 added to prevent log(0) or division by zero.
    - Formula: PSI_j = sum_{k=1}^B (Q_k - P_k) * ln( (Q_k + eps) / (P_k + eps) )
    - Alarm score: PSI_max = max_j PSI_j.
    """

    def __init__(self, X_cal: pd.DataFrame, n_bins: int = 10, epsilon: float = 1e-4):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.feature_names = list(X_cal.columns)
        self.n_bins = int(n_bins)
        self.epsilon = float(epsilon)
        self.bin_edges_: Dict[int, np.ndarray] = {}
        self.p_expected_: Dict[int, np.ndarray] = {}

        self._fit_bins()

    def _fit_bins(self) -> None:
        n_samples, n_features = self.X_cal.shape
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)

        for j in range(n_features):
            col = self.X_cal[:, j]
            edges = np.quantile(col, quantiles)
            # Remove duplicated edges for non-continuous distributions
            edges = np.unique(edges)
            if len(edges) < 2:
                # Constant feature: create tiny delta
                edges = np.array([col[0] - 1.0, col[0] + 1.0])

            # Ensure endpoints cover any finite value
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[j] = edges

            # Expected proportions on source calibration
            counts, _ = np.histogram(col, bins=edges)
            p = counts / max(1, np.sum(counts))
            self.p_expected_[j] = p

    def evaluate_feature(self, j: int, X_t_col: np.ndarray) -> float:
        """Compute PSI for a single feature column."""
        edges = self.bin_edges_[j]
        p = self.p_expected_[j]
        counts_t, _ = np.histogram(X_t_col, bins=edges)
        q = counts_t / max(1, np.sum(counts_t))

        # Pad q or p if edge uniqueness altered bin counts
        if len(q) != len(p):
            min_len = min(len(q), len(p))
            q = q[:min_len]
            p = p[:min_len]

        eps = self.epsilon
        psi_j = np.sum((q - p) * np.log((q + eps) / (p + eps)))
        return float(max(0.0, psi_j))

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> float:
        """Compute PSI_max across all features."""
        X_t = X_unlabeled.to_numpy(dtype=float)
        n_features = X_t.shape[1]
        psi_scores = []
        for j in range(n_features):
            score = self.evaluate_feature(j, X_t[:, j])
            psi_scores.append(score)
        return float(np.max(psi_scores)) if psi_scores else 0.0


class JensenShannonDivergenceDetector:
    """Univariate Jensen-Shannon Divergence detector using empirical histograms.

    Note on Convention:
    - Logarithm base: base 2.
    - Return value: True Jensen-Shannon DIVERGENCE in [0, 1] (square of JS distance).
    - JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q).
    """

    def __init__(self, X_cal: pd.DataFrame, n_bins: int = 10):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.n_bins = int(n_bins)

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> float:
        X_t = X_unlabeled.to_numpy(dtype=float)
        n_features = X_t.shape[1]
        jsd_values = []

        for j in range(n_features):
            col_cal = self.X_cal[:, j]
            col_t = X_t[:, j]

            min_val = min(np.min(col_cal), np.min(col_t))
            max_val = max(np.max(col_cal), np.max(col_t))
            if min_val == max_val:
                jsd_values.append(0.0)
                continue

            bins = np.linspace(min_val, max_val + 1e-6, self.n_bins + 1)
            hist_cal, _ = np.histogram(col_cal, bins=bins)
            hist_t, _ = np.histogram(col_t, bins=bins)

            p = hist_cal / max(1, np.sum(hist_cal))
            q = hist_t / max(1, np.sum(hist_t))

            # jensenshannon in scipy returns distance; square to obtain divergence
            js_dist = jensenshannon(p, q, base=2.0)
            jsd_values.append(float(js_dist**2))

        return float(np.max(jsd_values)) if jsd_values else 0.0


# =====================================================================
# 2. CONFIDENCE-BASED ACCURACY ESTIMATORS
# =====================================================================

class AverageConfidenceEstimator:
    """Average Confidence with Maximum Confidence (AC-MC) accuracy estimator.

    alpha_hat_AC = (1/n) sum_{i=1}^n max_{c} p_{i, c}
    """

    @staticmethod
    def estimate_accuracy(predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float)
        if p.ndim == 1:
            conf = np.maximum(p, 1.0 - p)
        else:
            conf = np.max(p, axis=1)
        return float(np.mean(conf))


class DifferenceOfConfidencesEstimator:
    """Difference of Confidences (DoC) accuracy estimator (Guillory et al., 2021).

    Acc_DoC(W_t) = Acc(D_cal) - (MeanConf(D_cal) - MeanConf(W_t))
    """

    def __init__(self, source_cal_acc: float, source_cal_mean_conf: float):
        self.source_cal_acc = float(source_cal_acc)
        self.source_cal_mean_conf = float(source_cal_mean_conf)

    def estimate_accuracy(self, predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float)
        if p.ndim == 1:
            conf_t = np.mean(np.maximum(p, 1.0 - p))
        else:
            conf_t = np.mean(np.max(p, axis=1))

        est = self.source_cal_acc - (self.source_cal_mean_conf - conf_t)
        return float(np.clip(est, 0.0, 1.0))


class AverageThresholdedConfidenceEstimator:
    """Average Thresholded Confidence (ATC) accuracy estimator (Garg et al., 2022).

    Learns threshold t* on source calibration set such that:
    mean( Conf(D_cal) < t* ) = 1 - Acc(D_cal)

    Estimated deployment accuracy:
    Acc_ATC(W_t) = (1/m) sum I( Conf(x_i) >= t* )
    """

    def __init__(self, atc_threshold: float):
        self.atc_threshold = float(atc_threshold)

    @classmethod
    def fit_threshold(cls, cal_probs: np.ndarray, cal_acc: float) -> "AverageThresholdedConfidenceEstimator":
        p = np.asarray(cal_probs, dtype=float)
        if p.ndim == 1:
            conf = np.maximum(p, 1.0 - p)
        else:
            conf = np.max(p, axis=1)

        err_rate = float(np.clip(1.0 - cal_acc, 0.0, 1.0))
        # Handle ties and finite calibration using method='higher'
        threshold = float(np.quantile(conf, err_rate, method="higher"))
        return cls(atc_threshold=threshold)

    def estimate_accuracy(self, predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float)
        if p.ndim == 1:
            conf_t = np.maximum(p, 1.0 - p)
        else:
            conf_t = np.max(p, axis=1)

        return float(np.mean(conf_t >= self.atc_threshold))


class CBPEPerformanceEstimator:
    """Confidence-Based Performance Estimation (NannyML CBPE formulation).

    Reconstructs expected accuracy and confusion matrix assuming calibration holds.
    For binary classification:
    p_correct_i = 1 - |y_hat_i - p_i|
    """

    @staticmethod
    def estimate_accuracy(predicted_probs: np.ndarray) -> float:
        p = np.asarray(predicted_probs, dtype=float)
        if p.ndim == 1:
            pseudo_y = (p >= 0.5).astype(float)
            p_correct = 1.0 - np.abs(pseudo_y - p)
        else:
            # Multiclass formulation
            pseudo_y = np.argmax(p, axis=1)
            p_correct = p[np.arange(len(p)), pseudo_y]

        return float(np.mean(p_correct))

    @staticmethod
    def estimate_confusion_matrix(
        predicted_probs: np.ndarray, num_classes: int = 2
    ) -> Dict[str, any]:
        """Compute expected confusion matrix and macro metrics under Poisson-binomial model."""
        p = np.asarray(predicted_probs, dtype=float)
        if p.ndim == 1:
            # Binary one-hot conversion
            p = np.column_stack([1.0 - p, p])

        preds = np.argmax(p, axis=1)
        n = len(preds)

        class_metrics = {}
        for c in range(num_classes):
            is_pred_c = (preds == c).astype(float)
            p_c = p[:, c]

            tp_hat = float(np.sum(is_pred_c * (1.0 - np.abs(1.0 - p_c))))
            fp_hat = float(np.sum(is_pred_c * np.abs(1.0 - p_c)))
            tn_hat = float(np.sum((1.0 - is_pred_c) * (1.0 - np.abs(p_c))))
            fn_hat = float(np.sum((1.0 - is_pred_c) * np.abs(p_c)))

            precision = tp_hat / max(1e-8, tp_hat + fp_hat)
            recall = tp_hat / max(1e-8, tp_hat + fn_hat)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)

            class_metrics[c] = {
                "TP": tp_hat, "FP": fp_hat, "TN": tn_hat, "FN": fn_hat,
                "precision": precision, "recall": recall, "f1": f1,
            }

        macro_f1 = float(np.mean([m["f1"] for m in class_metrics.values()]))
        overall_acc = float(np.sum([m["TP"] for m in class_metrics.values()]) / max(1, n))

        return {
            "overall_accuracy": overall_acc,
            "macro_f1": macro_f1,
            "class_metrics": class_metrics,
        }


# =====================================================================
# 3. CONFIDENCE OPTIMAL TRANSPORT WITH THRESHOLDING (COTT)
# =====================================================================

class ConfidenceOptimalTransportEstimator:
    """Confidence Optimal Transport with Thresholding (COTT) (Zhang et al., NeurIPS 2023).

    Solves discrete optimal transport between target predicted probability vectors
    and source one-hot label distributions under an L_infinity cost matrix C:
    P* = argmin_{P in Pi(mu_T, mu_S)} <P, C>

    Where:
    - Target marginal: uniform 1/n across target instances.
    - Source marginal: uniform 1/N across source calibration instances.
    - C_{ij} = || e_{y_i^S} - p_j^T ||_inf
    - Calibration cost threshold t is learned on source calibration:
      Cost_i^S = || e_{y_i^S} - p_i^S ||_inf
      mean( I( Cost_i^S >= t ) ) = Err(D_cal)
    - Target error estimate:
      epsilon_hat_COTT = sum_{i, j : C_{ij} >= t} P*_{ij}
      Acc_hat_COTT = 1 - epsilon_hat_COTT
    """

    def __init__(
        self,
        cost_threshold: float,
        cal_subsample_probs: np.ndarray,
        cal_subsample_labels: np.ndarray,
    ):
        self.cost_threshold = float(cost_threshold)
        self.p_cal = np.asarray(cal_subsample_probs, dtype=float)
        self.y_cal = np.asarray(cal_subsample_labels, dtype=int)
        self.num_classes = 2 if self.p_cal.ndim == 1 else self.p_cal.shape[1]

    @classmethod
    def fit(
        cls,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        cal_acc: float,
        max_cal_samples: int = 100,
        seed: int = 42,
    ) -> "ConfidenceOptimalTransportEstimator":
        """Calibrate cost threshold t on source calibration data."""
        p_raw = np.asarray(cal_probs, dtype=float)
        y_raw = np.asarray(cal_labels, dtype=int)
        n_cal = len(p_raw)

        # Convert 1D binary probabilities to 2D distributions
        if p_raw.ndim == 1:
            p_mat = np.column_stack([1.0 - p_raw, p_raw])
        else:
            p_mat = p_raw

        # Compute point-wise source nonconformity cost || e_{y_i} - p_i ||_inf
        source_costs = []
        for i in range(n_cal):
            e_y = np.zeros(p_mat.shape[1])
            e_y[y_raw[i]] = 1.0
            cost_i = np.max(np.abs(e_y - p_mat[i]))
            source_costs.append(cost_i)

        source_costs = np.array(source_costs)
        target_err = float(np.clip(1.0 - cal_acc, 0.0, 1.0))
        # Quantile threshold t* such that fraction with cost >= t* matches source error
        threshold = float(np.quantile(source_costs, 1.0 - target_err, method="higher"))

        # Subsample source calibration for tractable LP solving (HiGHS solver)
        if n_cal > max_cal_samples:
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(n_cal, size=max_cal_samples, replace=False)
            p_sub = p_mat[sub_idx]
            y_sub = y_raw[sub_idx]
        else:
            p_sub = p_mat
            y_sub = y_raw

        return cls(
            cost_threshold=threshold,
            cal_subsample_probs=p_sub,
            cal_subsample_labels=y_sub,
        )

    def estimate_accuracy(self, target_probs: np.ndarray, max_target_samples: int = 100) -> float:
        """Solve optimal transport LP and estimate target accuracy."""
        p_t = np.asarray(target_probs, dtype=float)
        if p_t.ndim == 1:
            p_t_mat = np.column_stack([1.0 - p_t, p_t])
        else:
            p_t_mat = p_t

        n_target = len(p_t_mat)
        # Subsample target batch if very large to keep LP solver fast (<0.05s)
        if n_target > max_target_samples:
            sub_idx = np.linspace(0, n_target - 1, max_target_samples, dtype=int)
            p_t_mat = p_t_mat[sub_idx]

        N = len(self.p_cal)
        n = len(p_t_mat)

        # Build Cost Matrix C_{ij} = || e_{y_i^S} - p_j^T ||_inf of shape (N, n)
        C = np.zeros((N, n), dtype=float)
        for i in range(N):
            e_y = np.zeros(self.num_classes)
            e_y[self.y_cal[i]] = 1.0
            # Broadcast across target samples
            diff = np.abs(e_y[None, :] - p_t_mat)  # shape (n, num_classes)
            C[i, :] = np.max(diff, axis=1)

        c_vec = C.ravel()  # Length N * n

        # Constraint matrices for marginals:
        # sum_j gamma_{ij} = 1/N (N constraints)
        # sum_i gamma_{ij} = 1/n (n constraints)
        A_eq = np.zeros((N + n, N * n), dtype=float)
        b_eq = np.zeros(N + n, dtype=float)

        for i in range(N):
            A_eq[i, i * n : (i + 1) * n] = 1.0
            b_eq[i] = 1.0 / N

        for j in range(n):
            A_eq[N + j, j::n] = 1.0
            b_eq[N + j] = 1.0 / n

        # Drop one redundant constraint to ensure full row rank
        res = linprog(
            c_vec,
            A_eq=A_eq[:-1],
            b_eq=b_eq[:-1],
            bounds=(0.0, None),
            method="highs",
        )

        if not res.success:
            # Fallback to mean thresholding if LP solver encounters numerical issue
            fallback_err = np.mean(C >= self.cost_threshold)
            return float(np.clip(1.0 - fallback_err, 0.0, 1.0))

        gamma_opt = res.x.reshape((N, n))

        # COTT estimated error: mass transported to cost >= threshold
        mask_threshold = (C >= self.cost_threshold)
        est_error = float(np.sum(gamma_opt[mask_threshold]))

        return float(np.clip(1.0 - est_error, 0.0, 1.0))
