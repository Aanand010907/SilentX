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
from scipy.optimize import linprog, linear_sum_assignment


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
# 3. PUBLISHED CONFIDENCE OPTIMAL TRANSPORT WITH THRESHOLDING (COTT)
# =====================================================================

class ConfidenceOptimalTransportEstimator:
    """Published Confidence Optimal Transport with Thresholding (COTT) (Lu et al., NeurIPS 2023).

    Official reference:
    - Paper: "Characterizing Out-of-Distribution Error via Optimal Transport", NeurIPS 2023.
    - arXiv: 2305.15640
    - Official repository: https://github.com/luyuzhe111/COT (run_estimation.py)

    Mathematical formulation:
    1. Preserves full probability vectors p_i in Delta^{k-1} for all samples.
    2. Represents ground-truth labels using one-hot vectors e_y in {0, 1}^k.
    3. Uses the L_infinity cost matrix:
       C_{ij} = || p_i - e_{y_j} ||_inf
    4. Calibration Phase (Source Reference Set):
       - Solves optimal matching P*_val = argmin_{P} <P, C^val> between source validation
         probability vectors and empirical source one-hot labels.
       - Extracts matched costs: c_i^matched = C^val_{i, pi(i)}.
       - Sets cost threshold t* such that the fraction of samples with c_i^matched >= t*
         matches the observed source validation classification error Err(D_val).
    5. Deployment Phase (Target Unlabeled Batch):
       - Computes target probability vectors p_j^T.
       - Solves optimal transport plan P* between target probability distribution and
         empirical source label distribution under L_infinity cost.
       - Estimates target error: epsilon_hat = sum_{j, i : C_{ji} >= t*} P*_{ji}.
       - Estimates target accuracy: Acc_hat = 1 - epsilon_hat.
    """

    def __init__(
        self,
        cost_threshold: float,
        source_ref_probs: np.ndarray,
        source_ref_labels: np.ndarray,
    ):
        self.cost_threshold = float(cost_threshold)
        self.p_ref = np.asarray(source_ref_probs, dtype=float)
        self.y_ref = np.asarray(source_ref_labels, dtype=int)
        self.num_classes = self.p_ref.shape[1]

    @classmethod
    def _to_prob_matrix(cls, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        if p.ndim == 1:
            p_mat = np.column_stack([1.0 - p, p])
        elif p.ndim == 2:
            p_mat = p
        else:
            raise ValueError(f"Expected 1D or 2D probability array, got shape {p.shape}")
        # Normalize rows to strictly sum to 1
        row_sums = np.sum(p_mat, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return np.clip(p_mat / row_sums, 0.0, 1.0)

    @classmethod
    def _compute_linf_cost_matrix(cls, probs: np.ndarray, one_hot_labels: np.ndarray) -> np.ndarray:
        """Compute pairwise L_infinity cost matrix between probability vectors and one-hot labels.

        C_{ij} = || p_i - e_{y_j} ||_inf = max_c | p_{i, c} - (e_{y_j})_c |
        """
        n_p, n_classes = probs.shape
        n_y = len(one_hot_labels)

        # Broadcast diff: shape (n_p, n_y, n_classes)
        # Using memory-efficient broadcasting for practical batch sizes
        diff = np.abs(probs[:, None, :] - one_hot_labels[None, :, :])
        return np.max(diff, axis=-1)

    @classmethod
    def fit(
        cls,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        cal_acc: Optional[float] = None,
        max_cal_samples: int = 150,
        seed: int = 42,
    ) -> "ConfidenceOptimalTransportEstimator":
        """Calibrate cost threshold t* on source calibration reference data.

        Strictly follows Section 3.2 of Lu et al. (NeurIPS 2023):
        Computes optimal transport plan P*_val on the validation set, calculates matched
        costs, and sets t* to match the empirical validation error.
        """
        p_mat = cls._to_prob_matrix(cal_probs)
        y_vec = np.asarray(cal_labels, dtype=int)
        n_cal, num_classes = p_mat.shape

        if len(y_vec) != n_cal:
            raise ValueError(f"Mismatch: cal_probs ({n_cal}) vs cal_labels ({len(y_vec)})")

        # Subsample reference set if needed to keep matching fast and deterministic
        if n_cal > max_cal_samples:
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(n_cal, size=max_cal_samples, replace=False)
            p_sub = p_mat[sub_idx]
            y_sub = y_vec[sub_idx]
        else:
            p_sub = p_mat
            y_sub = y_vec

        n_sub = len(y_sub)
        e_sub = np.zeros((n_sub, num_classes), dtype=float)
        for i, y_val in enumerate(y_sub):
            e_sub[i, y_val] = 1.0

        # Construct source validation cost matrix C_val (n_sub x n_sub)
        C_val = cls._compute_linf_cost_matrix(p_sub, e_sub)

        # Solve optimal matching via Hungarian algorithm (Proposition 2.1 in Peyre & Cuturi 2019)
        row_ind, col_ind = linear_sum_assignment(C_val)
        matched_costs = C_val[row_ind, col_ind]

        # Empirical validation error
        preds_sub = np.argmax(p_sub, axis=1)
        err_val = float(np.mean(preds_sub != y_sub))
        if cal_acc is not None:
            err_val = float(np.clip(1.0 - cal_acc, 0.0, 1.0))

        # Quantile threshold t* such that fraction of samples with matched cost >= t* matches err_val
        threshold = float(np.quantile(matched_costs, max(0.0, 1.0 - err_val), method="higher"))

        return cls(
            cost_threshold=threshold,
            source_ref_probs=p_sub,
            source_ref_labels=y_sub,
        )

    def estimate_accuracy(
        self,
        target_probs: np.ndarray,
        batch_size: int = 150,
        seed: int = 42,
    ) -> float:
        """Estimate target accuracy using COTT optimal transport with thresholding.

        Computes optimal transport plan between target predicted probabilities and
        source reference labels, accumulating mass transported to cost >= t*.
        """
        p_t = self._to_prob_matrix(target_probs)
        n_target = len(p_t)

        if n_target == 0:
            return 1.0

        N = len(self.p_ref)
        e_ref = np.zeros((N, self.num_classes), dtype=float)
        for i, y_val in enumerate(self.y_ref):
            e_ref[i, y_val] = 1.0

        # If target size is larger than batch_size, evaluate in chunks as in paper Sec 3.2
        if n_target > batch_size:
            chunk_errors = []
            n_chunks = int(np.ceil(n_target / batch_size))
            for k in range(n_chunks):
                start = k * batch_size
                end = min(n_target, (k + 1) * batch_size)
                p_chunk = p_t[start:end]
                err_chunk = self._estimate_batch_error(p_chunk, e_ref)
                chunk_errors.append(err_chunk * (end - start))
            total_err = sum(chunk_errors) / n_target
            return float(np.clip(1.0 - total_err, 0.0, 1.0))

        total_err = self._estimate_batch_error(p_t, e_ref)
        return float(np.clip(1.0 - total_err, 0.0, 1.0))

    def _estimate_batch_error(self, p_target: np.ndarray, e_ref: np.ndarray) -> float:
        """Estimate error on a single batch via optimal transport."""
        n = len(p_target)
        N = len(e_ref)

        # C matrix of shape (n, N): row = target sample, col = source reference label
        C = self._compute_linf_cost_matrix(p_target, e_ref)

        # Case 1: Equal marginal sizes (n == N) -> exact Hungarian matching
        if n == N:
            row_ind, col_ind = linear_sum_assignment(C)
            matched_costs = C[row_ind, col_ind]
            err_est = float(np.mean(matched_costs >= self.cost_threshold))
            return err_est

        # Case 2: Unequal marginal sizes -> Kantorovich linear program via HiGHS
        c_vec = C.ravel()  # shape (n * N,)

        # Constraints:
        # sum_j P_{ij} = 1/n (n constraints for target marginals)
        # sum_i P_{ij} = 1/N (N constraints for source marginals)
        A_eq = np.zeros((n + N, n * N), dtype=float)
        b_eq = np.zeros(n + N, dtype=float)

        for i in range(n):
            A_eq[i, i * N : (i + 1) * N] = 1.0
            b_eq[i] = 1.0 / n

        for j in range(N):
            A_eq[n + j, j::N] = 1.0
            b_eq[n + j] = 1.0 / N

        res = linprog(
            c_vec,
            A_eq=A_eq[:-1],
            b_eq=b_eq[:-1],
            bounds=(0.0, None),
            method="highs",
        )

        if not res.success:
            # Deterministic fallback to mean threshold violation if solver fails
            fallback_err = float(np.mean(C >= self.cost_threshold))
            return fallback_err

        gamma_opt = res.x.reshape((n, N))
        mask_threshold = (C >= self.cost_threshold)
        err_est = float(np.sum(gamma_opt[mask_threshold]))
        return err_est


# =====================================================================
# 4. EXPERIMENTAL SCALAR CONFIDENCE OPTIMAL TRANSPORT HEURISTIC
# =====================================================================

class ScalarConfidenceOTHeuristic:
    """Experimental 1D Scalar Confidence Optimal Transport Heuristic.

    IMPORTANT SCIENTIFIC DISTINCTION:
    This is NOT the published COTT algorithm from Lu et al. (NeurIPS 2023).
    It operates solely on 1D top-1 scalar confidence scores c = max_k p_k under an
    absolute difference cost |c_i^S - c_j^T|, transporting source error indicators
    to the target domain. Preserved under this explicit name for baseline comparison.
    """

    def __init__(
        self,
        source_confidences: np.ndarray,
        source_errors: np.ndarray,
    ):
        self.source_conf = np.asarray(source_confidences, dtype=float).ravel()
        self.source_errors = np.asarray(source_errors, dtype=float).ravel()

    @classmethod
    def fit(
        cls,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        max_samples: int = 100,
        seed: int = 42,
    ) -> "ScalarConfidenceOTHeuristic":
        p = np.asarray(cal_probs, dtype=float)
        y = np.asarray(cal_labels, dtype=int).ravel()
        if p.ndim == 1:
            conf = np.maximum(p, 1.0 - p)
            preds = (p >= 0.5).astype(int)
        else:
            conf = np.max(p, axis=1)
            preds = np.argmax(p, axis=1)

        errors = (preds != y).astype(float)
        n = len(conf)

        if n > max_samples:
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(n, size=max_samples, replace=False)
            conf = conf[sub_idx]
            errors = errors[sub_idx]

        return cls(source_confidences=conf, source_errors=errors)

    def estimate_accuracy(self, target_probs: np.ndarray, max_target_samples: int = 100) -> float:
        p_t = np.asarray(target_probs, dtype=float)
        if p_t.ndim == 1:
            conf_t = np.maximum(p_t, 1.0 - p_t)
        else:
            conf_t = np.max(p_t, axis=1)

        m = len(conf_t)
        if m > max_target_samples:
            sub_idx = np.linspace(0, m - 1, max_target_samples, dtype=int)
            conf_t = conf_t[sub_idx]

        N = len(self.source_conf)
        n = len(conf_t)

        # 1D cost: |c_i^S - c_j^T|
        C = np.abs(self.source_conf[:, None] - conf_t[None, :])
        c_vec = C.ravel()

        A_eq = np.zeros((N + n, N * n), dtype=float)
        b_eq = np.zeros(N + n, dtype=float)

        for i in range(N):
            A_eq[i, i * n : (i + 1) * n] = 1.0
            b_eq[i] = 1.0 / N

        for j in range(n):
            A_eq[N + j, j::n] = 1.0
            b_eq[N + j] = 1.0 / n

        res = linprog(
            c_vec,
            A_eq=A_eq[:-1],
            b_eq=b_eq[:-1],
            bounds=(0.0, None),
            method="highs",
        )

        if not res.success:
            return float(1.0 - np.mean(self.source_errors))

        gamma_opt = res.x.reshape((N, n))
        # Weight source errors by transported target mass
        target_error = float(np.sum(gamma_opt * self.source_errors[:, None]))
        return float(np.clip(1.0 - target_error, 0.0, 1.0))

