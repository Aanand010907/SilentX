"""TrustGuard Hybrid Post-Deployment Monitoring Architecture.

Integrates:
1. Attribution-Weighted L3 Input and Correlation Drift (S_attr)
2. Cross-Architecture Inductive Disagreement (S_dis) pairing Tree with Tabular MLP
3. Mondrian Split-Conformal Ambiguity & Empty-Set Cardinality (S_conf)
4. Classification-Layer Pseudo-Label Gradient Norm (GdScore / S_grad)
5. Variational Bayesian Last Layer Disagreement (D3M / S_d3m)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from src.models.frozen_pipeline import FrozenModelPipeline


class AttributionWeightedDriftMonitor:
    """Computes attribution-weighted L3 marginal drift and correlation shift."""

    def __init__(self, X_cal: pd.DataFrame, shap_weights: np.ndarray, corr_cal: np.ndarray, lambda_corr: float = 0.5):
        self.X_cal = X_cal.to_numpy(dtype=float)
        self.w = np.asarray(shap_weights, dtype=float).ravel()
        self.corr_cal = np.asarray(corr_cal, dtype=float)
        self.lambda_corr = float(lambda_corr)

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> Dict[str, float]:
        X_t = X_unlabeled.to_numpy(dtype=float)
        d = X_t.shape[1]

        # 1. Marginal L3 drift: (sum w_j * KS_j^3)^(1/3)
        ks_stats = []
        for j in range(d):
            res = ks_2samp(self.X_cal[:, j], X_t[:, j])
            ks_stats.append(res.statistic)

        ks_arr = np.array(ks_stats)
        # Linear expectation attribution weighting: sum w_j * KS_j
        # Avoids L3 cube-root amplification (w_j)^(1/3) on low-attribution noise features
        d_marg = float(np.sum(self.w * ks_arr))
        d_marg_l3 = float(np.sum(self.w * (ks_arr ** 3)) ** (1.0 / 3.0))

        # 2. Weighted Correlation Matrix Shift
        corr_t = np.corrcoef(X_t, rowvar=False)
        # Handle constant features or NaNs gracefully
        corr_t = np.nan_to_num(corr_t, nan=0.0)

        d_corr = 0.0
        for j in range(d):
            for k in range(j + 1, d):
                weight_jk = np.sqrt(self.w[j] * self.w[k])
                d_corr += weight_jk * abs(self.corr_cal[j, k] - corr_t[j, k])

        s_attr = min(1.0, float(d_marg + self.lambda_corr * d_corr))

        return {
            "s_attr": s_attr,
            "d_marg": d_marg,
            "d_marg_l3": d_marg_l3,
            "d_corr": float(d_corr),
        }


class CrossArchitectureDisagreementMonitor:
    """Measures inductive disagreement between tree ensemble and continuous neural network."""

    @staticmethod
    def evaluate_window(
        probs_tree: np.ndarray,
        probs_mlp: np.ndarray,
    ) -> Dict[str, float]:
        p_tree = np.asarray(probs_tree, dtype=float).ravel()
        p_mlp = np.asarray(probs_mlp, dtype=float).ravel()

        preds_tree = (p_tree >= 0.5).astype(int)
        preds_mlp = (p_mlp >= 0.5).astype(int)

        # 1. Hard prediction disagreement rate
        hard_disagreement = float(np.mean(preds_tree != preds_mlp))

        # 2. Jensen-Shannon divergence across predictions
        m = len(p_tree)
        jsd_total = 0.0
        for i in range(m):
            dist_tree = np.array([1.0 - p_tree[i], p_tree[i]])
            dist_mlp = np.array([1.0 - p_mlp[i], p_mlp[i]])
            js_dist = jensenshannon(dist_tree, dist_mlp, base=2.0)
            jsd_total += float(js_dist**2)

        mean_jsd = jsd_total / max(1, m)
        s_dis = float(min(1.0, hard_disagreement + 0.5 * mean_jsd))

        return {
            "s_dis": s_dis,
            "hard_disagreement_rate": hard_disagreement,
            "mean_jsd": mean_jsd,
        }


class TrustGuardCompositeMonitor:
    """The central TrustGuard multi-signal monitoring engine.

    Computes normalized excess signals relative to source calibration baseline:
    tilde{S}_k(W_t) = max(0, S_k(W_t) - S_k(D_cal))

    Combines:
    1. Attribution-Weighted L3 Input & Correlation Drift (S_attr)
    2. Cross-Architecture Inductive Disagreement (S_dis)
    3. Mondrian Conformal Instability / Abstention Rate (S_conf)
    4. Classification-Layer Pseudo-Label Gradient Norm (S_grad)
    5. Variational Bayesian Last Layer Disagreement (S_d3m)
    """

    def __init__(
        self,
        frozen_pipeline: FrozenModelPipeline,
        X_cal: pd.DataFrame,
        w_dis: float = 0.25,
        w_attr: float = 0.30,
        w_conf: float = 0.15,
        w_grad: float = 0.15,
        w_d3m: float = 0.15,
        alpha_alarm: float = 0.05,
        seed: int = 42,
    ):
        self.pipeline = frozen_pipeline
        self.X_cal = X_cal
        self.attr_monitor = AttributionWeightedDriftMonitor(
            X_cal=X_cal,
            shap_weights=frozen_pipeline.shap_weights,
            corr_cal=frozen_pipeline.source_corr_matrix,
        )
        self.cross_arch_monitor = CrossArchitectureDisagreementMonitor()

        # Weight normalization
        total_w = w_dis + w_attr + w_conf + w_grad + w_d3m
        self.w_dis = float(w_dis / total_w)
        self.w_attr = float(w_attr / total_w)
        self.w_conf = float(w_conf / total_w)
        self.w_grad = float(w_grad / total_w)
        self.w_d3m = float(w_d3m / total_w)
        self.alpha_alarm = float(alpha_alarm)
        self.seed = seed

        # Calibrate baseline signal levels on source calibration data
        self._calibrate_baselines(X_cal)

    def _calibrate_baselines(self, X_cal: pd.DataFrame) -> None:
        """Lock in-distribution baseline signal levels and the 95th-percentile alarm threshold."""
        # 1. Baseline Attribution Drift on cal
        attr_cal = self.attr_monitor.evaluate_window(X_cal)
        self.base_s_attr = float(attr_cal["s_attr"])

        # 2. Baseline Cross-Arch Disagreement on cal
        probs_tree_cal = self.pipeline.predict_primary_proba(X_cal)
        probs_mlp_cal = self.pipeline.predict_secondary_proba(X_cal)
        dis_cal = self.cross_arch_monitor.evaluate_window(probs_tree_cal, probs_mlp_cal)
        self.base_s_dis = float(dis_cal["s_dis"])

        # 3. Baseline Conformal Abstention on cal
        conf_cal = self.pipeline.conformal_calibrator.compute_cardinality_metrics(probs_tree_cal)
        self.base_s_conf = float(conf_cal["abstention_rate"])

        # 4. Baseline GdScore on cal
        self.base_s_grad = float(self.pipeline.source_cal_gdscore)

        # 5. Bootstrap calibrate risk threshold on source calibration sub-batches
        rng = np.random.default_rng(self.seed)
        n_cal = len(X_cal)
        batch_size = min(250, n_cal // 2)
        boot_risks = []

        for _ in range(50):
            idx = rng.choice(n_cal, size=batch_size, replace=True)
            X_sub = X_cal.iloc[idx].reset_index(drop=True)
            r = self._compute_raw_signals(X_sub)
            boot_risks.append(r["risk_tg"])

        # Lock (1 - alpha) source alarm threshold (e.g., 95th percentile)
        self.alarm_threshold = float(np.quantile(boot_risks, 1.0 - self.alpha_alarm))

    def _compute_raw_signals(self, X_unlabeled: pd.DataFrame) -> Dict[str, any]:
        probs_primary = self.pipeline.predict_primary_proba(X_unlabeled)
        probs_mlp = self.pipeline.predict_secondary_proba(X_unlabeled)

        # Signal 1: Attribution drift
        attr_out = self.attr_monitor.evaluate_window(X_unlabeled)
        s_attr = attr_out["s_attr"]
        excess_s_attr = float(max(0.0, s_attr - self.base_s_attr))

        # Signal 2: Cross-Architecture Disagreement
        dis_out = self.cross_arch_monitor.evaluate_window(probs_primary, probs_mlp)
        s_dis = dis_out["s_dis"]
        excess_s_dis = float(max(0.0, s_dis - self.base_s_dis))

        # Signal 3: Conformal Ambiguity
        conf_metrics = self.pipeline.conformal_calibrator.compute_cardinality_metrics(probs_primary)
        s_conf = conf_metrics["abstention_rate"]
        excess_s_conf = float(max(0.0, s_conf - self.base_s_conf))

        # Signal 4: GdScore
        gdscore_val = self.pipeline.compute_gdscore(X_unlabeled)
        excess_s_grad = float(max(0.0, (gdscore_val - self.base_s_grad) / max(0.01, self.base_s_grad)))

        # Signal 5: D3M Posterior Disagreement
        X_norm = self.pipeline.preprocessor.transform(X_unlabeled)
        d3m_out = self.pipeline.d3m_calibrator.evaluate_batch(self.pipeline.secondary_mlp, X_norm)
        s_d3m = float(d3m_out["disagreement_rate"])
        d3m_thresh = float(d3m_out["threshold_95"])
        excess_s_d3m = float(max(0.0, s_d3m - d3m_thresh))

        # Composite TrustGuard Risk Score
        risk_tg = float(np.clip(
            self.w_dis * excess_s_dis
            + self.w_attr * excess_s_attr
            + self.w_conf * excess_s_conf
            + self.w_grad * excess_s_grad
            + self.w_d3m * excess_s_d3m,
            0.0,
            1.0,
        ))

        return {
            "risk_tg": risk_tg,
            "s_dis": s_dis,
            "excess_s_dis": excess_s_dis,
            "s_attr": s_attr,
            "excess_s_attr": excess_s_attr,
            "s_conf": s_conf,
            "excess_s_conf": excess_s_conf,
            "s_grad": excess_s_grad,
            "gdscore_raw": gdscore_val,
            "s_d3m": s_d3m,
            "excess_s_d3m": excess_s_d3m,
            "d3m_is_alert": d3m_out["is_alert"],
            "probs_primary": probs_primary,
            "probs_mlp": probs_mlp,
            "conformal_metrics": conf_metrics,
        }

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> Dict[str, any]:
        """Compute all TrustGuard signals and composite risk score on an unlabeled window."""
        res = self._compute_raw_signals(X_unlabeled)
        res["alarm_threshold"] = self.alarm_threshold
        res["is_alarm"] = bool(res["risk_tg"] >= self.alarm_threshold)
        return res
