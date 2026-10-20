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
        d_marg = float(np.sum(self.w * (ks_arr ** 3)) ** (1.0 / 3.0))

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
            "d_marg_l3": d_marg,
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
    """The central TrustGuard multi-signal monitoring engine."""

    def __init__(
        self,
        frozen_pipeline: FrozenModelPipeline,
        X_cal: pd.DataFrame,
        w_dis: float = 0.35,
        w_attr: float = 0.25,
        w_conf: float = 0.20,
        w_grad: float = 0.20,
    ):
        self.pipeline = frozen_pipeline
        self.attr_monitor = AttributionWeightedDriftMonitor(
            X_cal=X_cal,
            shap_weights=frozen_pipeline.shap_weights,
            corr_cal=frozen_pipeline.source_corr_matrix,
        )
        self.cross_arch_monitor = CrossArchitectureDisagreementMonitor()

        self.w_dis = float(w_dis)
        self.w_attr = float(w_attr)
        self.w_conf = float(w_conf)
        self.w_grad = float(w_grad)

    def evaluate_window(self, X_unlabeled: pd.DataFrame) -> Dict[str, any]:
        """Compute all TrustGuard signals and composite risk score on an unlabeled window.

        Note: Strictly does not take or access ground-truth labels.
        """
        # 1. Primary and secondary model predictions
        probs_primary = self.pipeline.predict_primary_proba(X_unlabeled)
        probs_mlp = self.pipeline.predict_secondary_proba(X_unlabeled)

        # Signal 1: Attribution-Weighted Drift
        attr_out = self.attr_monitor.evaluate_window(X_unlabeled)
        s_attr = attr_out["s_attr"]

        # Signal 2: Cross-Architecture Disagreement
        dis_out = self.cross_arch_monitor.evaluate_window(probs_primary, probs_mlp)
        s_dis = dis_out["s_dis"]

        # Signal 3: Mondrian Conformal Ambiguity
        conf_metrics = self.pipeline.conformal_calibrator.compute_cardinality_metrics(probs_primary)
        s_conf = conf_metrics["abstention_rate"]

        # Signal 4: GdScore (Classification-Layer Gradient Norm)
        gdscore_val = self.pipeline.compute_gdscore(X_unlabeled)
        # Normalized excess above source calibration baseline
        s_grad = float(np.clip(
            (gdscore_val - self.pipeline.source_cal_gdscore) / max(0.01, self.pipeline.source_cal_gdscore),
            0.0,
            1.0,
        ))

        # Composite TrustGuard Risk Index
        risk_tg = float(np.clip(
            self.w_dis * s_dis + self.w_attr * s_attr + self.w_conf * s_conf + self.w_grad * s_grad,
            0.0,
            1.0,
        ))

        return {
            "risk_tg": risk_tg,
            "s_dis": s_dis,
            "s_attr": s_attr,
            "s_conf": s_conf,
            "s_grad": s_grad,
            "gdscore_raw": gdscore_val,
            "probs_primary": probs_primary,
            "probs_mlp": probs_mlp,
            "conformal_metrics": conf_metrics,
        }
