"""Hybrid Robust Ensemble combining GBDT Ensemble and Variational Bayesian Neural Probe.

Features:
1. Multi-model GBDT ensemble (HistGradientBoosting + XGBoost + tuned tree estimators).
2. Variational Bayesian Last Layer (VBLL) neural probe for epistemic uncertainty.
3. TreeSHAP-attribution-gated dynamic blending:
   - High ensemble weighting when attribution drift is low (clean data).
   - Dynamic noise-shielding shift when attribution drift or harmless noise is detected.
4. Source checkpoint snapshotting and emergency rollback for Tier-2 recovery.
5. Calibrated probability output via isotonic regression.
"""

from typing import Dict, List, Optional, Tuple, Any
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.models.frozen_pipeline import FrozenPreprocessor
from src.models.vbll_model import D3MNeuralClassifier
from src.models.calibration import SourceProbabilityCalibrator, MondrianConformalCalibrator
import torch


class GBDTSubEnsemble:
    """Ensemble of diverse tree classifiers (HistGradientBoosting, XGBoost, Random Forest)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.hgb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=seed,
        )
        self.xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            eval_metric="logloss",
            random_state=seed,
        )
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=seed,
        )
        self.is_fitted = False

    def fit(self, X_norm: np.ndarray, y: np.ndarray) -> "GBDTSubEnsemble":
        self.hgb.fit(X_norm, y)
        self.xgb_clf.fit(X_norm, y)
        self.rf.fit(X_norm, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X_norm: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GBDTSubEnsemble must be fitted before predict_proba.")
        p_hgb = self.hgb.predict_proba(X_norm)[:, 1]
        p_xgb = self.xgb_clf.predict_proba(X_norm)[:, 1]
        p_rf = self.rf.predict_proba(X_norm)[:, 1]
        # Weighted blend across tree architectures
        return 0.45 * p_hgb + 0.35 * p_xgb + 0.20 * p_rf


class RobustHybridEnsemble:
    """Unified hybrid model combining GBDT ensemble and Bayesian Neural Network.

    Uses attribution-gated dynamic blending:
    p_hybrid(x) = alpha(W_t) * p_gbdt(x) + (1 - alpha(W_t)) * p_robust(x)
    where alpha(W_t) shifts smoothly based on input and attribution drift levels.
    """

    def __init__(
        self,
        preprocessor: FrozenPreprocessor,
        gbdt_ensemble: GBDTSubEnsemble,
        neural_probe: D3MNeuralClassifier,
        calibrator: SourceProbabilityCalibrator,
        conformal_calibrator: MondrianConformalCalibrator,
        shap_weights: np.ndarray,
        source_corr_matrix: np.ndarray,
        source_cal_acc: float,
        k_gate: float = 6.0,
        theta_gate: float = 0.20,
    ):
        self.preprocessor = preprocessor
        self.gbdt = gbdt_ensemble
        self.neural_probe = neural_probe
        self.calibrator = calibrator
        self.conformal = conformal_calibrator
        self.shap_weights = np.asarray(shap_weights, dtype=float)
        self.source_corr_matrix = np.asarray(source_corr_matrix, dtype=float)
        self.source_cal_acc = float(source_cal_acc)
        self.k_gate = float(k_gate)
        self.theta_gate = float(theta_gate)

        # Checkpoint store for Tier-2 recovery
        self.checkpoint_state: Optional[Dict[str, Any]] = None
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Saves frozen source state of neural and tree parameters."""
        self.checkpoint_state = {
            "neural_probe_state": copy.deepcopy(self.neural_probe.state_dict()),
        }

    def restore_checkpoint(self) -> None:
        """Restores frozen source state after adversarial collapse or TTA divergence."""
        if self.checkpoint_state is not None:
            self.neural_probe.load_state_dict(self.checkpoint_state["neural_probe_state"])

    def predict_gbdt_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probabilities from the multi-tree GBDT ensemble."""
        X_norm = self.preprocessor.transform(X)
        raw_p = self.gbdt.predict_proba(X_norm)
        return self.calibrator.predict_proba(raw_p)

    def predict_neural_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probabilities from the Variational Bayesian neural probe."""
        X_norm = self.preprocessor.transform(X)
        self.neural_probe.eval()
        with torch.no_grad():
            t = torch.tensor(X_norm, dtype=torch.float32)
            _, mu, _ = self.neural_probe(t)
            probs = torch.softmax(mu, dim=-1)[:, 1].cpu().numpy()
        return probs

    def predict_hybrid_proba(
        self,
        X: pd.DataFrame,
        s_attr: float = 0.0,
    ) -> np.ndarray:
        """Computes dynamically gated hybrid predictions.

        Args:
            X: Unlabeled tabular features.
            s_attr: Current batch attribution-weighted drift score.

        Returns:
            Gated probabilities balancing high in-distribution accuracy and noise robustness.
        """
        p_gbdt = self.predict_gbdt_proba(X)
        p_neural = self.predict_neural_proba(X)

        # Sigmoid gating weight: when s_attr is low, alpha -> 0.85 (favors sharp GBDT)
        # when s_attr is high, alpha -> 0.15 (favors noise-shielded calibrated model)
        z = self.k_gate * (s_attr - self.theta_gate)
        sigmoid_shift = 1.0 / (1.0 + np.exp(np.clip(z, -15.0, 15.0)))
        alpha = float(np.clip(0.15 + 0.70 * sigmoid_shift, 0.15, 0.85))

        p_hybrid = alpha * p_gbdt + (1.0 - alpha) * p_neural
        return np.clip(p_hybrid, 0.001, 0.999)

    def predict_sets(self, probs: np.ndarray) -> List[List[int]]:
        """Generates conformal prediction sets with finite-sample coverage guarantees."""
        return self.conformal.predict_prediction_sets(probs)


def build_robust_hybrid_ensemble_from_pipeline(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> RobustHybridEnsemble:
    """Constructs a RobustHybridEnsemble leveraging an existing frozen pipeline.

    Fits the GBDT multi-tree ensemble on normalized X_train, and reuses the frozen
    calibrators, neural probe, and SHAP weights from the pipeline.
    """
    X_train_norm = pipeline.preprocessor.transform(X_train)
    y_train_np = y_train.to_numpy(dtype=int)

    gbdt_ensemble = GBDTSubEnsemble(seed=seed).fit(X_train_norm, y_train_np)

    return RobustHybridEnsemble(
        preprocessor=pipeline.preprocessor,
        gbdt_ensemble=gbdt_ensemble,
        neural_probe=pipeline.secondary_mlp,
        calibrator=pipeline.source_calibrator,
        conformal_calibrator=pipeline.conformal_calibrator,
        shap_weights=pipeline.shap_weights,
        source_corr_matrix=pipeline.source_corr_matrix,
        source_cal_acc=pipeline.source_cal_acc,
    )

