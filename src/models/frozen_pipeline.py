"""Frozen inference pipeline and multi-architecture model bundle for SilentX.

Enforces:
1. All preprocessing transformations are fitted strictly on source training data and frozen.
2. Primary tree classifier (XGBoost / HistGradientBoosting) paired with calibrated probabilities.
3. Secondary neural MLP with BatchNorm/Dropout and closed-form GdScore computation.
4. Low-capacity L2 Logistic Regression diagnostic classifier.
5. Locked reference calibration metrics (ATC threshold, Mondrian quantiles, SHAP reliance weights).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import shap

from src.models.calibration import (
    SourceProbabilityCalibrator,
    MondrianConformalCalibrator,
    compute_expected_calibration_error,
)
from src.models.vbll_model import D3MNeuralClassifier, D3MReferenceCalibrator
import torch


class FrozenPreprocessor:
    """Standardizes tabular features based on statistics fit strictly on source training data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "FrozenPreprocessor":
        self.feature_names_ = list(X_train.columns)
        self.scaler.fit(X_train.to_numpy(dtype=float))
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on source training data before transforming.")
        return self.scaler.transform(X[self.feature_names_].to_numpy(dtype=float))


class FrozenModelPipeline:
    """Immutable inference pipeline locking models, calibrators, and source reference stats."""

    def __init__(
        self,
        preprocessor: FrozenPreprocessor,
        primary_tree: Union[xgb.XGBClassifier, HistGradientBoostingClassifier],
        secondary_mlp: D3MNeuralClassifier,
        diagnostic_logreg: LogisticRegression,
        source_calibrator: SourceProbabilityCalibrator,
        conformal_calibrator: MondrianConformalCalibrator,
        d3m_calibrator: D3MReferenceCalibrator,
        feature_names: List[str],
        atc_threshold: float,
        doc_offset: float,
        shap_weights: np.ndarray,
        source_corr_matrix: np.ndarray,
        source_cal_acc: float,
        source_cal_mean_conf: float,
        source_cal_gdscore: float,
    ):
        self.preprocessor = preprocessor
        self.primary_tree = primary_tree
        self.secondary_mlp = secondary_mlp
        self.diagnostic_logreg = diagnostic_logreg
        self.source_calibrator = source_calibrator
        self.conformal_calibrator = conformal_calibrator
        self.d3m_calibrator = d3m_calibrator
        self.feature_names = feature_names

        # Locked source reference statistics
        self.atc_threshold = float(atc_threshold)
        self.doc_offset = float(doc_offset)
        self.shap_weights = np.asarray(shap_weights, dtype=float)
        self.source_corr_matrix = np.asarray(source_corr_matrix, dtype=float)
        self.source_cal_acc = float(source_cal_acc)
        self.source_cal_mean_conf = float(source_cal_mean_conf)
        self.source_cal_gdscore = float(source_cal_gdscore)

    def predict_primary_proba(self, X_unlabeled: pd.DataFrame) -> np.ndarray:
        """Compute calibrated probabilities from the frozen primary tree model."""
        X_norm = self.preprocessor.transform(X_unlabeled)
        raw_probs = self.primary_tree.predict_proba(X_norm)[:, 1]
        return self.source_calibrator.predict_proba(raw_probs)

    def predict_secondary_proba(self, X_unlabeled: pd.DataFrame) -> np.ndarray:
        """Compute probabilities from the secondary neural network probe."""
        X_norm = self.preprocessor.transform(X_unlabeled)
        self.secondary_mlp.eval()
        with torch.no_grad():
            t = torch.tensor(X_norm, dtype=torch.float32)
            _, mu, _ = self.secondary_mlp(t)
            probs = torch.softmax(mu, dim=-1)[:, 1].cpu().numpy()
        return probs

    def predict_diagnostic_proba(self, X_unlabeled: pd.DataFrame) -> np.ndarray:
        """Compute probabilities from the low-capacity diagnostic logistic regression."""
        X_norm = self.preprocessor.transform(X_unlabeled)
        return self.diagnostic_logreg.predict_proba(X_norm)[:, 1]

    def compute_gdscore(self, X_unlabeled: pd.DataFrame) -> float:
        """Compute single-step classification-layer gradient norm (GdScore) in closed form.

        g(x_i) = || (p(x_i) - e_{hat_y_i}) z_i^T ||_F
        where z_i is the penultimate representation and hat_y_i is hard pseudo-label.
        """
        X_norm = self.preprocessor.transform(X_unlabeled)
        self.secondary_mlp.eval()
        with torch.no_grad():
            t = torch.tensor(X_norm, dtype=torch.float32)
            h, mu, _ = self.secondary_mlp(t)
            probs = torch.softmax(mu, dim=-1)[:, 1].cpu().numpy()
            h_np = h.cpu().numpy()

        pseudo_y = (probs >= 0.5).astype(float)
        residual = np.abs(probs - pseudo_y)[:, None]  # |p - hat_y|
        # Frobenius norm of outer product (residual * z^T) is residual * ||z||_2
        h_norms = np.linalg.norm(h_np, axis=1)
        grad_norms = residual.ravel() * h_norms
        return float(np.mean(grad_norms))


def train_and_freeze_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    feature_names: List[str],
    tree_family: str = "xgboost",
    seed: int = 42,
) -> FrozenModelPipeline:
    """Train base learners, evaluate calibration, and lock all source reference statistics.

    Enforces that:
    1. Preprocessor is fitted solely on X_train.
    2. Primary, secondary, and diagnostic models are fitted on X_train.
    3. Calibrator, conformal quantiles, SHAP weights, and D3M threshold are fitted on X_cal.
    """
    # 1. Fit frozen preprocessor on source training data only
    preprocessor = FrozenPreprocessor().fit(X_train)
    X_train_norm = preprocessor.transform(X_train)
    X_cal_norm = preprocessor.transform(X_cal)

    y_train_np = y_train.to_numpy(dtype=int)
    y_cal_np = y_cal.to_numpy(dtype=int)

    # 2. Train Primary Classifier
    if tree_family == "xgboost":
        primary_tree = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
    else:
        primary_tree = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=seed,
        )
    primary_tree.fit(X_train_norm, y_train_np)

    # 3. Train Secondary Neural Network (MLP with BatchNorm & Dropout & VBLL head)
    n_features = len(feature_names)
    secondary_mlp = D3MNeuralClassifier(
        in_features=n_features,
        hidden_dims=[64, 32],
        dropout_rate=0.2,
        prior_var=1.0,
    )

    optimizer = torch.optim.AdamW(secondary_mlp.parameters(), lr=0.005, weight_decay=1e-3)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    secondary_mlp.train()
    for epoch in range(35):
        for bx, by in loader:
            optimizer.zero_grad()
            loss, _, _ = secondary_mlp.compute_elbo_loss(bx, by, num_mc_samples=3)
            loss.backward()
            optimizer.step()

    # 4. Train Diagnostic Logistic Regression
    diagnostic_logreg = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    diagnostic_logreg.fit(X_train_norm, y_train_np)

    # 5. Fit Source Probability Calibrator on X_cal
    cal_raw_probs = primary_tree.predict_proba(X_cal_norm)[:, 1]
    source_calibrator = SourceProbabilityCalibrator(n_bins=10, n_cv_splits=3, random_state=seed)
    source_calibrator.fit(cal_raw_probs, y_cal_np)

    cal_probs = source_calibrator.predict_proba(cal_raw_probs)
    cal_preds = (cal_probs >= 0.5).astype(int)
    cal_conf = np.maximum(cal_probs, 1.0 - cal_probs)
    cal_acc = float(np.mean(cal_preds == y_cal_np))
    cal_mean_conf = float(np.mean(cal_conf))

    # 6. Compute ATC Threshold & DoC Offset
    atc_threshold = float(np.quantile(cal_conf, max(0.0, 1.0 - cal_acc)))
    doc_offset = float(cal_mean_conf - cal_acc)

    # 7. Lock Mondrian Conformal Quantiles (alpha = 0.10)
    conformal_calibrator = MondrianConformalCalibrator(alpha=0.10)
    conformal_calibrator.fit(cal_probs, y_cal_np)

    # 8. Compute TreeSHAP Feature Reliance Weights on X_cal
    try:
        explainer = shap.TreeExplainer(primary_tree)
        shap_vals = explainer.shap_values(X_cal_norm)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_reliance = np.mean(np.abs(shap_vals), axis=0)
        shap_weights = shap_reliance / max(1e-8, np.sum(shap_reliance))
    except Exception:
        # Fallback to feature importances if TreeSHAP encounters non-standard tree structure
        shap_weights = getattr(primary_tree, "feature_importances_", np.ones(n_features) / n_features)
        shap_weights = shap_weights / np.sum(shap_weights)

    # 9. Compute Source Correlation Matrix on X_cal
    source_corr_matrix = np.corrcoef(X_cal_norm, rowvar=False)

    # 10. Fit D3M Bootstrap Reference Disagreement Distribution
    d3m_calibrator = D3MReferenceCalibrator(
        query_sizes=[10, 20, 50],
        num_bootstrap_rounds=100,
        posterior_samples_k=25,
        temperature=1.0,
        alpha=0.05,
        seed=seed,
    )
    d3m_calibrator.fit(secondary_mlp, X_cal_norm)

    # 11. Compute Calibration Set GdScore Baseline
    secondary_mlp.eval()
    with torch.no_grad():
        h_cal, mu_cal, _ = secondary_mlp(torch.tensor(X_cal_norm, dtype=torch.float32))
        cal_mlp_probs = torch.softmax(mu_cal, dim=-1)[:, 1].cpu().numpy()
        h_cal_np = h_cal.cpu().numpy()
        pseudo_cal = (cal_mlp_probs >= 0.5).astype(float)
        res_cal = np.abs(cal_mlp_probs - pseudo_cal)
        cal_gdscore = float(np.mean(res_cal * np.linalg.norm(h_cal_np, axis=1)))

    return FrozenModelPipeline(
        preprocessor=preprocessor,
        primary_tree=primary_tree,
        secondary_mlp=secondary_mlp,
        diagnostic_logreg=diagnostic_logreg,
        source_calibrator=source_calibrator,
        conformal_calibrator=conformal_calibrator,
        d3m_calibrator=d3m_calibrator,
        feature_names=feature_names,
        atc_threshold=atc_threshold,
        doc_offset=doc_offset,
        shap_weights=shap_weights,
        source_corr_matrix=source_corr_matrix,
        source_cal_acc=cal_acc,
        source_cal_mean_conf=cal_mean_conf,
        source_cal_gdscore=cal_gdscore,
    )
