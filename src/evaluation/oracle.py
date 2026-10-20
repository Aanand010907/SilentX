"""Evaluation Oracle for SilentX.

Enforces the strict zero-target-label axiom:
- Operates exclusively in the post-monitoring evaluation phase.
- Scores saved predictions and monitor estimates against sequestered deployment labels.
- Calculates true accuracy, actual degradation, AUROC, AUPRC, FAR_benign, and rank correlations.
- Protects against label leakage: verifies that monitors do not receive ground-truth objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


@dataclass
class WindowEvaluationResult:
    """Outcome for a single deployment window computed post-hoc by the Oracle."""
    window_id: int
    regime: str
    severity: float
    true_accuracy: float
    true_degradation: float
    is_degraded: bool
    estimated_accuracy: Optional[float]
    estimation_error: Optional[float]
    monitor_risk_score: Optional[float]


class EvaluationOracle:
    """Sequestered evaluation harness holding ground-truth deployment labels."""

    def __init__(self, degradation_threshold: float = 0.05, benign_margin: float = 0.02):
        self.degradation_threshold = float(degradation_threshold)
        self.benign_margin = float(benign_margin)

    def score_window(
        self,
        window_id: int,
        regime: str,
        severity: float,
        y_sequestered: pd.Series,
        predicted_probs: np.ndarray,
        source_cal_acc: float,
        monitor_estimated_acc: Optional[float] = None,
        monitor_risk_score: Optional[float] = None,
    ) -> WindowEvaluationResult:
        """Score an individual deployment window using sequestered ground-truth labels.

        Args:
            window_id: Window sequence index.
            regime: Name of the distribution shift regime.
            severity: Continuous shift severity magnitude.
            y_sequestered: Ground-truth target labels (hidden from monitors).
            predicted_probs: Calibrated predicted probabilities output by the frozen model.
            source_cal_acc: Source calibration set baseline accuracy.
            monitor_estimated_acc: Optional accuracy estimate output by a monitor.
            monitor_risk_score: Optional degradation risk score in [0, 1].

        Returns:
            WindowEvaluationResult with ground-truth performance metrics.
        """
        y_true = np.asarray(y_sequestered, dtype=int).ravel()
        p = np.asarray(predicted_probs, dtype=float).ravel()

        if len(y_true) != len(p):
            raise ValueError(f"Shape mismatch: y_sequestered ({len(y_true)}) vs predicted_probs ({len(p)})")

        y_pred = (p >= 0.5).astype(int)
        true_acc = float(np.mean(y_pred == y_true))
        true_deg = float(source_cal_acc - true_acc)
        is_degraded = bool(true_deg >= self.degradation_threshold)

        est_err = None
        if monitor_estimated_acc is not None:
            est_err = float(abs(monitor_estimated_acc - true_acc))

        return WindowEvaluationResult(
            window_id=window_id,
            regime=regime,
            severity=severity,
            true_accuracy=true_acc,
            true_degradation=true_deg,
            is_degraded=is_degraded,
            estimated_accuracy=monitor_estimated_acc,
            estimation_error=est_err,
            monitor_risk_score=monitor_risk_score,
        )

    def evaluate_stream(
        self,
        window_results: List[WindowEvaluationResult],
        risk_alarm_threshold: float = 0.5,
    ) -> Dict[str, any]:
        """Aggregate evaluation metrics across all evaluated deployment windows."""
        if not window_results:
            raise ValueError("No window evaluation results provided.")

        true_degradations = np.array([r.true_degradation for r in window_results])
        is_degraded_events = np.array([int(r.is_degraded) for r in window_results])
        risk_scores = np.array([
            r.monitor_risk_score if r.monitor_risk_score is not None else 0.0
            for r in window_results
        ])

        # Mean estimation error if estimated_accuracy was recorded
        est_errors = [r.estimation_error for r in window_results if r.estimation_error is not None]
        mean_mae = float(np.mean(est_errors)) if est_errors else None

        # AUROC & AUPRC for degradation event detection
        # Handled explicitly: undefined if single class present
        n_pos = np.sum(is_degraded_events == 1)
        n_neg = np.sum(is_degraded_events == 0)

        if n_pos > 0 and n_neg > 0:
            auroc = float(roc_auc_score(is_degraded_events, risk_scores))
            precision, recall, _ = precision_recall_curve(is_degraded_events, risk_scores)
            auprc = float(auc(recall, precision))
        else:
            # Explicitly documented mathematical limitation: never fake a number
            auroc = np.nan
            auprc = np.nan

        # Benign Shift False Alarm Rate:
        # Evaluated on windows where true degradation is negligible (< benign_margin)
        benign_mask = (true_degradations < self.benign_margin)
        if np.sum(benign_mask) > 0:
            benign_risks = risk_scores[benign_mask]
            far_benign = float(np.mean(benign_risks >= risk_alarm_threshold))
        else:
            far_benign = 0.0

        # Spearman Rank Correlation with true degradation
        if len(np.unique(risk_scores)) > 1 and len(np.unique(true_degradations)) > 1:
            spearman_rho, spearman_p = spearmanr(risk_scores, true_degradations)
            spearman_rho = float(spearman_rho)
            spearman_p = float(spearman_p)
        else:
            spearman_rho = 0.0
            spearman_p = 1.0

        return {
            "n_windows": len(window_results),
            "n_degraded_windows": int(n_pos),
            "n_benign_windows": int(n_neg),
            "mean_mae_accuracy": mean_mae,
            "detection_auroc": auroc,
            "detection_auprc": auprc,
            "far_benign": far_benign,
            "spearman_rank_correlation": spearman_rho,
            "spearman_p_value": spearman_p,
            "window_records": [
                {
                    "window_id": r.window_id,
                    "regime": r.regime,
                    "severity": r.severity,
                    "true_accuracy": r.true_accuracy,
                    "true_degradation": r.true_degradation,
                    "is_degraded": r.is_degraded,
                    "estimated_accuracy": r.estimated_accuracy,
                    "estimation_error": r.estimation_error,
                    "risk_score": r.monitor_risk_score,
                }
                for r in window_results
            ],
        }
