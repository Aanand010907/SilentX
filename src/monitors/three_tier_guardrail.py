"""Three-Tier Operational Guardrail Architecture for SilentX.

Integrates:
- Tier 1: Benign Covariate Drift Gating (suppresses false alarms on harmless noise using S_attr & S_grad).
- Tier 2: Collapse-Safe TTA & Emergency Checkpoint Rollback.
- Tier 3: Conformal Prediction Triage & Analyst Escalation Queue for non-singleton and OOD samples.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.models.hybrid_ensemble import RobustHybridEnsemble


class OperationalVerdict(Enum):
    NORMAL_EXECUTION = "NORMAL_EXECUTION"
    BENIGN_DRIFT_SUPPRESSED = "BENIGN_DRIFT_SUPPRESSED"
    GENUINE_DETERIORATION_ALERT = "GENUINE_DETERIORATION_ALERT"
    COLLAPSE_ROLLBACK_TRIGGERED = "COLLAPSE_ROLLBACK_TRIGGERED"
    CONFORMAL_ANALYST_ESCALATION = "CONFORMAL_ANALYST_ESCALATION"


@dataclass
class EscalatedSample:
    batch_id: int
    sample_index: int
    predicted_set: List[int]
    set_size: int
    predicted_probability: float
    is_severe_ood: bool
    features: Dict[str, float]


class AnalystEscalationQueue:
    """In-memory triage queue holding ambiguous or OOD samples for human analyst review."""

    def __init__(self, max_capacity: int = 5000):
        self.max_capacity = max_capacity
        self.queue: List[EscalatedSample] = []

    def enqueue(self, sample: EscalatedSample) -> None:
        if len(self.queue) < self.max_capacity:
            self.queue.append(sample)

    def get_summary(self) -> Dict[str, Any]:
        if not self.queue:
            return {"total_escalated": 0, "severe_ood_count": 0, "multi_label_count": 0, "empty_set_count": 0}
        sizes = [s.set_size for s in self.queue]
        return {
            "total_escalated": len(self.queue),
            "severe_ood_count": sum(1 for s in self.queue if s.is_severe_ood),
            "multi_label_count": sum(1 for s in sizes if s > 1),
            "empty_set_count": sum(1 for s in sizes if s == 0),
        }

    def clear(self) -> None:
        self.queue.clear()


@dataclass
class ThreeTierRecord:
    batch_id: int
    n_samples: int
    verdict: OperationalVerdict
    risk_tg: float
    alarm_threshold: float
    tier1_input_drift_flag: bool
    tier1_benign_suppressed: bool
    tier2_rollback_triggered: bool
    tier3_abstention_rate: float
    tier3_escalated_count: int
    signals: Dict[str, float] = field(default_factory=dict)


class ThreeTierOperationalGuardrail:
    """The unified three-tier closed-loop operational guardrail."""

    def __init__(
        self,
        hybrid_ensemble: RobustHybridEnsemble,
        trustguard_monitor: TrustGuardCompositeMonitor,
        input_drift_threshold: float = 0.20,
        entropy_collapse_threshold: float = 0.15,
        max_abstention_tolerance: float = 0.35,
    ):
        self.ensemble = hybrid_ensemble
        self.monitor = trustguard_monitor
        self.input_drift_threshold = float(input_drift_threshold)
        self.entropy_collapse_threshold = float(entropy_collapse_threshold)
        self.max_abstention_tolerance = float(max_abstention_tolerance)
        self.escalation_queue = AnalystEscalationQueue()

    def evaluate_batch(
        self,
        X_batch: pd.DataFrame,
        batch_id: int = 0,
        ks_max: float = 0.0,
    ) -> Tuple[ThreeTierRecord, np.ndarray, List[List[int]]]:
        """Evaluates an unlabeled batch through all three operational tiers in sequence.

        Returns:
            record: Detailed decision audit record.
            hybrid_probs: Dynamically blended prediction probabilities.
            conformal_sets: Conformal prediction sets per sample.
        """
        m = len(X_batch)
        if m == 0:
            raise ValueError("Batch cannot be empty.")

        # 1. Compute TrustGuard multi-signal vital signs
        tg_eval = self.monitor.evaluate_window(X_batch)
        s_attr = float(tg_eval["s_attr"])
        s_dis = float(tg_eval["s_dis"])
        s_conf = float(tg_eval["s_conf"])
        s_grad = float(tg_eval["s_grad"])
        s_d3m = float(tg_eval["s_d3m"])
        risk_tg = float(tg_eval["risk_tg"])
        is_tg_alarm = bool(tg_eval["is_alarm"])

        # 2. Get dynamic hybrid predictions using attribution gating
        hybrid_probs = self.ensemble.predict_hybrid_proba(X_batch, s_attr=s_attr)
        conformal_sets = self.ensemble.predict_sets(hybrid_probs)

        # Compute batch entropy
        eps = 1e-12
        p1 = np.clip(hybrid_probs, eps, 1.0 - eps)
        p0 = 1.0 - p1
        sample_entropy = -(p0 * np.log(p0) + p1 * np.log(p1))
        batch_entropy = float(np.mean(sample_entropy))

        # --- Tier 1: Benign Covariate Drift Gating ---
        tier1_input_flag = bool(ks_max >= self.input_drift_threshold)
        tier1_benign_suppressed = False
        tier1_genuine_alarm = False

        if tier1_input_flag:
            # If input drifted, but internal excess risk is below alarm threshold:
            if not is_tg_alarm and s_attr < 0.35 and s_grad < 0.10:
                tier1_benign_suppressed = True
            else:
                tier1_genuine_alarm = True
        elif is_tg_alarm:
            tier1_genuine_alarm = True

        # --- Tier 2: Collapse-Safe TTA & Emergency Checkpoint Rollback ---
        tier2_rollback = False
        if batch_entropy < self.entropy_collapse_threshold:
            # Overconfident collapse detected!
            tier2_rollback = True
            self.ensemble.restore_checkpoint()
        elif is_tg_alarm and (s_d3m >= 0.40) and (np.mean(hybrid_probs > 0.95) > 0.60):
            # Deceptive overconfidence signature: internal jitter with extreme probabilities
            tier2_rollback = True
            self.ensemble.restore_checkpoint()

        # --- Tier 3: Conformal Safety Gating & Analyst Escalation ---
        set_sizes = [len(s) for s in conformal_sets]
        non_singleton_indices = [i for i, sz in enumerate(set_sizes) if sz != 1]
        abstention_rate = float(len(non_singleton_indices) / max(m, 1))

        for idx in non_singleton_indices:
            row_features = X_batch.iloc[idx].to_dict()
            sz = set_sizes[idx]
            self.escalation_queue.enqueue(
                EscalatedSample(
                    batch_id=batch_id,
                    sample_index=idx,
                    predicted_set=conformal_sets[idx],
                    set_size=sz,
                    predicted_probability=float(hybrid_probs[idx]),
                    is_severe_ood=(sz == 0 or sz >= 3),
                    features=row_features,
                )
            )

        # Determine Final Operational Verdict
        if tier2_rollback:
            verdict = OperationalVerdict.COLLAPSE_ROLLBACK_TRIGGERED
        elif tier1_genuine_alarm:
            verdict = OperationalVerdict.GENUINE_DETERIORATION_ALERT
        elif tier1_benign_suppressed:
            verdict = OperationalVerdict.BENIGN_DRIFT_SUPPRESSED
        elif abstention_rate > self.max_abstention_tolerance:
            verdict = OperationalVerdict.CONFORMAL_ANALYST_ESCALATION
        else:
            verdict = OperationalVerdict.NORMAL_EXECUTION

        record = ThreeTierRecord(
            batch_id=batch_id,
            n_samples=m,
            verdict=verdict,
            risk_tg=risk_tg,
            alarm_threshold=self.monitor.alarm_threshold,
            tier1_input_drift_flag=tier1_input_flag,
            tier1_benign_suppressed=tier1_benign_suppressed,
            tier2_rollback_triggered=tier2_rollback,
            tier3_abstention_rate=abstention_rate,
            tier3_escalated_count=len(non_singleton_indices),
            signals={
                "s_attr": s_attr,
                "s_dis": s_dis,
                "s_conf": s_conf,
                "s_grad": s_grad,
                "s_d3m": s_d3m,
                "batch_entropy": batch_entropy,
                "ks_max": ks_max,
            },
        )

        return record, hybrid_probs, conformal_sets
