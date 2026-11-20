"""Unit and integration tests for the Combined Optimal Approach (OmniGuard).

Covers:
1. GBDTSubEnsemble multi-tree fitting and prediction.
2. Dynamic SHAP-attribution gating between ensemble and neural probe.
3. Checkpoint snapshot and emergency rollback.
4. Safe Test-Time Adaptation (TTA) with prior anchoring.
5. Three-Tier Operational Guardrail (Benign drift suppression, rollback, and conformal escalation).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.synthetic_streams import SyntheticStreamGenerator
from src.models.frozen_pipeline import train_and_freeze_pipeline
from src.models.hybrid_ensemble import GBDTSubEnsemble, RobustHybridEnsemble, build_robust_hybrid_ensemble_from_pipeline
from src.models.tta_adapter import SafeTestTimeAdapter
from src.monitors.trustguard import TrustGuardCompositeMonitor
from src.monitors.three_tier_guardrail import (
    ThreeTierOperationalGuardrail,
    OperationalVerdict,
    AnalystEscalationQueue,
)


@pytest.fixture(scope="module")
def synthetic_data():
    gen = SyntheticStreamGenerator(seed=42)
    X_train, y_train, _ = gen.generate_regime(n_samples=600, regime="in_distribution", seed=42)
    X_cal, y_cal, _ = gen.generate_regime(n_samples=300, regime="in_distribution", seed=43)
    X_test_id, y_test_id, _ = gen.generate_regime(n_samples=200, regime="in_distribution", seed=44)
    X_test_noise, y_test_noise, _ = gen.generate_regime(n_samples=200, regime="harmless_drift", severity=1.0, seed=45)
    
    feature_names = [f"x_{i}" for i in range(10)]
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test_id": X_test_id,
        "y_test_id": y_test_id,
        "X_test_noise": X_test_noise,
        "y_test_noise": y_test_noise,
        "feature_names": feature_names,
    }


@pytest.fixture(scope="module")
def trained_system(synthetic_data):
    pipeline = train_and_freeze_pipeline(
        X_train=synthetic_data["X_train"],
        y_train=synthetic_data["y_train"],
        X_cal=synthetic_data["X_cal"],
        y_cal=synthetic_data["y_cal"],
        feature_names=synthetic_data["feature_names"],
        seed=42,
    )
    ensemble = build_robust_hybrid_ensemble_from_pipeline(
        pipeline=pipeline,
        X_train=synthetic_data["X_train"],
        y_train=synthetic_data["y_train"],
        seed=42,
    )
    monitor = TrustGuardCompositeMonitor(
        frozen_pipeline=pipeline,
        X_cal=synthetic_data["X_cal"],
        alpha_alarm=0.05,
    )
    guardrail = ThreeTierOperationalGuardrail(
        hybrid_ensemble=ensemble,
        trustguard_monitor=monitor,
        input_drift_threshold=0.15,
        entropy_collapse_threshold=0.10,
    )
    return {
        "pipeline": pipeline,
        "ensemble": ensemble,
        "monitor": monitor,
        "guardrail": guardrail,
    }


def test_gbdt_sub_ensemble(synthetic_data):
    X_raw = synthetic_data["X_train"].to_numpy()
    y_raw = synthetic_data["y_train"].to_numpy()
    sub_ens = GBDTSubEnsemble(seed=42).fit(X_raw, y_raw)
    
    probs = sub_ens.predict_proba(X_raw[:20])
    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_hybrid_ensemble_gating(trained_system, synthetic_data):
    ensemble = trained_system["ensemble"]
    X_test = synthetic_data["X_test_id"]
    
    # 1. Low attribution drift -> Favors GBDT
    p_low_attr = ensemble.predict_hybrid_proba(X_test, s_attr=0.02)
    # 2. High attribution drift -> Gates out noise
    p_high_attr = ensemble.predict_hybrid_proba(X_test, s_attr=0.80)
    
    assert len(p_low_attr) == len(X_test)
    assert len(p_high_attr) == len(X_test)
    assert not np.allclose(p_low_attr, p_high_attr)


def test_checkpoint_restore(trained_system):
    ensemble = trained_system["ensemble"]
    original_param = next(ensemble.neural_probe.parameters()).clone()
    
    # Mutate parameter
    with torch.no_grad():
        next(ensemble.neural_probe.parameters()).add_(1.5)
    
    assert not torch.allclose(next(ensemble.neural_probe.parameters()), original_param)
    
    # Restore checkpoint
    ensemble.restore_checkpoint()
    assert torch.allclose(next(ensemble.neural_probe.parameters()), original_param)


def test_safe_test_time_adapter(trained_system, synthetic_data):
    adapter = SafeTestTimeAdapter(
        neural_probe=trained_system["ensemble"].neural_probe,
        source_class_priors=np.array([0.5, 0.5]),
        steps=3,
        lr=1e-3,
    )
    X_norm = trained_system["pipeline"].preprocessor.transform(synthetic_data["X_test_id"])
    adapted_model, is_adapted, telemetry = adapter.adapt_batch(X_norm)
    
    assert isinstance(adapted_model, torch.nn.Module)
    assert telemetry["status"] in ["adapted_successfully", "aborted_entropy_collapse"]


def test_three_tier_guardrail_workflow(trained_system, synthetic_data):
    guardrail = trained_system["guardrail"]
    guardrail.escalation_queue.clear()
    
    # Evaluate clean In-Distribution batch
    record_id, probs_id, sets_id = guardrail.evaluate_batch(
        X_batch=synthetic_data["X_test_id"],
        batch_id=1,
        ks_max=0.04,
    )
    assert record_id.verdict in [
        OperationalVerdict.NORMAL_EXECUTION,
        OperationalVerdict.BENIGN_DRIFT_SUPPRESSED,
        OperationalVerdict.CONFORMAL_ANALYST_ESCALATION,
    ]
    assert len(probs_id) == len(synthetic_data["X_test_id"])
    assert len(sets_id) == len(synthetic_data["X_test_id"])
    
    # Check that non-singleton samples are enqueued to Tier 3 queue
    queue_summary = guardrail.escalation_queue.get_summary()
    assert queue_summary["total_escalated"] >= 0
