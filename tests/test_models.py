"""Unit and integration tests for base learners, frozen inference, and VBLL."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.models.frozen_pipeline import (
    FrozenPreprocessor,
    train_and_freeze_pipeline,
)
from src.models.vbll_model import (
    VariationalBayesianLastLayer,
    D3MNeuralClassifier,
    D3MReferenceCalibrator,
)
from src.data.synthetic_streams import generate_synthetic_stream


def test_frozen_preprocessor_is_immutable():
    """Verify preprocessor fits only on train and freezes mean/std."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    X_test = pd.DataFrame({"a": [100.0, 200.0, 300.0], "b": [1000.0, 2000.0, 3000.0]})

    prep = FrozenPreprocessor().fit(X_train)
    # Check fitted statistics
    np.testing.assert_allclose(prep.scaler.mean_, [2.0, 20.0])

    # Transform test set using frozen train statistics
    transformed_test = prep.transform(X_test)
    assert transformed_test.shape == (3, 2)
    # Mean of transformed test must NOT be zero because test has shifted
    assert not np.allclose(transformed_test.mean(axis=0), [0.0, 0.0])


def test_vbll_posterior_sampling_and_elbo():
    """Verify PyTorch VBLL forward pass, Monte Carlo sampling, and ELBO loss."""
    model = D3MNeuralClassifier(in_features=10, hidden_dims=[32, 16], prior_var=1.0)

    x_dummy = torch.randn(20, 10)
    y_dummy = torch.randint(0, 2, (20,))

    # Test forward pass
    h, mu, logvar = model(x_dummy)
    assert h.shape == (20, 16)
    assert mu.shape == (20, 2)
    assert logvar.shape == (20, 2)

    # Test sampling
    samples = model.vbll.sample_logits(mu, logvar, num_samples=15, temperature=1.0)
    assert samples.shape == (15, 20, 2)

    # Test ELBO loss
    total_loss, ce_loss, kl = model.compute_elbo_loss(x_dummy, y_dummy, num_mc_samples=5)
    assert torch.isfinite(total_loss)
    assert torch.isfinite(ce_loss)
    assert torch.isfinite(kl)
    assert kl.item() >= 0.0  # KL divergence is non-negative


def test_pipeline_training_and_inference_integrity():
    """Verify end-to-end frozen pipeline training and inference across all model families."""
    X_train, y_train, _ = generate_synthetic_stream(n_samples=600, seed=42)
    X_cal, y_cal, _ = generate_synthetic_stream(n_samples=200, seed=43)

    pipeline = train_and_freeze_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        feature_names=list(X_train.columns),
        tree_family="xgboost",
        seed=42,
    )

    X_test, _, _ = generate_synthetic_stream(n_samples=100, seed=44)

    # Test primary tree predictions
    p_primary = pipeline.predict_primary_proba(X_test)
    assert p_primary.shape == (100,)
    assert np.all((p_primary >= 0.0) & (p_primary <= 1.0))

    # Test secondary neural probe predictions
    p_secondary = pipeline.predict_secondary_proba(X_test)
    assert p_secondary.shape == (100,)
    assert np.all((p_secondary >= 0.0) & (p_secondary <= 1.0))

    # Test diagnostic logistic regression predictions
    p_diag = pipeline.predict_diagnostic_proba(X_test)
    assert p_diag.shape == (100,)
    assert np.all((p_diag >= 0.0) & (p_diag <= 1.0))

    # Test GdScore calculation
    gdscore = pipeline.compute_gdscore(X_test)
    assert isinstance(gdscore, float)
    assert gdscore >= 0.0

    # Test SHAP reliance weights
    assert len(pipeline.shap_weights) == 10
    np.testing.assert_allclose(np.sum(pipeline.shap_weights), 1.0, atol=1e-5)
