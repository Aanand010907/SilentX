"""Unit and integration tests for data contracts, synthetic streams, and schema validation."""

import pytest
import numpy as np
import pandas as pd

from src.data.synthetic_streams import (
    generate_synthetic_stream,
    SyntheticStreamGenerator,
    SUPPORTED_REGIMES,
)
from src.data.schema_validation import validate_dataset_schema, SchemaValidationError
from src.data.dataset_registry import DatasetRegistry


def test_synthetic_stream_reproducibility():
    """Verify that identical random seeds yield identical synthetic datasets."""
    X1, y1, _ = generate_synthetic_stream(n_samples=500, regime="in_distribution", seed=42)
    X2, y2, _ = generate_synthetic_stream(n_samples=500, regime="in_distribution", seed=42)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_all_six_shift_regimes_generate_valid_data():
    """Verify that all 6 shift regimes execute and produce valid feature shapes and binary targets."""
    for regime in SUPPORTED_REGIMES:
        X, y, meta = generate_synthetic_stream(n_samples=300, regime=regime, severity=1.0, seed=123)

        assert X.shape == (300, 10), f"Regime {regime} failed on shape: {X.shape}"
        assert len(y) == 300
        assert set(np.unique(y)).issubset({0, 1})
        assert not np.isnan(X.to_numpy()).any()
        assert meta["regime"] == regime


def test_schema_validation_catches_corrupted_data():
    """Verify that schema validation detects missing columns, NaNs, and unexpected classes."""
    valid_df = pd.DataFrame({"x_0": [1.0, 2.0], "x_1": [3.0, 4.0], "target": [0, 1]})

    # Valid schema should pass
    validate_dataset_schema(valid_df, feature_columns=["x_0", "x_1"], target_column="target")

    # Missing column
    with pytest.raises(SchemaValidationError, match="Missing required feature"):
        validate_dataset_schema(valid_df, feature_columns=["x_0", "x_missing"], target_column="target")

    # NaN in features
    df_with_nan = valid_df.copy()
    df_with_nan.loc[0, "x_0"] = np.nan
    with pytest.raises(SchemaValidationError, match="unallowed nulls"):
        validate_dataset_schema(df_with_nan, feature_columns=["x_0", "x_1"], target_column="target")

    # Invalid target class
    df_bad_target = pd.DataFrame({"x_0": [1.0, 2.0], "x_1": [3.0, 4.0], "target": [0, 99]})
    with pytest.raises(SchemaValidationError, match="unexpected classes"):
        validate_dataset_schema(
            df_bad_target, feature_columns=["x_0", "x_1"], target_column="target", expected_classes=(0, 1)
        )


def test_dataset_registry_strict_split_proportions():
    """Verify that DatasetRegistry enforces the 75/25 source train/cal ratio and partitions windows."""
    bundle = DatasetRegistry.load_dataset(
        name="synthetic_stream", source_train_ratio=0.75, window_size=200, n_source_samples=2000, seed=42
    )

    total_source = len(bundle.X_source_train) + len(bundle.X_source_cal)
    assert total_source == 2000
    assert len(bundle.X_source_train) == 1500
    assert len(bundle.X_source_cal) == 500

    # Verify stratified target balance across splits
    train_rate = bundle.y_source_train.mean()
    cal_rate = bundle.y_source_cal.mean()
    assert abs(train_rate - cal_rate) < 0.05

    # Check deployment windows integrity
    assert len(bundle.deployment_windows) > 0
    for w in bundle.deployment_windows:
        assert len(w.X) == 200
        assert len(w.y) == 200
        assert not np.isnan(w.X.to_numpy()).any()
