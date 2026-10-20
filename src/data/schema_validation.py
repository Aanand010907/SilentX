"""Schema validation utilities for SilentX datasets."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when data violates structural or statistical invariants."""
    pass


def validate_dataset_schema(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: Optional[str] = None,
    expected_classes: Optional[Tuple[int, ...]] = (0, 1),
    allow_missing: bool = False,
) -> None:
    """Validate tabular dataset schema, feature integrity, and target distribution.

    Args:
        data: DataFrame to validate.
        feature_columns: List of feature column names.
        target_column: Optional target column name.
        expected_classes: Optional tuple of expected discrete class labels.
        allow_missing: Whether NaN values are permitted in feature columns.

    Raises:
        SchemaValidationError: If any validation rule is violated.
    """
    if not isinstance(data, pd.DataFrame):
        raise SchemaValidationError(f"Expected pandas DataFrame, got {type(data).__name__}")

    if data.empty:
        raise SchemaValidationError("Dataset is empty (0 rows).")

    # Validate feature columns existence
    missing_cols = [c for c in feature_columns if c not in data.columns]
    if missing_cols:
        raise SchemaValidationError(f"Missing required feature columns: {missing_cols}")

    # Check for missing values
    if not allow_missing:
        null_counts = data[feature_columns].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise SchemaValidationError(
                f"Features contain unallowed nulls: {cols_with_nulls.to_dict()}"
            )

    # Check feature numerical finite values
    for col in feature_columns:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            if np.isinf(series.to_numpy(dtype=float)).any():
                raise SchemaValidationError(f"Feature '{col}' contains infinite values.")
        else:
            # Check categorical or string types
            if not pd.api.types.is_categorical_dtype(series) and not pd.api.types.is_object_dtype(series):
                raise SchemaValidationError(
                    f"Feature '{col}' has unsupported dtype {series.dtype}."
                )

    # Validate target column if provided
    if target_column is not None:
        if target_column not in data.columns:
            raise SchemaValidationError(f"Target column '{target_column}' missing from DataFrame.")

        target = data[target_column]
        if target.isnull().any():
            raise SchemaValidationError(f"Target column '{target_column}' contains null values.")

        if expected_classes is not None:
            unique_classes = np.unique(target)
            invalid_classes = set(unique_classes) - set(expected_classes)
            if invalid_classes:
                raise SchemaValidationError(
                    f"Target column '{target_column}' contains unexpected classes: {invalid_classes}. "
                    f"Expected subset of: {expected_classes}"
                )
