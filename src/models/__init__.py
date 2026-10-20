"""Model families, calibration, and frozen inference pipelines for SilentX."""

from src.models.calibration import (
    compute_expected_calibration_error,
    SourceProbabilityCalibrator,
    MondrianConformalCalibrator,
)
from src.models.vbll_model import (
    VariationalBayesianLastLayer,
    TabularNeuralBackbone,
    D3MNeuralClassifier,
    D3MReferenceCalibrator,
)
from src.models.frozen_pipeline import (
    FrozenPreprocessor,
    FrozenModelPipeline,
    train_and_freeze_pipeline,
)

__all__ = [
    "compute_expected_calibration_error",
    "SourceProbabilityCalibrator",
    "MondrianConformalCalibrator",
    "VariationalBayesianLastLayer",
    "TabularNeuralBackbone",
    "D3MNeuralClassifier",
    "D3MReferenceCalibrator",
    "FrozenPreprocessor",
    "FrozenModelPipeline",
    "train_and_freeze_pipeline",
]
