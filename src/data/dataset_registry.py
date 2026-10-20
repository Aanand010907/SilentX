"""Unified Dataset Registry and Data Contract for SilentX.

Enforces:
1. Strict 75/25 stratified source train/calibration split.
2. Isolated deployment windows where labels are held separately for the Oracle.
3. Preprocessing statistics fit strictly on source training data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.synthetic_streams import (
    SyntheticStreamGenerator,
    SUPPORTED_REGIMES,
    ShiftRegime,
)
from src.data.real_datasets import OpenMLElectricityLoader
from src.data.schema_validation import validate_dataset_schema


@dataclass
class DeploymentWindow:
    """An ordered deployment window of observation records."""
    window_id: int
    regime: str
    severity: float
    X: pd.DataFrame
    y: pd.Series  # Sequestered ground-truth labels for Oracle evaluation only
    metadata: Dict[str, any]

    @property
    def unlabeled_features(self) -> pd.DataFrame:
        """Safe view for monitors containing only unlabeled feature columns."""
        return self.X.copy()


@dataclass
class DatasetBundle:
    """The unified data bundle fulfilling the research specification."""
    name: str
    feature_names: List[str]
    target_name: str
    X_source_train: pd.DataFrame
    y_source_train: pd.Series
    X_source_cal: pd.DataFrame
    y_source_cal: pd.Series
    deployment_windows: List[DeploymentWindow]
    metadata: Dict[str, any]


class DatasetRegistry:
    """Registry and loader for all benchmark datasets."""

    @classmethod
    def load_dataset(
        cls,
        name: str = "synthetic_stream",
        source_train_ratio: float = 0.75,
        window_size: int = 500,
        n_source_samples: int = 4000,
        seed: int = 42,
        **kwargs,
    ) -> DatasetBundle:
        """Load and partition a dataset according to the SilentX protocol.

        Args:
            name: Dataset identifier ("synthetic_stream" or "openml_electricity").
            source_train_ratio: Ratio of source data used for training (default 0.75).
            window_size: Size of deployment windows $m$.
            n_source_samples: Total source samples before train/cal split.
            seed: Random seed for deterministic reproducibility.

        Returns:
            DatasetBundle containing source splits and deployment windows.
        """
        if name == "synthetic_stream":
            return cls._load_synthetic(
                source_train_ratio=source_train_ratio,
                window_size=window_size,
                n_source_samples=n_source_samples,
                seed=seed,
                **kwargs,
            )
        elif name == "openml_electricity":
            return cls._load_electricity(
                source_train_ratio=source_train_ratio,
                window_size=window_size,
                seed=seed,
                **kwargs,
            )
        else:
            raise ValueError(f"Dataset '{name}' not yet registered. Supported: ['synthetic_stream', 'openml_electricity']")

    @classmethod
    def _load_synthetic(
        cls,
        source_train_ratio: float = 0.75,
        window_size: int = 500,
        n_source_samples: int = 4000,
        seed: int = 42,
        **kwargs,
    ) -> DatasetBundle:
        gen = SyntheticStreamGenerator(seed=seed)

        # 1. Generate clean in-distribution source dataset
        X_src, y_src, _ = gen.generate_regime(
            n_samples=n_source_samples, regime="in_distribution", severity=0.0, seed=seed
        )

        # 2. Stratified 75/25 split into source_train and source_cal
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_src,
            y_src,
            train_size=source_train_ratio,
            stratify=y_src,
            random_state=seed,
            shuffle=True,
        )

        validate_dataset_schema(
            pd.concat([X_train, y_train], axis=1),
            feature_columns=gen.FEATURE_NAMES,
            target_column=gen.TARGET_COLUMN,
        )

        # 3. Generate sequential deployment windows across all 6 regimes at multiple severities
        deployment_windows: List[DeploymentWindow] = []
        w_id = 0

        # Include ID windows
        for _ in range(2):
            X_w, y_w, meta = gen.generate_regime(
                n_samples=window_size, regime="in_distribution", severity=0.0, seed=seed + 100 + w_id
            )
            deployment_windows.append(DeploymentWindow(
                window_id=w_id, regime="in_distribution", severity=0.0, X=X_w, y=y_w, metadata=meta
            ))
            w_id += 1

        # Shifted regimes
        for regime in ["harmless_drift", "correlation_collapse", "prior_shift", "concept_shift", "confidence_collapse"]:
            for sev in [0.5, 1.0]:
                X_w, y_w, meta = gen.generate_regime(
                    n_samples=window_size, regime=regime, severity=sev, seed=seed + 200 + w_id
                )
                deployment_windows.append(DeploymentWindow(
                    window_id=w_id, regime=regime, severity=sev, X=X_w, y=y_w, metadata=meta
                ))
                w_id += 1

        return DatasetBundle(
            name="synthetic_stream",
            feature_names=gen.FEATURE_NAMES,
            target_name=gen.TARGET_COLUMN,
            X_source_train=X_train.reset_index(drop=True),
            y_source_train=y_train.reset_index(drop=True),
            X_source_cal=X_cal.reset_index(drop=True),
            y_source_cal=y_cal.reset_index(drop=True),
            deployment_windows=deployment_windows,
            metadata={
                "n_source_train": len(X_train),
                "n_source_cal": len(X_cal),
                "n_windows": len(deployment_windows),
                "window_size": window_size,
                "seed": seed,
            },
        )

    @classmethod
    def _load_electricity(
        cls,
        source_train_ratio: float = 0.75,
        window_size: int = 500,
        seed: int = 42,
        **kwargs,
    ) -> DatasetBundle:
        data = OpenMLElectricityLoader.load(source_train_ratio=source_train_ratio)

        X_deploy = data["X_deploy"]
        y_deploy = data["y_deploy"]

        # Partition deployment stream into sequential chronological windows of size m
        deployment_windows: List[DeploymentWindow] = []
        n_windows = len(X_deploy) // window_size
        for w_id in range(min(n_windows, 20)):  # Bounded to 20 windows for fast evaluation
            start = w_id * window_size
            end = start + window_size
            X_w = X_deploy.iloc[start:end].reset_index(drop=True)
            y_w = y_deploy.iloc[start:end].reset_index(drop=True)
            deployment_windows.append(DeploymentWindow(
                window_id=w_id,
                regime="temporal_market_stream",
                severity=float(w_id) / max(1, n_windows),
                X=X_w,
                y=y_w,
                metadata={"start_idx": start, "end_idx": end},
            ))

        return DatasetBundle(
            name="openml_electricity",
            feature_names=data["feature_names"],
            target_name=data["target_name"],
            X_source_train=data["X_train"],
            y_source_train=data["y_train"],
            X_source_cal=data["X_cal"],
            y_source_cal=data["y_cal"],
            deployment_windows=deployment_windows,
            metadata={
                "n_source_train": len(data["X_train"]),
                "n_source_cal": len(data["X_cal"]),
                "n_windows": len(deployment_windows),
                "window_size": window_size,
                "is_surrogate": data.get("is_surrogate", False),
                "seed": seed,
            },
        )
