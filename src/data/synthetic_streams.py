"""Synthetic distribution shift generator for controlled post-deployment benchmarking.

This module implements the 6 canonical shift regimes used in the SilentX / TrustGuard
stress-test benchmark. It provides exact parametric control over marginal covariate
drift, correlation structures, class priors, and concept mappings.

Identifiability Note:
---------------------
Proposition (Unidentifiability of Pure Unobservable Concept Shift):
If P_t(X) = P_source(X) identically everywhere on X, and the ground-truth labeling
function flips (P_t(Y=1|X) = 1 - P_source(Y=1|X)) without leaving ANY footprint in
covariates, latent representations, or hypothesis disagreements, no label-free algorithm
can detect the accuracy drop because observable inputs and deterministic outputs are
statistically indistinguishable from source calibration.
The regimes below reflect realistic physical, financial, and behavioral shifts where
concept changes co-occur with boundary crossing, broken feature couplings, or epistemic
uncertainty divergence.
"""

from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd

ShiftRegime = Literal[
    "in_distribution",
    "harmless_drift",
    "correlation_collapse",
    "prior_shift",
    "concept_shift",
    "confidence_collapse",
]

SUPPORTED_REGIMES: List[ShiftRegime] = [
    "in_distribution",
    "harmless_drift",
    "correlation_collapse",
    "prior_shift",
    "concept_shift",
    "confidence_collapse",
]


class SyntheticStreamGenerator:
    """Parametric generator for synthetic tabular streams under controlled shift."""

    FEATURE_NAMES = [f"x_{i}" for i in range(10)]
    CAUSAL_FEATURES = ["x_0", "x_1", "x_2"]
    PROXY_FEATURES = ["x_3"]
    NOISE_FEATURES = [f"x_{i}" for i in range(4, 10)]
    TARGET_COLUMN = "target"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_regime(
        self,
        n_samples: int = 2000,
        regime: ShiftRegime = "in_distribution",
        severity: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
        """Generate a synthetic batch under a specified shift regime.

        Args:
            n_samples: Number of observations to generate.
            regime: One of the 6 canonical shift regimes.
            severity: Continuous shift magnitude (0.0 = clean ID, 1.0 = full shift).
            seed: Optional local random seed override.

        Returns:
            features_df: DataFrame of 10 continuous predictors.
            target_series: Binary labels (0 or 1).
            metadata: Shift description and structural parameters.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)

        if regime not in SUPPORTED_REGIMES:
            raise ValueError(f"Unknown regime '{regime}'. Supported: {SUPPORTED_REGIMES}")

        # Base 10 standard normal features:
        # X0, X1, X2: true causal features
        # X3: collinear proxy for X0 (rho ~ 0.85)
        # X4..X9: independent noise features
        X = rng.standard_normal((n_samples, 10))
        X[:, 3] = 0.85 * X[:, 0] + np.sqrt(1.0 - 0.85**2) * rng.standard_normal(n_samples)

        # Baseline structural equation logit
        logits = 2.2 * X[:, 0] - 1.8 * X[:, 1] + 1.4 * (X[:, 0] * X[:, 2])

        metadata = {
            "regime": regime,
            "severity": float(severity),
            "n_samples": n_samples,
            "seed": seed or self.seed,
            "causal_features": self.CAUSAL_FEATURES,
            "proxy_features": self.PROXY_FEATURES,
            "noise_features": self.NOISE_FEATURES,
            "identifiability_limitation_applies": False,
        }

        if regime == "in_distribution":
            # Strictly stationary source regime
            pass

        elif regime == "harmless_drift":
            # Massive mean and variance shift in noise features X5..X9
            # Causal features X0..X2 and true boundary P(Y|X) remain completely intact
            X[:, 5:] += severity * 4.5
            X[:, 5:] *= (1.0 + severity * 1.5)
            metadata["description"] = "Covariate shift isolated to uninformative noise features."

        elif regime == "correlation_collapse":
            # The correlation between proxy X3 and causal driver X0 inverts,
            # pushing instances into off-manifold regions where models misgeneralize
            rho_shifted = 0.85 - 1.7 * severity
            X[:, 3] = rho_shifted * X[:, 0] + np.sqrt(max(0.01, 1.0 - rho_shifted**2)) * rng.standard_normal(n_samples)
            X[:, 0] += severity * 1.2
            # Logit incorporates mild interaction collapse
            logits = (2.2 * (1.0 - 0.5 * severity)) * X[:, 0] - 1.8 * X[:, 1] + 1.4 * (X[:, 0] * X[:, 2])
            metadata["description"] = "Broken feature coupling between proxy X3 and causal driver X0."

        elif regime == "prior_shift":
            # Label prior shift: modulate target base rate while preserving P(X|Y)
            # Achieved by shifting the logit intercept
            intercept_shift = severity * 2.2
            logits = logits + intercept_shift
            metadata["description"] = "Target class prevalence shift via latent intercept modulation."

        elif regime == "concept_shift":
            # Decision boundary rotates: coefficients on X0 and X1 invert sign
            # while marginal X features experience minimal displacement
            X[:, 1] += severity * 0.3
            logits = (
                (2.2 - 4.2 * severity) * X[:, 0]
                + (-1.8 + 3.8 * severity) * X[:, 1]
                - 1.4 * (X[:, 0] * X[:, 2])
            )
            metadata["description"] = "Rotation of the conditional decision hyperplane P(Y|X)."
            metadata["identifiability_limitation_applies"] = (severity > 0.8)

        elif regime == "confidence_collapse":
            # Extreme OOD extrapolation on causal features + inverted ground truth concept.
            # Models trained with axis-aligned splits or hyperplanes extrapolate with
            # saturated near-1.0 confidence while actual labels are completely inverted.
            X[:, 0] += severity * 3.5
            X[:, 1] -= severity * 2.8
            logits = -3.0 * X[:, 0] + 2.5 * X[:, 1]
            metadata["description"] = "Extreme OOD extrapolation on causal features with inverted concept."

        # Compute ground-truth conditional probabilities and binary labels
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        y = (rng.random(n_samples) < probs).astype(int)

        features_df = pd.DataFrame(X, columns=self.FEATURE_NAMES)
        target_series = pd.Series(y, name=self.TARGET_COLUMN)

        return features_df, target_series, metadata


def generate_synthetic_stream(
    n_samples: int = 2000,
    regime: ShiftRegime = "in_distribution",
    severity: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
    """Functional convenience wrapper for SyntheticStreamGenerator."""
    gen = SyntheticStreamGenerator(seed=seed)
    return gen.generate_regime(n_samples=n_samples, regime=regime, severity=severity, seed=seed)
