"""Safe Test-Time Adaptation (TTA) with Prior Anchoring and Collapse Recovery.

Implements:
1. Entropy Minimization on unlabeled target batches.
2. Source-prior KL divergence anchor to prevent catastrophic cluster collapse:
   L_TTA = H(p_t) + mu_reg * KL(bar_p_t || pi_source)
3. Parameter isolation: only adapts affine/BatchNorm parameters, freezing backbones.
4. Early abort and automatic rollback if prediction entropy drops below safety bounds.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbll_model import D3MNeuralClassifier


class SafeTestTimeAdapter:
    """Safely adapts neural probe representations on unlabeled target batches."""

    def __init__(
        self,
        neural_probe: D3MNeuralClassifier,
        source_class_priors: np.ndarray,
        mu_reg: float = 0.5,
        steps: int = 4,
        lr: float = 1e-3,
        min_entropy_threshold: float = 0.15,
    ):
        self.base_probe = neural_probe
        self.source_priors = np.asarray(source_class_priors, dtype=float).ravel()
        self.mu_reg = float(mu_reg)
        self.steps = int(steps)
        self.lr = float(lr)
        self.min_entropy_threshold = float(min_entropy_threshold)

    def adapt_batch(
        self,
        X_norm: np.ndarray,
    ) -> Tuple[D3MNeuralClassifier, bool, Dict[str, float]]:
        """Adapts a cloned copy of the neural probe on an unlabeled target batch.

        Args:
            X_norm: Standardized features of the current deployment window.

        Returns:
            adapted_model: Cloned model with adapted affine parameters (or base model if aborted).
            is_adapted: True if adaptation succeeded safely, False if rolled back.
            metrics: Telemetry tracking initial vs adapted batch entropy and loss.
        """
        adapted_model = copy.deepcopy(self.base_probe)
        adapted_model.train()

        # Freeze all layers except BatchNorm parameters
        bn_params = []
        for param in adapted_model.parameters():
            param.requires_grad = False

        for module in adapted_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.train()
                module.requires_grad_(True)
                bn_params.extend(module.parameters())

        if not bn_params or len(X_norm) < 10:
            return self.base_probe, False, {"status": "skipped_insufficient_samples"}

        optimizer = torch.optim.AdamW(bn_params, lr=self.lr, weight_decay=1e-4)
        x_tensor = torch.tensor(X_norm, dtype=torch.float32)
        pi_s = torch.tensor(self.source_priors, dtype=torch.float32)

        initial_entropy = 0.0
        final_entropy = 0.0

        for step in range(self.steps):
            optimizer.zero_grad()
            _, mu, _ = adapted_model(x_tensor)
            probs = torch.softmax(mu, dim=-1)

            # 1. Conditional entropy
            eps = 1e-12
            sample_entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
            batch_entropy = torch.mean(sample_entropy)

            if step == 0:
                initial_entropy = float(batch_entropy.item())

            # 2. Source-prior KL regularization: KL(bar_p || pi_source)
            mean_batch_p = torch.mean(probs, dim=0)
            kl_prior = torch.sum(mean_batch_p * torch.log((mean_batch_p + eps) / (pi_s + eps)))

            loss = batch_entropy + self.mu_reg * kl_prior
            loss.backward()
            optimizer.step()

            final_entropy = float(batch_entropy.item())

        # Safety Check: If entropy collapsed below the threshold, abort and rollback!
        if final_entropy < self.min_entropy_threshold:
            # Overconfident collapse detected -> Reject adaptation
            return self.base_probe, False, {
                "initial_entropy": initial_entropy,
                "final_entropy": final_entropy,
                "status": "aborted_entropy_collapse",
            }

        return adapted_model, True, {
            "initial_entropy": initial_entropy,
            "final_entropy": final_entropy,
            "status": "adapted_successfully",
        }
