"""Variational Bayesian Last Layer (VBLL) and D3M Reference Calibration for SilentX.

Implements the genuine probabilistic formulation from Nguyen et al. (NeurIPS 2025):
- Feature extractor backbone followed by Gaussian variational posterior over logits q(z|x).
- Trained via Monte Carlo ELBO loss with analytical KL divergence against Gaussian prior p(z).
- Temperature-scaled posterior sampling and bootstrap reference calibration of maximum
  posterior disagreement rate Phi.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalBayesianLastLayer(nn.Module):
    """Variational Bayesian Last Layer (VBLL) producing Gaussian variational posterior logits."""

    def __init__(self, in_features: int, num_classes: int = 2, prior_var: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.prior_var = float(prior_var)

        # Mean projection: W_mu, b_mu
        self.mu_linear = nn.Linear(in_features, num_classes)
        # Log-variance projection: W_logvar, b_logvar (diagonal Gaussian parameterization)
        self.logvar_linear = nn.Linear(in_features, num_classes)

        # Initialize log-variance to small negative value (variance ~ 0.1)
        nn.init.constant_(self.logvar_linear.weight, 0.0)
        nn.init.constant_(self.logvar_linear.bias, -2.0)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and log-variance parameters for logits.

        Args:
            h: Penultimate feature tensor of shape (batch_size, in_features).

        Returns:
            mu: Logit posterior mean (batch_size, num_classes).
            logvar: Logit posterior log-variance (batch_size, num_classes).
        """
        mu = self.mu_linear(h)
        logvar = torch.clamp(self.logvar_linear(h), min=-10.0, max=5.0)
        return mu, logvar

    def sample_logits(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Draw Monte Carlo logit samples via the reparameterization trick.

        Returns:
            logits_samples: Tensor of shape (num_samples, batch_size, num_classes).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *mu.shape), device=mu.device, dtype=mu.dtype)
        # Scaled by temperature
        samples = (mu.unsqueeze(0) + std.unsqueeze(0) * eps) / max(temperature, 1e-6)
        return samples

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute analytical KL divergence between diagonal Gaussian q(z) and prior p(z) ~ N(0, sigma_0^2 I).

        KL(q || p) = 0.5 * sum( (var + mu^2)/sigma_0^2 - 1 - log(var/sigma_0^2) )
        """
        var = torch.exp(logvar)
        prior_var = self.prior_var
        kl = 0.5 * torch.sum(
            (var + mu**2) / prior_var - 1.0 - (logvar - np.log(prior_var)),
            dim=-1,
        )
        return kl.mean()


class TabularNeuralBackbone(nn.Module):
    """Tabular MLP feature extractor equipped with BatchNorm and Dropout layers."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.penultimate_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class D3MNeuralClassifier(nn.Module):
    """Complete D3M model integrating TabularNeuralBackbone with VariationalBayesianLastLayer."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        prior_var: float = 1.0,
    ):
        super().__init__()
        self.backbone = TabularNeuralBackbone(
            in_features=in_features, hidden_dims=hidden_dims, dropout_rate=dropout_rate
        )
        self.vbll = VariationalBayesianLastLayer(
            in_features=self.backbone.penultimate_dim, num_classes=2, prior_var=prior_var
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning penultimate representation and variational logit parameters."""
        h = self.backbone(x)
        mu, logvar = self.vbll(h)
        return h, mu, logvar

    def compute_elbo_loss(
        self, x: torch.Tensor, y: torch.Tensor, num_mc_samples: int = 5, beta_kl: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute negative ELBO loss = - E_{q}[log p(y|z)] + beta * KL[q || p]."""
        _, mu, logvar = self.forward(x)
        # Sample logits
        logit_samples = self.vbll.sample_logits(mu, logvar, num_samples=num_mc_samples)
        # Monte Carlo cross entropy
        y_expanded = y.unsqueeze(0).expand(num_mc_samples, -1)
        ce_loss = F.cross_entropy(
            logit_samples.view(-1, 2), y_expanded.reshape(-1), reduction="mean"
        )
        kl = self.vbll.kl_divergence(mu, logvar)
        total_loss = ce_loss + beta_kl * kl
        return total_loss, ce_loss, kl


class D3MReferenceCalibrator:
    """Bootstrap reference calibrator for D3M maximum posterior disagreement rate.

    Preserves the exact protocol from Nguyen et al. (NeurIPS 2025):
    - Evaluates query batch sizes m in {10, 20, 50}.
    - Samples K posterior predictions with temperature tau.
    - Locks the (1 - alpha) quantile of bootstrap disagreement rates Phi as the reference alarm threshold.
    """

    def __init__(
        self,
        query_sizes: List[int] = [10, 20, 50],
        num_bootstrap_rounds: int = 100,
        posterior_samples_k: int = 25,
        temperature: float = 1.0,
        alpha: float = 0.05,
        seed: int = 42,
    ):
        self.query_sizes = query_sizes
        self.num_bootstrap_rounds = num_bootstrap_rounds
        self.k = posterior_samples_k
        self.temperature = temperature
        self.alpha = alpha
        self.seed = seed
        self.thresholds_: Dict[int, float] = {}
        self.reference_distributions_: Dict[int, np.ndarray] = {}

    def fit(self, model: D3MNeuralClassifier, X_cal: np.ndarray) -> "D3MReferenceCalibrator":
        """Compute bootstrap reference disagreement distributions and thresholds on source calibration split."""
        model.eval()
        device = next(model.parameters()).device
        rng = np.random.default_rng(self.seed)

        X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32, device=device)
        n_cal = len(X_cal)

        with torch.no_grad():
            _, mu_all, logvar_all = model(X_cal_tensor)

        mu_np = mu_all.cpu().numpy()
        logvar_np = logvar_all.cpu().numpy()

        for m in self.query_sizes:
            phi_rounds = []
            for _ in range(self.num_bootstrap_rounds):
                # Draw bootstrap batch of query size m
                idx = rng.choice(n_cal, size=m, replace=True)
                batch_mu = torch.tensor(mu_np[idx], device=device)
                batch_logvar = torch.tensor(logvar_np[idx], device=device)

                # Compute posterior predictive pseudo-label y_bar
                # Expectation of softmax across logits
                samples = model.vbll.sample_logits(
                    batch_mu, batch_logvar, num_samples=self.k, temperature=self.temperature
                )
                probs = F.softmax(samples, dim=-1)  # (K, m, 2)
                expected_probs = probs.mean(dim=0)  # (m, 2)
                y_bar = expected_probs.argmax(dim=-1).cpu().numpy()  # (m,)

                # Draw categorical sample predictions for each of the K posterior passes
                # Sample categorical using uniform noise trick
                probs_np = probs.cpu().numpy()
                disagreement_rates = []
                for k_idx in range(self.k):
                    p_k = probs_np[k_idx]
                    y_k = (rng.random(m) < p_k[:, 1]).astype(int)
                    dis_rate = np.mean(y_k != y_bar)
                    disagreement_rates.append(dis_rate)

                phi_t = float(np.max(disagreement_rates))
                phi_rounds.append(phi_t)

            phi_arr = np.array(phi_rounds)
            self.reference_distributions_[m] = phi_arr
            # 1 - alpha quantile (e.g. 95th percentile)
            threshold = float(np.quantile(phi_arr, 1.0 - self.alpha))
            self.thresholds_[m] = threshold

        return self

    def evaluate_batch(
        self, model: D3MNeuralClassifier, X_batch: np.ndarray, query_size: Optional[int] = None
    ) -> Dict[str, any]:
        """Compute D3M posterior disagreement rate on an unlabeled deployment batch."""
        model.eval()
        device = next(model.parameters()).device
        m = len(X_batch)
        assigned_m = query_size or min(self.query_sizes, key=lambda q: abs(q - m))

        X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
        rng = np.random.default_rng(self.seed + m)

        with torch.no_grad():
            _, mu, logvar = model(X_tensor)
            samples = model.vbll.sample_logits(
                mu, logvar, num_samples=self.k, temperature=self.temperature
            )
            probs = F.softmax(samples, dim=-1)
            expected_probs = probs.mean(dim=0)
            y_bar = expected_probs.argmax(dim=-1).cpu().numpy()

            probs_np = probs.cpu().numpy()
            disagreement_rates = []
            for k_idx in range(self.k):
                p_k = probs_np[k_idx]
                y_k = (rng.random(m) < p_k[:, 1]).astype(int)
                disagreement_rates.append(np.mean(y_k != y_bar))

            phi_batch = float(np.max(disagreement_rates))

        threshold = self.thresholds_.get(assigned_m, 0.20)
        is_alert = (phi_batch >= threshold)

        return {
            "disagreement_rate": phi_batch,
            "threshold_95": threshold,
            "is_alert": bool(is_alert),
            "matched_query_size": assigned_m,
        }
