# SilentX — Experimental Protocol & Evaluation Standards

**Project:** SilentX  
**System:** TrustGuard  
**Version:** 1.0.0 (Stage 1)

---

## 1. Core Principles & Strict Label Isolation

The SilentX evaluation protocol enforces the zero-target-label axiom:

1. **Information Barrier:**
   - The monitoring layer (`src/monitors/`) possesses an interface that takes only unlabeled feature matrices $X_{\text{unlabeled}}$ and frozen model states $\theta_{\text{frozen}}$.
   - The monitor outputs either an estimated accuracy $\widehat{\text{Acc}}(W_t)$ or a continuous degradation risk score $R(W_t) \in [0, 1]$.
   - Under no circumstance does a monitor receive ground-truth deployment labels $y_{\text{deploy}}$, ground-truth error vectors, or target loss values.
2. **Evaluation Oracle Vault (`src/evaluation/oracle.py`):**
   - Ground-truth target labels $Y_{\text{deploy}}$ are sequestered in an evaluation vault.
   - Once all monitoring signals and predictions are committed and logged, the `EvaluationOracle` accesses the sequestered labels to compute true accuracy, actual degradation, estimation error, and detection metrics.
   - Any attempt by monitoring code to access the oracle raises a fatal contract violation.

---

## 2. Evaluation Metrics

### 2.1 True Performance and Degradation
- **Window True Accuracy:**
  $$\text{Acc}(W_t) = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(\hat{y}(x_i) = y_i\right)$$
- **True Accuracy Degradation:**
  $$\Delta \text{Acc}(W_t) = \text{Acc}(D_{\text{cal}}) - \text{Acc}(W_t)$$
- **Binary Degradation Event:**
  $$E_t = \mathbb{I}\left(\Delta \text{Acc}(W_t) \ge \tau_{\text{drop}}\right)$$
  The standard threshold is locked at $\tau_{\text{drop}} = 0.05$ (a 5 percentage point accuracy decline relative to source calibration).

### 2.2 Accuracy Estimation Error (for UAE / CBPE baselines)
- **Mean Absolute Error (MAE):**
  $$\text{MAE}_{\text{acc}} = \frac{1}{K} \sum_{t=1}^{K} \left| \widehat{\text{Acc}}(W_t) - \text{Acc}(W_t) \right|$$

### 2.3 Degradation Detection Performance
- **AUROC and AUPRC:**
  Computed between the monitor's continuous alarm score $R(W_t)$ (or negative estimated accuracy $-\widehat{\text{Acc}}(W_t)$) and the ground-truth degradation event $E_t$ across all deployment windows.
  *Edge Case Rule:* If a deployment stream contains 0 positive degradation events ($E_t = 0 \;\forall t$) or 100% positive events, AUROC/AUPRC is mathematically undefined. In this case, the metric is explicitly logged as `NaN` (or documented with a note) and never replaced with an arbitrary favorable value (e.g. 1.0).

### 2.4 Benign-Shift False Alarm Rate ($\text{FAR}_{\text{benign}}$)
- Evaluated on windows undergoing verified **Harmless Covariate Shift** where $\Delta \text{Acc}(W_t) < 0.02$:
  $$\text{FAR}_{\text{benign}} = \frac{1}{|W_{\text{benign}}|} \sum_{W \in W_{\text{benign}}} \mathbb{I}\left(R(W) > \text{Threshold}_{95}(D_{\text{cal}})\right)$$
  Target requirement for TrustGuard: $\text{FAR}_{\text{benign}} \le 0.05$.

### 2.5 Rank Correlation ($\rho_s$)
- **Spearman Rank Correlation:**
  Measures the monotonic relationship between the monitor's risk score $R(W_t)$ and actual degradation $\Delta \text{Acc}(W_t)$:
  $$\rho_s = \text{SpearmanCorr}\left(\{R(W_t)\}_{t=1}^K, \{\Delta \text{Acc}(W_t)\}_{t=1}^K\right)$$
  Accompanied by 95% bootstrap confidence intervals ($B = 1000$ resamples).

### 2.6 Expected Calibration Error (ECE)
- Computed by binning predicted probabilities into $M=10$ equal-width bins $B_m \subset (0, 1]$:
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 3. Data Splitting & Seed Management

1. **Source Train / Calibration Split:**
   - 75% source training split for fitting models and preprocessing transformers.
   - 25% held-out source calibration split for probability calibration, quantile locking, and reference statistics.
   - Stratified by target class.
2. **Replication Seeds:**
   - Experiments are executed across three independent seeds: `42`, `123`, `999`.
   - All random generators (NumPy, PyTorch, Scikit-Learn, XGBoost) are explicitly seeded at every step.
3. **Statistical Significance Testing:**
   - Pairwise method comparisons use the two-sided Wilcoxon signed-rank test across windows and seeds.
   - Multiple hypothesis testing across estimators is corrected using the Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$.
