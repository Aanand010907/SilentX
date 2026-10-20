# SilentX — Four-Stage, 16-Week Implementation Roadmap

**Project:** SilentX  
**System:** TrustGuard Hybrid Monitoring Architecture  
**Execution Horizon:** October 2026 – February 2027  
**Current Milestone:** Stage 1 (Weeks 1–4)

---

## High-Level Program Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    16-WEEK RESEARCH ROADMAP                                        |
+----------------------------------------------------------------------------------------------------+
| Stage 1 (Weeks 1-4)  | Infrastructure, Schema Harmonization, and Base Calibration                  |
|                      | -> Data contracts, 75/25 source splits, frozen pipelines, ECE calibration,  |
|                      |    D3M bootstrap reference, finite-sample Mondrian conformal locking        |
+----------------------+-----------------------------------------------------------------------------+
| Stage 2 (Weeks 5-8)  | Label-Free Estimators & Model-Aware Signals                                 |
|                      | -> AC, DoC, ATC, CBPE, COTT, Input drift (KS/PSI/JSD), Attribution L3,     |
|                      |    Cross-Architecture Disagreement, D3M, AETTA (PDD+Entropy), GdScore      |
+----------------------+-----------------------------------------------------------------------------+
| Stage 3 (Weeks 9-12) | Test-Time Adaptation & Oracle Selection Audit                               |
|                      | -> Marginal Quantile, BatchNorm/Entropy TTA, Filtered Pseudo-Labels,        |
|                      |    Saerens EM, Candidate Pool Selection Regret vs. Sequestered Oracle      |
+----------------------+-----------------------------------------------------------------------------+
| Stage 4 (Weeks 13-16)| TrustGuard Guardrail, Ablations & Governance Synthesis                       |
|                      | -> Three-tier operational guardrail (benign suppression, rollback,          |
|                      |    conformal escalation), Sensitivity/Ablation analyses, NIST AI 800-4 map  |
+----------------------------------------------------------------------------------------------------+
```

---

## Stage 1 — Infrastructure, Schema Harmonization, and Base Calibration (Weeks 1–4)
*Focus of Current Execution*

### 1.1 Objectives
1. Establish a hermetic, reproducible local research environment running Python 3.11+ with pinned dependencies and Docker support.
2. Build a unified data architecture (`DatasetRegistry`) enforcing strict source/deployment isolation, deterministic splitting (75% train / 25% calibration), schema validation, and frozen preprocessing.
3. Implement and lock the primary classifier families:
   - Primary: Gradient-Boosted Trees (HistGradientBoosting / XGBoost).
   - Secondary / Inductive Probe: Tabular Neural Network (MLP) with BatchNorm and Dropout layers.
   - Diagnostic: $L_2$-regularized Logistic Regression.
   - Variational Bayesian Last Layer (VBLL) interface for D3M.
4. Execute source-only probability calibration using 3-fold stratified cross-validation on the held-out 25% reference split, comparing raw vs isotonic Expected Calibration Error (ECE).
5. Compute and lock source reference statistics:
   - ATC source error threshold $t^*$.
   - Mondrian split-conformal per-class non-conformity quantiles $\hat{q}_0, \hat{q}_1$ at $\alpha = 0.10$ with exact finite-sample correction.
   - D3M bootstrap disagreement reference distribution $\Phi$ for query sizes $m \in \{10, 20, 50\}$.
   - Feature attribution reliance weights $w_j$ via SHAP.
6. Verify strict ground-truth isolation between the monitoring layer and evaluation oracle.

### 1.2 Expected Artifacts
- `src/data/synthetic_streams.py`: 6 controlled synthetic shift regimes.
- `src/data/dataset_registry.py`: Unified dataset loading interface with metadata contracts.
- `src/data/real_datasets.py`: OpenML Electricity, TableShift interface stubs, and CIC-IDS schema normalizer.
- `src/models/frozen_pipeline.py`: Frozen preprocessing transformer and multi-family model wrappers.
- `src/models/calibration.py`: ECE calculator, 3-fold isotonic validator, and Mondrian conformal calibrator.
- `src/models/vbll_model.py`: PyTorch Variational Bayesian Last Layer with ELBO loss and posterior logit sampling.
- `src/evaluation/oracle.py`: Sequestered evaluation oracle enforcing strict label isolation.
- `configs/experiment.yaml`: Experiment configuration schema.
- `run_phase1.py` & `run_poc.py`: Executable Stage 1 calibration pipeline and fast smoke test.
- Comprehensive behavioral test suite in `tests/`.

### 1.3 Acceptance Criteria & Exit Conditions
- All unit and integration tests in `tests/` pass with zero failures.
- No deployment target labels can be accessed by monitor functions (verified by automated boundary penetration tests).
- Preprocessing statistics (means, scales, encodings) are strictly fit on the 75% source training split and frozen.
- Calibration decisions (raw vs isotonic) are logged with empirical ECE proof.
- Finite-sample Mondrian conformal quantiles achieve empirical coverage within nominal tolerance on source validation splits.
- D3M bootstrap reference distributions are generated and serialized for batch sizes $m \in \{10, 20, 50\}$.

---

## Stage 2 — Label-Free Estimators and Model-Aware Signals (Weeks 5–8)

### 2.1 Objectives
1. Implement and benchmark foundational baseline estimators:
   - Confidence proxies: Average Confidence (AC), Difference of Confidences (DoC), Average Thresholded Confidence (ATC), and Confidence Optimal Transport with Thresholding (COTT).
   - Expected confusion matrix estimation via NannyML CBPE (Poisson binomial formulation) for Accuracy, F1, and ROC-AUC.
   - Traditional univariate input drift baselines: Maximum Kolmogorov-Smirnov ($\text{KS}_{\max}$), Population Stability Index ($\text{PSI}_{\max}$), and Jensen-Shannon Divergence ($\text{JSD}$).
2. Implement model-aware structural signals:
   - Attribution-Weighted $L_3$ Input & Correlation Drift ($S_{\text{attr}}$) with SHAP importance rank correlation.
   - Cross-Architecture Inductive Disagreement ($S_{\text{dis}}$) pairing tree ensembles against tabular neural networks with Jensen-Shannon Divergence.
   - D3M Bayesian posterior disagreement tracking against the reference quantile $\text{Quantile}_{0.95}(\Phi)$.
   - AETTA Test-Time Dropout Prediction Disagreement (PDD) with batch prediction entropy weighting $E_{\text{avg}}$ and EMA smoothing.
   - Classification-layer pseudo-label gradient norm (GdScore) with MaNo / SoftTrun extensions.
   - Mondrian conformal set cardinality tracking (singleton, empty, multi-label, and $|C(x)| \ge 3$ OOD rates).
3. Benchmark across synthetic shifts, OpenML Electricity, and real-world tabular streams.

### 2.2 Expected Artifacts
- `src/monitors/baselines.py`: Implementations of KS, PSI, JSD, AC, DoC, ATC, CBPE, COTT.
- `src/monitors/model_aware.py`: Implementations of Attribution $L_3$, Cross-Arch Disagreement, D3M, AETTA, GdScore, and Mondrian Conformal monitors.
- `src/monitors/trustguard.py`: Composite multi-signal engine.
- Benchmark reports on benign vs malignant shift discrimination.

### 2.3 Acceptance Criteria & Exit Conditions
- Demonstrates empirical proof of the "Post-Deployment Monitoring Paradox": traditional input drift triggers false alarms under benign shift, while CBPE/ATC report high accuracy under concept collapse.
- Bounded benign-shift false alarm rate ($\text{FAR}_{\text{benign}} \le 0.05$) for TrustGuard signals.
- AUROC/AUPRC degradation detection exceeding baselines under correlation collapse and concept shift.

---

## Stage 3 — Test-Time Adaptation and Model-Selection Audit (Weeks 9–12)

### 3.1 Objectives
1. Implement four test-time adaptation and correction methods:
   - Marginal Quantile Alignment via empirical CDF matching.
   - BatchNorm and Entropy-Based Adaptation (comparing uniform-marginal regularizer vs source-prior anchor $\mu \text{KL}(\bar{p} \parallel \pi_S)$).
   - Filtered Pseudo-Label Refinement (conformal singleton filter vs cross-architecture agreement filter).
   - Saerens EM Label-Shift Correction.
2. Establish candidate model pools across deployment streams:
   - Pool: $\{f_{\text{frozen}}, f_{\text{quantile}}, f_{\text{BN-entropy}}, f_{\text{pseudo-label}}, f_{\text{Saerens}}\}$.
3. Execute the Oracle Selection Audit:
   - Each label-free monitor nominates a candidate model for deployment.
   - The sequestered evaluation oracle computes true post-adaptation accuracy and Selection Regret:
     $$\text{Regret}(M) = \max_{f \in \text{Pool}} \text{Acc}(f, W_t) - \text{Acc}(f_M^*, W_t)$$
4. Audit the Adversarial Collapse Phenomenon under temporal network intrusion streams (CIC-IDS2017/2018) and socioeconomic shifts.

### 3.2 Expected Artifacts
- `src/adaptation/quantile_alignment.py`
- `src/adaptation/batchnorm_entropy.py`
- `src/adaptation/pseudo_labeling.py`
- `src/adaptation/saerens_em.py`
- `src/evaluation/selection_audit.py`
- Quantitative Selection Regret reports and collapse transition curves.

### 3.3 Acceptance Criteria & Exit Conditions
- Empirical demonstration that ATC and CBPE nominate class-collapsed models under severe shift.
- Proof that entropy-scaled AETTA, D3M, and GdScore prevent collapse exploitation and minimize selection regret.

---

## Stage 4 — TrustGuard Guardrail and Research Synthesis (Weeks 13–16)

### 4.1 Objectives
1. Construct the complete Three-Tier TrustGuard Operational Guardrail:
   - Tier 1: Benign Covariate Drift Suppression (gating alarms when D3M and GdScore remain within reference bands).
   - Tier 2: Collapse-Safe TTA Model Recovery (reverting adapted weights to frozen checkpoint upon entropy or disagreement collapse).
   - Tier 3: Per-Sample Conformal Escalation (routing non-singleton $|C(x)| \neq 1$ or $|C(x)| \ge 3$ instances to human analysts).
2. Perform sensitivity analyses and ablations across batch sizes $m$, sampling temperature $\tau$, and individual signal weights.
3. Compute bootstrap confidence intervals ($B = 1000$) and paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction.
4. Synthesize compliance and risk evidence mapped directly to NIST AI 800-4 and the NIST AI Risk Management Framework (RMF).
5. Produce final research paper, reproducible archive, and documentation.

### 4.2 Expected Artifacts
- `src/guardrail/operational_policy.py`: Automated decision policy and escalation manager.
- `src/evaluation/ablations.py`: Signal ablation experiments.
- `docs/NIST_AI_800_4_MAPPING.md`: Complete governance and control mapping.
- Final research report, figures, and publication tables in `results/`.

### 4.3 Acceptance Criteria & Exit Conditions
- End-to-end reproducible benchmark runnable on CPU in under 15 minutes.
- Verified reduction in false alarm rate without sacrificing degradation recall.
- Clean governance audit report matching NIST AI 800-4 categories.
