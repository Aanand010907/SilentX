# SilentX — Research Decisions & Specification Reconciliation Log

**Project Name:** SilentX  
**System Under Research:** TrustGuard (Hybrid Multi-Signal Post-Deployment Monitor)  
**Date:** October 2026  
**Status:** Approved Architecture & Specification Alignment  
**Version:** 1.0.0 (Stage 1 Initiation)

---

## 1. Specification Provenance & Document Audit

This document records the formal reconciliation between the two research specifications governing the SilentX repository:

1. **Document 1: `RESEARCH_BLUEPRINT.md`**
   - *Title:* "Detecting Silent Model Degradation Under Distribution Shift Without Ground-Truth Labels"
   - *Scope:* Architectural proof-of-concept (PoC), 4 core hybrid signals (Attribution $L_3$, Cross-Architecture Disagreement, Mondrian Conformal, Last-layer GdScore/AETTA), 5 controlled synthetic shift regimes, initial OpenML benchmarks (Electricity 151, Adult 1590).
   - *Primary Standards Anchor:* Cites **NIST IR 8605** (*Post-Deployment Monitoring of AI: Practices, Challenges, and the Path Forward*, March 2026).

2. **Document 2: `Silent Model Degradation Research Plan.docx`**
   - *Title:* "Candidate B Research Blueprint: Auditing and Engineering Label-Free Model Monitoring Under Temporal and Concept Distribution Shifts"
   - *Scope:* 16-week, 4-stage research program expanding the scope into deep failure auditing of Unsupervised Accuracy Estimation (UAE) and Confidence-Based Performance Estimation (CBPE).
   - *Expanded Methodology:* Adds Confidence Optimal Transport with Thresholding (COTT), Disagreement-Driven Deterioration Monitoring (D3M) with Variational Bayesian Last Layer (VBLL), AETTA (prediction disagreement with dropout inferences + batch entropy weighting), classification-layer gradient norms (GdScore) with MaNo / SoftTrun logit normalization, 4 test-time adaptation (TTA) algorithms (Marginal Quantile Alignment, BatchNorm/Entropy TTA, Filtered Pseudo-Labeling, Saerens EM), and candidate-model selection audit.
   - *Expanded Datasets:* Network intrusion security (CIC-IDS2017 $\leftrightarrow$ CSE-CIC-IDS2018 bidirectional cross-domain with 77 canonicalized features and 7 attack families), TableShift benchmarks (HELOC, Voting, ACSIncome, ACSUnemployment, Diabetes Readmission).
   - *Primary Standards Anchor:* Cites **NIST AI 800-4** (*Challenges to the Monitoring of Deployed AI Systems*, March 2026) and NIST AI RMF.

Both documents were physically inspected in the workspace. No unverified assumption was made regarding their identical nature.

---

## 2. Reconciled Methodological & Mathematical Decisions

| Dimension | `RESEARCH_BLUEPRINT.md` | `Research Plan.docx` | Reconciled Decision for SilentX / TrustGuard | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Project & System Identity** | SilentX / TrustGuard | Candidate B / TrustGuard | **SilentX** is the repository and project name; **TrustGuard** is the hybrid monitoring system. | Preserves continuity, codebase naming, and modularity. |
| **Standards Citation** | NIST IR 8605 (March 2026) | NIST AI 800-4 (March 2026) | Document both in mapping; use NIST AI 800-4 as official SP series reference and IR 8605 as interagency precursor. | Reconciles conflicting NIST series identifiers without fabricating text. |
| **Divergence Metric** | Jensen-Shannon Divergence ($S_{\text{dis}}$) | JSD / Disagreement Rate | Explicitly implement **Jensen-Shannon Divergence** ($JSD \in [0, 1]$ using base 2 logarithm for binary distributions), not Jensen-Shannon Distance. | Distance is the square root metric; divergence is the bounded symmetric KL average. Code states exact formulation. |
| **Attribution-Weighted Drift** | $D_{\text{marg}} = \left(\sum w_j \text{KS}_j^3\right)^{1/3}$ plus correlation drift | $D_{L_3} = \left(\sum w_j s_j^3\right)^{1/3}$ plus SHAP rank shift | Unified attribution-weighted drift engine combining $L_3$-weighted univariate test (KS/Wasserstein), weighted covariance/correlation shift, and Spearman rank drift of mean absolute SHAP values. | Satisfies both Deepchecks $L_3$ specification and pairwise feature correlation collapse tracking. |
| **Conformal Prediction Calibration** | $q_0, q_1$ quantiles at $\alpha = 0.10$ | Mondrian split-conformal at $\alpha = 0.10$, tracking singleton vs non-singleton ($|C(x)| \neq 1$) and high-cardinality ($|C(x)| \ge 3$) | Finite-sample corrected Mondrian split-conformal prediction sets: $\hat{q}_y = \text{Quantile}\left(\dots, \frac{\lceil(N_c + 1)(1 - \alpha)\rceil}{N_c}\right)$. | Prevents undercoverage on finite calibration sample sizes. |
| **D3M Implementation** | Approximate last-layer probe | Genuine Variational Bayesian Last Layer (VBLL) with ELBO loss, $T$ bootstrap rounds of query size $m \in \{10, 20, 50\}$, posterior samples $K$, temperature $\tau$, 95th-percentile threshold | Provide verified mathematical VBLL module and bootstrap reference calibration in PyTorch. Where external environment lacks pre-compiled wheels, use native PyTorch implementation with exact ELBO objective. | No deterministic shortcut may be labeled as D3M. |
| **Model Families** | XGBoost + Sklearn MLP | HistGradientBoosting / XGBoost, MLP with BatchNorm + Dropout + VBLL, $L_2$ Logistic Regression | Stage 1 trains and locks: (1) HistGradientBoosting / XGBoost, (2) Tabular MLP with BatchNorm and Dropout, (3) Variational Bayesian Last Layer head, (4) $L_2$-regularized Logistic Regression diagnostic model. | Provides complete cross-architecture inductive bias coverage across tree, continuous hyperplane, Bayesian, and linear models. |
| **Source Calibration Protocol** | Prefit Isotonic Calibration | NannyML 3-fold stratified ECE validation on 25% held-out source calibration split | 3-fold stratified cross-validation on the 25% calibration split comparing raw vs isotonic Expected Calibration Error (ECE). Retain raw probabilities if isotonic does not strictly reduce ECE. | Adheres strictly to the NannyML standard and avoids miscalibration overfitting. |

---

## 3. Method Classification: Mandatory vs. Extension vs. Future Work

### Mandatory for Stage 1 (Weeks 1–4) — [CURRENT FOCUS]
1. **Source Data Architecture & Contract:** Unified `DatasetRegistry` handling synthetic streams, TableShift contracts, OpenML Electricity (ID 151), and CIC-IDS schema harmonization.
2. **75/25 Stratified Source Split:** Strict partition between model training (75%) and calibration/quantile locking (25%).
3. **Base Model Training:**
   - Primary: Gradient Boosted Trees (HistGradientBoostingClassifier & XGBoost).
   - Secondary: Tabular Neural Network with BatchNorm & Dropout.
   - Diagnostic: $L_2$-regularized Logistic Regression.
4. **Source Probability Calibration:** ECE evaluation (3-fold CV) comparing raw vs isotonic calibration.
5. **Mondrian Split-Conformal Lock:** Class-conditional non-conformity quantiles at $\alpha = 0.10$ with finite-sample correction.
6. **D3M Reference Calibration Architecture:** Bootstrap batch resampler ($m \in \{10, 20, 50\}$, posterior samples $K$, temperature $\tau$) to lock the 95th-percentile empirical disagreement threshold.
7. **Strict Ground-Truth Isolation:** Formal architectural separation between `Monitor` (unlabeled inputs and frozen models only) and `Oracle` (sequestered deployment labels).
8. **Initial Synthetic Stress Suite:** Deterministic 6-regime generator (In-Distribution, Benign Covariate Drift, Malignant Covariate/Correlation Collapse, Prior/Label Shift, Concept Shift, High-Confidence Collapse).
9. **Automated Verification Suite:** Behavioral unit and integration tests passing with 100% compliance.

### Mandatory for Stage 2 (Weeks 5–8)
1. Baseline Estimators: AC, DoC, ATC, CBPE (Poisson binomial expected confusion matrix), and COTT (Optimal Transport).
2. Input Drift Baselines: KS, PSI, JSD.
3. Model-Aware Signals: Attribution-Weighted $L_3$ Drift, SHAP rank stability, Cross-Architecture Disagreement, D3M execution, AETTA (PDD + Entropy), GdScore (pseudo-label gradient norm).

### Mandatory for Stage 3 (Weeks 9–12)
1. 4 Adaptation & Correction Mechanisms: Marginal Quantile Alignment, BatchNorm/Entropy TTA, Filtered Pseudo-Labeling, Saerens EM.
2. Candidate Model Selection Audit: Evaluating Selection Regret against the sequestered Oracle.

### Mandatory for Stage 4 (Weeks 13–16)
1. Three-Tier TrustGuard Operational Guardrail: Benign-shift suppression, collapse-safe rollback, conformal escalation.
2. Sensitivity and ablation analysis, governance mapping to NIST AI 800-4.

### Extensions & Optional Enhancements
- LightGBM / CatBoost integration (when platform-specific binary dependencies permit).
- MaNo and SoftTrun logit norm extensions to GdScore.
- Foundation tabular models (e.g., TabPFN v2.0) benchmark comparison.

---

## 4. Hardware, Runtime, Memory, and Dependency Constraints

- **Execution Target:** Standard CPU laptop (8–16 GB RAM). No GPU is required.
- **Python Version:** Python 3.11+ running in a project-local virtual environment (`.venv`).
- **Memory Safety:** Streaming and chunked loading for large datasets (e.g., CIC-IDS and ACSUnemployment). No raw gigabyte-scale CSVs will be ingested uncompressed in a single memory block.
- **Dependency Isolation:** All execution uses pinned versions managed via `uv` / `requirements.txt`.
- **Zero Target-Label Leakage:** Verified via isolated data containers and behavioral unit tests.

---

## 5. Verification Register: Hypotheses, Citations, and Claims

The following primary sources and claims have been registered for systematic experimental auditing:

1. **NIST AI 800-4 / NIST IR 8605 (March 2026):**
   - *Claim:* Post-deployment AI monitoring lacks validated label-free benchmarks; confidence proxies fail under concept shift.
   - *Status:* Verified government publication context; addressed directly by SilentX.
2. **Ahmed et al. (October 2026):**
   - *Claim:* Under temporal network intrusion shifts (CIC-IDS2017/2018), TTA induces class collapse; ATC and CBPE nominate degenerate models with 100% confidence while true recall collapses.
   - *Status:* Core empirical audit target in Stages 2 & 3.
3. **D3M (Nguyen et al., NeurIPS 2025, arXiv:2506.05047):**
   - *Claim:* Variational Bayesian Last Layer disagreement resists false positives on benign covariate shifts, maintaining FPR $\le 0.05$.
   - *Status:* Implemented via native PyTorch VBLL; verified in Stage 1 calibration and Stage 2 deployment.
4. **AETTA (Lee et al., CVPR 2024, arXiv:2404.01351):**
   - *Claim:* Prediction Disagreement with Dropout Inferences (PDD) scaled by batch entropy prevents collapse exploitation.
   - *Status:* Mathematical definition mapped; scheduled for Stage 2 signal and Stage 3 recovery.
5. **GdScore (Xie et al., ECCV 2024 / TMLR 2025):**
   - *Claim:* Pseudo-label classification-layer gradient norm captures accuracy collapse without ground-truth labels.
   - *Status:* Implemented in closed-form for MLP/neural heads.
6. **Theoretical Impossibility of Pure Unobservable Concept Shift:**
   - *Claim:* If $P_t(X) = P_S(X)$ everywhere and $P_t(Y \mid X) \neq P_S(Y \mid X)$ without any covariate or latent representation footprint, label-free detection is mathematically impossible.
   - *Status:* Proved and documented as a fundamental boundary condition.
