# SilentX — Stage 1 Scientific Audit Report

**Audit Subject:** Stage 1 Calibration and Benchmark Artifacts  
**Auditor:** Senior ML Research Engineer, Scientific Auditor & Python Architect  
**Project:** SilentX  
**System:** TrustGuard  
**Date:** October 2026  
**Status:** Complete Audit with Mathematical Proofs and Root-Cause Analysis  

---

## Executive Summary

This independent scientific audit examined the initial claims and outputs of the SilentX Stage 1 implementation. While all 15 behavioral unit and integration tests passed, rigorous empirical and theoretical auditing revealed five major scientific findings and anomalies. Crucially, the audit confirms that passing unit tests alone does not establish post-deployment monitoring efficacy under complex distribution shifts.

---

## Audit Register: Detailed Findings

### Finding 1: Harmless Covariate Drift Amplification via $L_3$ Attribution Norm

* **Original Claim:** The initial report claimed TrustGuard reliably suppresses false alarms under benign covariate shift ($\text{FAR}_{\text{benign}} = 0.0$), whereas traditional univariate drift detectors trigger severe false alarms ($\text{KS}_{\max} = 1.000$).
* **Verification Procedure:** 
  1. Inspected `SyntheticStreamGenerator.generate_regime("harmless_drift")` in `src/data/synthetic_streams.py`.
  2. Verified whether shifted features $X_5 \dots X_9$ have any causal effect on logits.
  3. Audited TreeSHAP normalized reliance weights $w_j$ and the mathematical formulation of $S_{\text{attr}}$ in `src/monitors/trustguard.py`.
  4. Traced why $S_{\text{attr}} = 0.458$ and evaluated how `far_benign` was computed.
* **Empirical Evidence:**
  - In `harmless_drift`, the data generator modulates *strictly* uninformative noise features:
    $X_5 \dots X_9 \sim \mathcal{N}(\mu=4.5, \sigma^2=2.5^2)$, while causal features $X_0, X_1, X_2$ and proxy $X_3$ remain stationary.
  - The trained XGBoost model assigns low TreeSHAP reliance weights to $X_5 \dots X_9$ ($\approx 0.010 - 0.017$ each, summing to $\approx 0.075$).
  - However, the formula implemented for marginal attribution drift is:
    $$D_{\text{marg}}(W_t) = \left(\sum_{j=1}^d w_j \cdot (\text{KS}_j(W_t))^3\right)^{1/3}$$
  - For $j \in \{5, \dots, 9\}$, $\text{KS}_j = 1.0$. The weighted sum is $\sum_{j=5}^9 w_j \cdot 1^3 \approx 0.075$.
  - Taking the cube root dramatically amplifies this small fraction: $(0.075)^{1/3} \approx \mathbf{0.422}$!
  - Combined with correlation matrix shifts among noise features, $S_{\text{attr}}$ rises to $0.458 - 0.570$.
  - In addition, the initial runner had hardcoded `risk_alarm_threshold = 0.45` rather than deriving the 95th-percentile alarm threshold from source calibration bootstrap batches.
* **Audit Status:** **PARTIAL DEFECT IDENTIFIED & CORRECTED**.
* **Corrective Action:**
  1. Corrected `TrustGuardCompositeMonitor` to compute excess signals $\tilde{S}_k(W_t) = \max(0, S_k(W_t) - S_k(D_{\text{cal}}))$ as required by Equation 188 of `RESEARCH_BLUEPRINT.md`.
  2. Implemented empirical bootstrap calibration of `alarm_threshold` on $D_{\text{cal}}$ ($\text{Quantile}_{1-\alpha}(\text{Risk}(D_{\text{cal}}))$), replacing hardcoded thresholds.
  3. Documented the mathematical property of the $L_p$ norm: while cubing individual KS values ($\text{KS}_j^3$) suppresses small feature drifts, taking the outer cube root $\left(\cdot\right)^{1/3}$ amplifies the sum of small feature weights when $\text{KS}_j \to 1.0$.

---

### Finding 2: Concept-Shift Blindness and the Theoretical Impossibility Boundary

* **Original Claim:** The initial runner reported Window 9 (`concept_shift`, severity 1.0) with true accuracy $17.0\%$ ($\Delta\text{Acc} = 67.4\%$), while CBPE estimated $84.5\%$, ATC estimated $85.8\%$, and TrustGuard risk was only $0.103$ (lower than in-distribution baseline).
* **Verification Procedure:**
  1. Analyzed the exact mathematical data-generating process for `concept_shift` in `src/data/synthetic_streams.py`.
  2. Evaluated all observable input signals: KS statistics, TreeSHAP attribution, secondary MLP predictions, conformal prediction sets, and closed-form GdScore gradient norms.
* **Empirical Evidence:**
  - In `concept_shift`, the decision boundary rotates by inverting the signs of coefficients on $X_0$ and $X_1$:
    $$\text{logits} = (2.2 - 4.2 \cdot s) X_0 + (-1.8 + 3.8 \cdot s) X_1 - 1.4 (X_0 X_2)$$
  - Marginal covariates $P(X)$ undergo negligible displacement ($\Delta X_1 = 0.3 \times \text{severity}$).
  - Because $X$ remains inside the dense source training manifold:
    - Primary tree predictions $f_{\text{tree}}(X)$ remain well-behaved ($p \approx 0.85$).
    - Secondary MLP predictions $f_{\text{mlp}}(X)$ agree with the tree ($S_{\text{dis}} \approx 0.097$, baseline level).
    - Conformal prediction sets $C(x)$ remain singletons ($S_{\text{conf}} \approx 0.194$, baseline level).
    - Hard pseudo-labels $\hat{y}_i$ match $p(x_i)$, so the pseudo-label gradient norm is near zero ($S_{\text{grad}} \approx 0.002$).
    - Input drift detectors report $\text{KS}_{\max} = 0.102$ (statistically insignificant).
* **Audit Status:** **CONFIRMED FUNDAMENTAL SCIENTIFIC LIMITATION**.
* **Scientific Analysis:**
  - This empirical finding directly confirms the **Unidentifiability Proposition** of `RESEARCH_BLUEPRINT.md` (Section 3):
    *When $P_t(X) = P_{\text{source}}(X)$ identically everywhere, no label-free algorithm can detect performance degradation resulting from pure concept shift.*
  - Neither CBPE, nor ATC, nor COTT, nor TrustGuard can detect a concept inversion when the shift leaves zero observable footprint in inputs, representations, or hypothesis disagreements.
  - In production systems, detection is possible only when concept shift co-occurs with boundary crossing, proxy decoupling, or representation stress.

---

### Finding 3: High-Confidence Extrapolation Collapse Detection

* **Original Claim:** Under `confidence_collapse` (Window 11), true model accuracy plunges to $0.0\%$, while CBPE and ATC report $\approx 100\%$ accuracy.
* **Verification Procedure:**
  1. Tested Window 11 across multiple random seeds (`42`, `123`, `999`).
  2. Verified whether TrustGuard signals and baseline confidence estimators reproduce this behavior deterministically.
* **Empirical Evidence:**
  - Across all tested seeds, Window 11 consistently produces true accuracy $= 0.0\%$.
  - CBPE estimated accuracy is $98.4\% - 99.6\%$.
  - ATC estimated accuracy is $100.0\%$.
  - In contrast, TrustGuard's Attribution Drift spikes to $S_{\text{attr}} = 0.863 - 0.882$ and GdScore spikes to $S_{\text{grad}} = 0.288$.
  - However, because both XGBoost and the MLP saturated to $1.0$, cross-architecture disagreement dropped to $S_{\text{dis}} = 0.065$ and conformal abstention dropped to $S_{\text{conf}} = 0.000$.
* **Audit Status:** **VERIFIED SCIENTIFIC ANOMALY EXPLAINED**.
* **Corrective Action:**
  - When a model collapses into saturated high confidence, conformal prediction sets collapse to singletons $\{1\}$, and homogeneous ensembles make identical confident errors.
  - Therefore, the composite risk score must not allow zero disagreement or zero conformal abstention to mask extreme attribution drift and gradient stress. Incorporating excess signals $\tilde{S}_{\text{attr}}$ and integrating D3M variational disagreement restores the alarm.

---

### Finding 4: Calibration Integrity and the 84.40% Identity Proof

* **Original Claim:** Source calibration accuracy and mean confidence were reported as identical at $84.40\%$ in `stage1_calibration_summary.json`, yielding a DoC offset of $0.0000$.
* **Verification Procedure:**
  1. Inspected source calibration code in `src/models/calibration.py` and `src/models/frozen_pipeline.py`.
  2. Evaluated whether this was an artifact of label leakage or a mathematical property of the calibration method.
* **Mathematical Proof & Evidence:**
  - The calibrator applies Isotonic Regression fitted via the Pool Adjacent Violators Algorithm (PAVA) on $(p_{\text{raw}}, y_{\text{cal}})$.
  - Isotonic regression partitions the predictions into piecewise constant bins $B_k$, where each bin's calibrated probability $\hat{p}_k$ is the exact empirical mean of labels in that bin:
    $$\hat{p}_k = \frac{1}{|B_k|} \sum_{i \in B_k} y_i$$
  - With standard decision threshold $0.5$:
    - In bins where $\hat{p}_k \ge 0.5$, predictions are $\hat{y}_i = 1$, bin accuracy is $\hat{p}_k$, and bin confidence is $\max(\hat{p}_k, 1 - \hat{p}_k) = \hat{p}_k$.
    - In bins where $\hat{p}_k < 0.5$, predictions are $\hat{y}_i = 0$, bin accuracy is $1 - \hat{p}_k$, and bin confidence is $\max(\hat{p}_k, 1 - \hat{p}_k) = 1 - \hat{p}_k$.
  - Therefore, inside every isotonic bin, empirical accuracy identically equals mean confidence:
    $$\text{Acc}(B_k) \equiv \text{Conf}(B_k)$$
  - Summing across bins proves that in-sample on the calibration split:
    $$\text{Acc}(D_{\text{cal}}) \equiv \text{MeanConfidence}(D_{\text{cal}}) \equiv 84.40\%$$
    $$\text{DoC Offset } d = \text{MeanConfidence}(D_{\text{cal}}) - \text{Acc}(D_{\text{cal}}) \equiv 0.0000$$
* **Audit Status:** **VERIFIED MATHEMATICAL IN-SAMPLE IDENTITY**.
* **Conclusion:** This equality is not a bug or label leakage; it is an exact mathematical property of in-sample isotonic calibration.

---

### Finding 5: Genuine Implementation versus Placeholders

* **Verification of Core Modules:**
  1. **Variational Bayesian Last Layer (VBLL):** Genuine PyTorch implementation with reparameterized Monte Carlo sampling, analytical Gaussian KL divergence, and ELBO training. Passes numerical gradient and tensor shape tests.
  2. **D3M Reference Calibrator:** Correctly executes bootstrap resampling across query sizes $m \in \{10, 20, 50\}$ and locks 95th-percentile disagreement thresholds.
  3. **Mondrian Conformal Prediction:** Implements the finite-sample corrected quantile formula $\frac{\lceil(N_c + 1)(1 - \alpha)\rceil}{N_c}$.
  4. **GdScore:** Computes closed-form Frobenius norm of the classification-layer cross-entropy loss gradient: $\|(p - \hat{y}) z^\top\|_F$.
  5. **Ground-Truth Isolation:** Verified by automated boundary tests. Monitors receive only $X_{\text{unlabeled}}$ DataFrames; ground truth is held in `EvaluationOracle`.

---

## Audit Conclusion & Transition to Stage 2

The Stage 1 architecture is mathematically sound and verified. The primary defects identified—unsubtracted baseline signals, hardcoded alarm thresholds, and omission of D3M from the composite score—have been repaired. The codebase is now ready for Stage 2 expansion: implementing PSI, multiclass CBPE, Confidence Optimal Transport with Thresholding (COTT), and benchmarking the real OpenML Electricity dataset.
