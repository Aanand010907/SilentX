# Stage 2 Scientific Audit & Benchmark Revalidation: COTT, Calibration Separation, and OpenML Electricity

**Auditor:** Senior Machine Learning Research Engineer & Independent Scientific Auditor  
**Repository:** `SilentX`  
**Monitoring Framework:** `TrustGuard`  
**Base Commit Audited:** `e0c98bf`  
**Revalidation Commit:** Working tree under validation  
**Python Environment:** Python 3.11.17 (`.venv`), PyTorch 2.11, SciPy 1.17, Scikit-learn 1.8, SHAP 0.50, Pillow 12.1  

---

## 1. Executive Summary & Audit Verdict

This audit re-evaluates the Stage 2 claims of SilentX, specifically investigating the algorithmic authenticity of Confidence Optimal Transport with Thresholding (COTT), source calibration split separation, and the complete chronological benchmarking of OpenML Electricity (ID 151).

### Key Audit Findings:
1. **COTT Algorithm Discrepancy (Resolved):** The previously reported implementation contained two critical discrepancies:
   - The execution summary mischaracterized COTT as a 1D scalar-confidence transport heuristic weighting source error indicators.
   - The code in `baselines.py` evaluated source validation costs point-wise ($\|e_{y_i} - p_i\|_\infty$) rather than solving the optimal bipartite matching between validation predictive probabilities and empirical source labels as mandated by the published paper (Lu et al., NeurIPS 2023).
   - **Remediation:** Implemented the exact published COTT algorithm (`ConfidenceOptimalTransportEstimator`) using $L_\infty$ distance, Hungarian validation matching, quantile thresholding, and Kantorovich target transportation. Preserved the experimental 1D transport method separately as `ScalarConfidenceOTHeuristic`.
2. **In-Sample Calibration Overfitting (Resolved):** In the prior version, the entire `X_cal` partition was used both to fit isotonic calibration and to evaluate reference calibration accuracy/thresholds, resulting in an artificial in-sample identity ($\text{DoC offset} = 0.0000$).
   - **Remediation:** Partitioned source calibration chronologically into `X_cal_fit` (indices 15,000–17,499, $N=2{,}500$) for probability calibrator fitting, and `X_cal_ref` (indices 17,500–19,999, $N=2{,}500$) for independent threshold/accuracy evaluation.
3. **Deployment Stream Range Truncation (Resolved):** The prior run evaluated only 25 windows (12,500 samples), leaving 12,812 observations unbenchmarked.
   - **Remediation:** Extended the evaluation across all 25,312 deployment instances, producing 51 chronological windows (50 full windows of $m=500$ and 1 final flagged window of $m=312$).
4. **Time-Series Statistical Dependence (Resolved):** Rather than assuming i.i.d. windows, method comparisons were recalculated using a time-series-aware **Moving-Block Bootstrap (MBB)** ($b \in \{2, 3, 5, 8\}$, $B=2{,}000$ iterations) with Benjamini–Hochberg FDR control.
   - **Result:** Published COTT significantly outperforms CBPE ($p_{\text{MBB}} < 0.001$, FDR adj $p = 0.000$), ATC ($p_{\text{MBB}} < 0.001$, FDR adj $p = 0.000$), DoC ($p_{\text{MBB}} = 0.006$, FDR adj $p = 0.009$), and AC ($p_{\text{MBB}} < 0.001$, FDR adj $p = 0.000$).

---

## 2. Detailed Algorithmic Audit of COTT

### 2.1 The Reference Published Algorithm (NeurIPS 2023)
- **Paper:** *Characterizing Out-of-Distribution Error via Optimal Transport*, NeurIPS 2023 (Yuzhe Lu, Yilong Qin, Runtian Zhai, et al.; arXiv:2305.15640).
- **Official Repository:** `https://github.com/luyuzhe111/COT` (`run_estimation.py`, `utils.py`).

#### Mathematical Specification:
1. **Probability Vectors & One-Hot Labels:**
   For every sample $x_i$, preserve the complete probability simplex vector $\mathbf{p}_i = \vec{f}(x_i) \in \Delta^{k-1}$.
   Represent each source reference label using its one-hot vector $\mathbf{e}_{y} \in \{0, 1\}^k$.
2. **$L_\infty$ Cost Function:**
   $$c(\mathbf{p}, \mathbf{e}_y) = \|\mathbf{p} - \mathbf{e}_y\|_\infty = \max_{c=1}^k |p_c - (e_y)_c|$$
3. **Source Validation Thresholding:**
   On a labeled validation set $\{(\mathbf{p}_i^{\text{val}}, y_i^{\text{val}})\}_{i=1}^{n_{\text{val}}}$:
   - Construct cost matrix $\mathbf{C}^{\text{val}} \in \mathbb{R}^{n_{\text{val}} \times n_{\text{val}}}$ with $\mathbf{C}^{\text{val}}_{ij} = \|\mathbf{p}_i^{\text{val}} - \mathbf{e}_{y_j^{\text{val}}}\|_\infty$.
   - Solve optimal matching $\mathbf{P}_{\text{val}}^* = \arg\min_{\mathbf{P} \in \Pi} \langle \mathbf{P}, \mathbf{C}^{\text{val}}\rangle$ via the Hungarian algorithm (`linear_sum_assignment`).
   - Extract matched transport costs: $c_i^{\text{matched}} = \mathbf{C}^{\text{val}}_{i, \pi(i)}$.
   - Calculate empirical validation classification error:
     $$\text{Err}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \mathbb{I}(\arg\max_c p_{i, c}^{\text{val}} \ne y_i^{\text{val}})$$
   - Determine threshold $t^*$ such that the fraction of matched samples with cost $\ge t^*$ matches $\text{Err}_{\text{val}}$:
     $$t^* = \text{Quantile}_{(1 - \text{Err}_{\text{val}})}\left(\{c_i^{\text{matched}}\}_{i=1}^{n_{\text{val}}}\right)$$
4. **Target Batch Error Estimation:**
   For an unlabeled target batch $\{\mathbf{p}_j^T\}_{j=1}^{n_T}$:
   - Construct cost matrix $\mathbf{C} \in \mathbb{R}^{n_T \times n_{\text{val}}}$ with $\mathbf{C}_{ji} = \|\mathbf{p}_j^T - \mathbf{e}_{y_i^{\text{val}}}\|_\infty$.
   - Solve optimal transport plan $\mathbf{P}^* = \arg\min_{\mathbf{P} \in \Pi} \langle \mathbf{P}, \mathbf{C}\rangle$ via `linear_sum_assignment` (if $n_T = n_{\text{val}}$) or HiGHS linear programming (if $n_T \ne n_{\text{val}}$).
   - Target error estimate:
     $$\hat{\epsilon}_{\text{COTT}} = \sum_{j, i : \mathbf{C}_{ji} \ge t^*} \mathbf{P}^*_{ji}$$
   - Target accuracy estimate:
     $$\widehat{\text{Acc}}_{\text{COTT}} = \max\left(0, \min\left(1, 1 - \hat{\epsilon}_{\text{COTT}}\right)\right)$$

### 2.2 Comparison with Previous Code & Preservation of Scalar Heuristic
| Dimension | Published COTT (Corrected) | Previous `baselines.py` | Experimental Scalar Heuristic (`ScalarConfidenceOTHeuristic`) |
| :--- | :--- | :--- | :--- |
| **Input Representation** | Full probability simplex $\mathbf{p} \in \Delta^{k-1}$ | Full probability simplex $\mathbf{p}$ | 1D scalar top-1 confidence $c = \max_k p_k \in [0, 1]$ |
| **Label Representation** | One-hot vector $\mathbf{e}_y \in \{0, 1\}^k$ | One-hot vector $\mathbf{e}_y$ | Binary error indicator $e_i \in \{0, 1\}$ |
| **Cost Metric** | $L_\infty$ distance $\|\mathbf{p} - \mathbf{e}_y\|_\infty$ | $L_\infty$ distance | Absolute difference $|c_i^S - c_j^T|$ |
| **Validation Calibration** | Solves Hungarian matching $\mathbf{P}^*_{\text{val}}$ | Point-wise loss $\|\mathbf{p}_i - \mathbf{e}_{y_i}\|_\infty$ | No threshold (direct mass transfer) |
| **Target Estimation** | Transported mass to cost $\ge t^*$ | Transported mass to cost $\ge t^*$ | Mass-weighted source error indicators $\sum \Gamma_{ij}^* e_i^S$ |
| **Status in Repository** | `ConfidenceOptimalTransportEstimator` | **Replaced / Corrected** | `ScalarConfidenceOTHeuristic` (Preserved separately) |

---

## 3. Calibration Integrity & Split Separation

To prevent in-sample overfitting and eliminate artificial calibration identities, the source data partitions were strictly divided:
- **`X_train` ($N=15{,}000$, indices 0–14,999):** Used to fit `StandardScaler`, primary tree classifier (XGBoost), secondary MLP, and diagnostic logistic regression.
- **`X_cal_fit` ($N=2{,}500$, indices 15,000–17,499):** Used strictly to evaluate raw probabilities and fit the 3-fold cross-validated Isotonic Probability Calibrator.
- **`X_cal_ref` ($N=2{,}500$, indices 17,500–19,999):** Held-out reference set evaluated with the frozen calibrator. Used to derive:
  - Reference Calibration Accuracy: **82.52%**
  - Reference Mean Confidence: **86.49%**
  - Out-of-Sample DoC Offset: **0.0397** (demonstrating true, non-trivial miscalibration)
  - ATC Threshold $\tau_{\text{ATC}}$: **0.7131**
  - COTT Cost Threshold $t^*$: **0.4384**
  - Mondrian Conformal Quantiles ($\alpha = 0.10$)
  - TreeSHAP reliance weights and D3M reference disagreement distribution.
- **`X_deploy` ($N=25{,}312$, indices 20,000–45,311):** Unlabeled evaluation stream evaluated under strict zero-target-label isolation.

---

## 4. OpenML Electricity Dataset Chronological Revalidation

### 4.1 Chronological Order Verification
OpenML Electricity (ID 151) consists of 45,312 instances recorded in 30-minute intervals from May 7, 1996 to December 5, 1998.
- **Verification:**
  - Every 48 consecutive rows correspond exactly to a complete 24-hour day (`groupby(index // 48)['period'].count() == 48` is `True` for all 944 days).
  - Day-of-week transitions are strictly sequential ($d_{t+1} \equiv (d_t + 1) \pmod 7$ for all 943 day boundaries with 0 skips).
  - The 5 instances of negative differences in the float `date` column represent ARFF min-max scaling boundary artifacts, whereas native row order is monotonically advancing in physical time. Sorting by `date` would scramble the 48-period day progression; preserving row order is required for chronological validity.

### 4.2 Full Deployment Stream Coverage
The deployment stream was partitioned into 51 sequential windows:
- Windows $0 \dots 49$: 50 full windows of $m=500$ instances each (indices 20,000 to 44,999).
- Window $50$: 1 final window of $m=312$ instances (indices 45,000 to 45,311), explicitly flagged with `is_partial_window = True`.
- **Total evaluated samples:** Exactly 25,312 deployment instances (100% data coverage, 0 discarded rows).

---

## 5. Recomputed Benchmark Results

### 5.1 Real OpenML Electricity Benchmark (51 Deployment Windows, $N=25{,}312$)
Summary metrics from [`results/stage2_electricity_aggregate.csv`](file:///c:/New%20folder%20(2)/projects/SilentX/results/stage2_electricity_aggregate.csv):

| Monitor / Estimator | Mean Estimation MAE | Notes & Operational Properties |
| :--- | :---: | :--- |
| **Published COTT (`ConfidenceOptimalTransportEstimator`)** | **0.0965** | **Lowest MAE across all 51 windows; captures joint distribution shifts.** |
| **Scalar OT Heuristic (`ScalarConfidenceOTHeuristic`)** | $0.1140$ | Competent 1D baseline, but lacks multidimensional class-simplex geometry. |
| **DoC (`DifferenceOfConfidencesEstimator`)** | $0.1729$ | Improves upon AC via out-of-sample offset subtraction ($\text{offset} = 0.0397$). |
| **CBPE (`CBPEPerformanceEstimator`)** | $0.2053$ | Underestimates degradation when target joint distribution shifts. |
| **ATC (`AverageThresholdedConfidenceEstimator`)** | $0.2068$ | Fails to detect degradation when overconfident samples exceed $\tau_{\text{ATC}}$. |
| **AC (`AverageConfidenceEstimator`)** | $0.2053$ | Severely over-optimistic (mean target confidence $\approx 86.5\%$). |

#### Degradation Detection Performance (Ground Truth Threshold: $\Delta \text{Acc} \ge 0.05$):
- **Total Deployment Windows:** 51
- **Degraded Windows:** 35 ($68.6\%$)
- **Benign Windows:** 16 ($31.4\%$)
- **TrustGuard Detection AUROC:** **0.8054**
- **TrustGuard Detection AUPRC:** **0.9153**
- **TrustGuard Spearman Rank Correlation:** $\rho_s = \mathbf{0.7040}$ ($p = 8.28 \times 10^{-9}$)

---

### 5.2 Time-Series Moving-Block Bootstrap (MBB) Statistical Testing
Paired error differences $d_t = |e_{\text{baseline}, t}| - |e_{\text{COTT}, t}|$ across the 51 chronological windows, computed via Moving-Block Bootstrap ($b=3$, $B=2{,}000$ iterations) and Benjamini–Hochberg FDR correction:

| Comparison | Mean $\Delta$ Error | 95% MBB CI ($b=3$) | MBB Raw $p$ | MBB FDR Adj $p$ | Sensitivity ($b=2, 5, 8$) | Wilcoxon Adj $p$ | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CBPE vs. COTT** | $+0.1087$ | $[+0.0450, +0.1656]$ | $<0.001$ | **0.0000** | $p=0.000, 0.000, 0.000$ | $0.0002$ | **COTT significantly superior** |
| **ATC vs. COTT** | $+0.1103$ | $[+0.0460, +0.1680]$ | $<0.001$ | **0.0000** | $p=0.000, 0.000, 0.000$ | $0.0002$ | **COTT significantly superior** |
| **DoC vs. COTT** | $+0.0763$ | $[+0.0168, +0.1295]$ | $0.0060$ | **0.0090** | $p=0.002, 0.010, 0.010$ | $0.0047$ | **COTT significantly superior** |
| **AC vs. COTT** | $+0.1087$ | $[+0.0450, +0.1656]$ | $<0.001$ | **0.0000** | $p=0.000, 0.000, 0.000$ | $0.0002$ | **COTT significantly superior** |
| **ScalarOT vs. COTT** | $+0.0175$ | $[-0.0247, +0.0521]$ | $0.5150$ | $0.5150$ | $p=0.412, 0.384, 0.368$ | $0.3534$ | COTT lower MAE; diff not sig. |
| **CBPE vs. ATC** | $-0.0015$ | $[-0.0046, +0.0014]$ | $0.2680$ | $0.3216$ | $p=0.240, 0.290, 0.390$ | $0.2389$ | No significant difference |

---

### 5.3 Multi-Seed Synthetic Benchmark (Seeds 42, 123, 999)
Full records saved in [`results/stage2_synthetic_summary.csv`](file:///c:/New%20folder%20(2)/projects/SilentX/results/stage2_synthetic_summary.csv):

| Regime | True Acc | CBPE Est | ATC Est | COTT Est | Scalar OT | KS Max | PSI Max | TrustGuard Risk | Alarm Triggered? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Distribution** | 0.799 | 0.813 | 0.891 | 0.753 | 0.803 | 0.101 | 0.087 | 0.036 | Low (0.17) |
| **Harmless Drift** | 0.782 | 0.804 | 0.871 | 0.752 | 0.803 | 0.942 | 7.023 | 0.291 | High (1.00) |
| **Correlation Collapse** | 0.752 | 0.833 | 0.902 | 0.688 | 0.803 | 0.372 | 0.876 | 0.119 | High (1.00) |
| **Prior Shift** | 0.772 | 0.815 | 0.891 | 0.753 | 0.803 | 0.087 | 0.075 | 0.037 | Low (0.17) |
| **Concept Shift** | 0.338 | 0.809 | 0.861 | 0.722 | 0.803 | 0.126 | 0.118 | 0.055 | High (0.83) |
| **Confidence Collapse** | 0.010 | 0.968 | 0.993 | 0.538 | 0.803 | 0.793 | 5.531 | 0.222 | High (1.00) |

---

## 6. Audit Checklist & Status of Claims

- [x] **Claim 1: COTT Algorithm Equivalency:** The prior report described 1D scalar transport, whereas `baselines.py` implemented point-wise loss. **CORRECTED**: Implemented full published COTT (`ConfidenceOptimalTransportEstimator`) with $L_\infty$ distance, Hungarian matching, and target Kantorovich LP; preserved `ScalarConfidenceOTHeuristic` separately.
- [x] **Claim 2: COTT Superiority on OpenML Electricity:** **VERIFIED**: Recomputed across 51 chronological windows ($N=25{,}312$). Published COTT achieved **0.0965 MAE**, statistically outperforming CBPE (0.2053), ATC (0.2068), and DoC (0.1729) under Moving-Block Bootstrap ($p < 0.01$) and Benjamini–Hochberg FDR control.
- [x] **Claim 3: Source Calibration Independence:** **VERIFIED**: Split source calibration into `X_cal_fit` ($N=2{,}500$) and held-out `X_cal_ref` ($N=2{,}500$), eliminating in-sample evaluation and yielding a non-trivial DoC offset of 0.0397.
- [x] **Claim 4: OpenML Electricity Chronological Coverage:** **VERIFIED**: Analyzed 48-period day progression, confirmed exact temporal sequence without sorting bugs, and evaluated all 25,312 deployment instances across 51 windows (50 full + 1 partial).
- [x] **Claim 5: Automated Test Suite:** **VERIFIED**: 28 automated tests passing in 6.76s covering $L_\infty$ cost construction, unequal transport sizes, tied costs, scalar heuristic distinction, split separation, and full window coverage.
