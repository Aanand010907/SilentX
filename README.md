# SilentX — Detecting Silent Model Degradation Under Distribution Shift

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Stage 1 Verified](https://img.shields.io/badge/stage-1_calibration_locked-brightgreen.svg)]()
[![System: TrustGuard](https://img.shields.io/badge/system-TrustGuard_Hybrid_Monitor-orange.svg)]()

> **SilentX** is a research framework investigating how deployed machine-learning classifiers suffer silent performance degradation under distribution shift when ground-truth labels are delayed or unobserved.
> The proposed hybrid monitoring system is **TrustGuard**.

---

## 1. Executive Summary & Research Mandate

Machine learning classifiers deployed in high-stakes domains (network intrusion detection, credit risk, healthcare triage) operate under a fundamental asymmetry: input distributions evolve continuously upon deployment, while ground-truth labels arrive with severe latency or remain entirely unobservable.

Standard monitoring architectures suffer from a diagnostic dichotomy:
1. **Unsupervised Input Drift (KS, PSI, JSD):** Flags *any* statistical change in $P(X)$, causing relentless false alarms under **Harmless Covariate Shift** when uninformative noise features drift while the decision boundary remains unaffected.
2. **Confidence-Based Performance Estimation (CBPE, ATC, DoC):** Assumes conditional invariance $P(Y \mid X)$ and source calibration transfer. Under **Concept Shift**, **Spurious Correlation Collapse**, or **OOD Extrapolation**, classifiers output saturated $\ge 95\%$ confidence while actual accuracy or recall collapses toward zero.

**TrustGuard** resolves this paradox by combining:
- **Attribution-Weighted $L_3$ Input & Correlation Drift ($S_{\text{attr}}$):** Filters out benign drift in low-importance features.
- **Cross-Architecture Inductive Disagreement ($S_{\text{dis}}$):** Exploits conflicting inductive biases between tree ensembles (XGBoost/HistGradientBoosting) and continuous neural networks (MLP).
- **Mondrian Split-Conformal Abstention ($S_{\text{conf}}$):** Tracks non-singleton prediction sets ($|C(x)| \neq 1$) at significance $\alpha = 0.10$ with finite-sample corrections.
- **Variational Bayesian Last Layer & Gradient Probes (D3M & GdScore):** Evaluates epistemic logit posterior disagreement and classification-layer pseudo-label gradient norms.

---

## 2. Four-Stage Research Roadmap

The 16-week research program is organized into four sequential stages (see [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)):

- **Stage 1 (Weeks 1–4) — [CURRENT / COMPLETED]:** Infrastructure, schema harmonization, 75/25 source train/calibration split, frozen model families (Tree, MLP, VBLL, Logistic Regression), 3-fold ECE probability calibration, finite-sample Mondrian conformal locking, and D3M bootstrap reference calibration.
- **Stage 2 (Weeks 5–8):** Baseline estimators (AC, DoC, ATC, CBPE, COTT) and model-aware signal benchmark on synthetic streams, OpenML Electricity, and TableShift datasets.
- **Stage 3 (Weeks 9–12):** Test-time adaptation (Quantile Alignment, BatchNorm/Entropy TTA, Filtered Pseudo-Labels, Saerens EM) and the Oracle Selection Audit to detect adversarial class collapse.
- **Stage 4 (Weeks 13–16):** Three-tier TrustGuard Operational Guardrail (benign suppression, collapse recovery, conformal escalation) and NIST AI 800-4 regulatory mapping.

---

## 3. Strict Ground-Truth Isolation Axiom

In SilentX, monitoring is strictly isolated from evaluation:
- **Monitors** receive *only* unlabeled feature batches ($X_{\text{unlabeled}}$) and frozen model states. They cannot inspect, receive, or import deployment ground-truth labels.
- **Evaluation Oracle (`src/evaluation/oracle.py`)** sequesters deployment labels in an evaluation vault, calculating true accuracy, actual degradation, AUROC, AUPRC, and rank correlations post-hoc after monitoring predictions are committed.

---

## 4. Environment Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Standard CPU laptop (8–16 GB RAM, no GPU required)

### Local Virtual Environment
Using [uv](https://github.com/astral-sh/uv) (recommended):
```powershell
# Create venv and install dependencies
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

Using standard Python:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 5. Execution Commands

### 1. Run the Automated Test Suite
```powershell
pytest -v tests/
```

### 2. Run the 20-Second Proof-of-Concept (Smoke Test)
```powershell
python run_poc.py
```
Outputs quantitative comparison across 5 shift regimes to `results/poc_summary.csv`.

### 3. Run Stage 1 Calibration Pipeline & Benchmark
```powershell
python run_phase1.py
```
Locks source probability calibration, Mondrian quantiles, D3M disagreement distributions, runs deployment stream evaluation, and generates `results/stage1_calibration_summary.json` and `results/stage1_evaluation.csv`.

---

## 6. Project Architecture

```
SilentX/
├── RESEARCH_BLUEPRINT.md            # Initial research specification
├── Silent Model Degradation...docx  # 16-week expanded research plan
├── README.md                        # Project documentation
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # Build and test configuration
├── Dockerfile                       # Container definition
├── run_poc.py                       # Fast 20s proof-of-concept runner
├── run_phase1.py                    # Stage 1 calibration & evaluation runner
├── configs/
│   └── experiment.yaml              # Global experimental parameters
├── docs/
│   ├── RESEARCH_DECISIONS.md        # Specification reconciliation & decisions log
│   ├── IMPLEMENTATION_ROADMAP.md    # Complete 16-week research roadmap
│   ├── DATASETS.md                  # Dataset catalog, schemas, and shift profiles
│   └── EXPERIMENT_PROTOCOL.md       # Metrics, splitting, and isolation protocol
├── src/
│   ├── data/
│   │   ├── synthetic_streams.py     # 6 controlled shift regimes generator
│   │   ├── dataset_registry.py      # Unified data contract & window partitioner
│   │   ├── real_datasets.py         # OpenML Electricity & CIC-IDS harmonizer
│   │   └── schema_validation.py     # Tabular schema and dtype validator
│   ├── models/
│   │   ├── frozen_pipeline.py       # Frozen preprocessor and multi-model pipeline
│   │   ├── calibration.py           # ECE check, isotonic CV, Mondrian conformal
│   │   └── vbll_model.py            # PyTorch Variational Bayesian Last Layer & D3M
│   ├── monitors/
│   │   ├── baselines.py             # KS, PSI, JSD, AC, DoC, ATC, CBPE
│   │   └── trustguard.py            # Attribution L3, Cross-Arch, Conformal, GdScore
│   └── evaluation/
│       └── oracle.py                # Sequestered ground-truth evaluation vault
├── tests/
│   ├── test_data_contracts.py       # Data integrity, reproducibility & schema tests
│   ├── test_models.py               # Preprocessor freezing, models & VBLL tests
│   ├── test_calibration.py          # ECE, Mondrian quantiles & D3M bootstrap tests
│   └── test_label_isolation.py      # Zero-leakage & Oracle metric correctness tests
└── results/
    ├── stage1_calibration_summary.json
    ├── stage1_evaluation.csv
    └── poc_summary.csv
```

---

## 7. Governance & Standards Alignment

SilentX aligns with:
- **NIST AI 800-4** (*Challenges to the Monitoring of Deployed AI Systems*, March 2026).
- **NIST AI Risk Management Framework (AI RMF 1.0)** — MEASURE and MANAGE functions.
- **OWASP Agent Control Standard** (September 2026).
