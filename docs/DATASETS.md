# SilentX — Unified Dataset Architecture & Shift Profiles

**Project:** SilentX  
**System:** TrustGuard  
**Version:** 1.0.0 (Stage 1)

---

## 1. Overview and Data Boundary Contracts

In the SilentX research program, all datasets conform to a standardized contract governed by the `DatasetRegistry`. The interface enforces:

1. **Source Splitting:** Strict separation of source data into **75% source training** (for model fitting and preprocessing statistics) and **25% held-out source calibration** (for isotonic probability calibration, conformal quantiles, and reference disagreement distributions).
2. **Preprocessing Isolation:** All imputers, standardizers, and categorical encodings are fitted **strictly on the 75% source training split** and frozen. Frozen transformers are applied unchanged to calibration and deployment observations.
3. **Target Label Sequestering:** Target deployment labels are stored in a dedicated evaluation vault accessible **only** to the post-hoc evaluation Oracle. No monitor receives or inspects deployment labels.
4. **Chronological & Official Integrity:** For streaming datasets (Electricity) and canonical benchmarks (TableShift), temporal ordering and official domain partitions are preserved; random shuffling across domain boundaries is strictly prohibited.

---

## 2. Dataset Catalog & Specifications

### 2.1 Synthetic Ground-Truth Suite (`Synth-6Stream`)
- **Identifier:** `synthetic_stream`
- **Source:** Deterministic NumPy generator (`src/data/synthetic_streams.py`)
- **Volume:** $N = 10,000$ records (configurable for quick smoke test: $N=2,500$), $d = 10$ features (3 causal drivers $X_0, X_1, X_2$; 1 correlated proxy $X_3$; 6 noise features $X_4 \dots X_9$).
- **Shift Regimes Implemented:**
  1. `in_distribution` (ID): $P_t(X, Y) = P_{\text{source}}(X, Y)$.
  2. `harmless_drift`: $P_t(X_{\text{noise}}) \neq P_{\text{source}}(X_{\text{noise}})$, while causal features and $P(Y \mid X)$ remain invariant. True $\Delta \text{Acc} \approx 0.0$.
  3. `correlation_collapse`: Feature correlation between proxy $X_3$ and causal $X_0$ inverts/collapses, pushing samples off the source training manifold. $\Delta \text{Acc} > 0.05$.
  4. `prior_shift`: Target class balance shifts from 50/50 to 80/20 via rejection sampling while $P(X \mid Y)$ remains invariant.
  5. `concept_shift`: Decision hyperplane rotates; $P_t(Y \mid X) \neq P_{\text{source}}(Y \mid X)$ with minimal marginal $P(X)$ change.
  6. `confidence_collapse`: Extreme out-of-distribution extrapolation on causal features paired with inverted concept. The frozen model outputs confidences $\ge 0.95$ while true accuracy collapses toward $0\%$.
- **Role in Research:** Deterministic testbed isolating why confidence proxies (CBPE, ATC) fail while model-aware signals (D3M, GdScore, Attribution $L_3$) succeed.

### 2.2 Temporal Stream: OpenML Electricity (`elec2`)
- **Identifier:** `openml_electricity` (Dataset ID 151)
- **Source:** Australian New South Wales Electricity Market (OpenML ID: 151 / `sklearn.datasets.fetch_openml`)
- **Volume:** 45,312 sequential instances; 8 continuous and categorical attributes (date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer).
- **Target:** Direction of price change (`UP` vs `DOWN`).
- **Shift Mechanism:** Natural temporal drift over 2.5 years of market deregulation.
- **Partitioning Protocol:** Chronological sequential split: first 25% (weeks 1–32) for source training (75%) and calibration (25%); subsequent 75% partitioned into ordered deployment windows ($m = 500$ observations).

### 2.3 Network Intrusion: CIC-IDS2017 & CSE-CIC-IDS2018
- **Identifier:** `cic_ids_2017_2018`
- **Source:** Canadian Institute for Cybersecurity (UNB)
- **Features:** 77 canonicalized flow attributes (e.g., flow duration, inter-arrival times, packet lengths, sub-flow statistics). Column names normalized across different flow exporter versions.
- **Attack Families:** 7 shared classes containing $\ge 100$ flows:
  1. Benign
  2. Botnet
  3. Brute-Force FTP
  4. Brute-Force SSH
  5. DDoS
  6. DoS
  7. Web/Infiltration
- **Evaluation Protocol:** Bidirectional cross-domain transfer:
  - Direction A: Train on CIC-IDS2017 $\to$ Deploy on CSE-CIC-IDS2018.
  - Direction B: Train on CSE-CIC-IDS2018 $\to$ Deploy on CIC-IDS2017.
- **Shift Mechanism:** Multi-year adversarial and background network evolution, severe class imbalance, test-time adaptation collapse.
- **Memory & Access Control:** Chunked CSV reading / compressed Parquet cache. Sampled development mode (10,000 flows/domain) and full benchmark mode.

### 2.4 Socioeconomic & Financial Benchmarks: TableShift Suite
- **HELOC (FICO Credit Risk):**
  - TableShift `heloc`. 38 features, 9,690 total rows. Predefined partitions: 2,220 train, 278 validation, 278 ID test, 6,914 OOD test.
  - Shift Mechanism: Subpopulation and severe prior shift partitioned by external risk score.
- **Voting:**
  - TableShift `voting`. 380 features, 70,038 rows. Predefined geographic domain shift (Southern US OOD test stream of 23,103 rows).
- **ACSIncome & ACSUnemployment (folktables / TableShift):**
  - Source: US Census Public Use Microdata Sample (PUMS).
  - ACSIncome: Evaluates pure target-semantics concept shift by contrasting California 2018 vs 2024 records with shifted policy income thresholds.
  - ACSUnemployment: 1,664,500 rows. Managed via chunked streaming to comply with 8–16 GB RAM laptop constraints.
- **Diabetes Readmission:**
  - TableShift `diabetes_readmission`. Tabular electronic health records. Evaluates demographic subpopulation shift (patients aged $>85$).

---

## 3. Resource & Execution Constraints

To maintain full compatibility with a standard CPU laptop (8–16 GB RAM, no GPU required):

1. **Development Samples vs Full Benchmarks:**
   - Default smoke-test and development runs use bounded subsamples (e.g., $N=2,500$ synthetic, $N=5,000$ Electricity, $N=10,000$ CIC-IDS).
   - Full benchmark runs read sequentially or in chunks, never holding raw uncompressed multi-gigabyte data in RAM simultaneously.
2. **Caching Strategy:**
   - Downloaded and processed datasets are cached in `.cache/datasets/` in Parquet or compressed Feather format.
3. **Offline / Network-Restricted Operation:**
   - The pipeline checks local cache before attempting network fetches.
   - If an external dataset is unreachable or licensing requires credentialized download (e.g., FICO HELOC), the system provides clear diagnostic instructions and falls back gracefully to the deterministic synthetic and cached streams without breaking the test suite.
