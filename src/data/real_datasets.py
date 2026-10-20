"""Real-world dataset interfaces and schema normalizers for SilentX.

Covers:
1. OpenML Electricity (ID 151) - Natural temporal concept drift.
2. TableShift Suite (HELOC, Voting, ACSIncome, Diabetes Readmission) - Domain and subpopulation shift.
3. CIC-IDS2017 / CSE-CIC-IDS2018 - Temporal network security flow harmonization.
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import datetime
import numpy as np
import pandas as pd


# Canonical 7 attack families for CIC-IDS2017/2018 harmonization
CANONICAL_ATTACK_FAMILIES = [
    "Benign",
    "Botnet",
    "Brute-Force FTP",
    "Brute-Force SSH",
    "DDoS",
    "DoS",
    "Web/Infiltration",
]


class OpenMLElectricityLoader:
    """Loader and chronological partitioner for OpenML Electricity (Dataset ID 151)."""

    FEATURE_COLS = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    TARGET_COL = "class"

    @classmethod
    def load(
        cls,
        cache_dir: str = ".cache/datasets",
        max_samples: Optional[int] = None,
        source_ratio: float = 0.25,
        source_train_ratio: float = 0.75,
        manifest_path: Optional[str] = "results/electricity_manifest.json",
    ) -> Dict[str, any]:
        """Load Electricity dataset, preserving strict chronological ordering.

        Chronological protocol:
        - The earliest source_ratio (default 25%) of the stream is reserved for the source domain.
        - Within the source domain, the earliest 75% is used for training and the remaining 25% for calibration.
        - The remaining 75% of the timeline forms sequential deployment windows.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "electricity_151.parquet")

        df = None
        is_real = False
        retrieval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                is_real = True
            except Exception:
                df = None

        if df is None:
            try:
                from sklearn.datasets import fetch_openml
                raw = fetch_openml(data_id=151, as_frame=True, parser="auto")
                df = raw.frame.copy()
                df.to_parquet(cache_file)
                is_real = True
            except Exception as e:
                # If network or OpenML is completely unreachable, fallback to faithful surrogate
                df = cls._generate_faithful_surrogate(n_samples=max_samples or 10000)
                is_real = False

        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples].copy()

        # Map binary target to {0, 1}: 'DOWN' -> 0, 'UP' -> 1
        target_series = df[cls.TARGET_COL].astype(str).str.strip().str.upper()
        y = (target_series == "UP").astype(int)
        X = df.drop(columns=[cls.TARGET_COL]).copy()

        # Ensure all features including 'day' are numeric float
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        n_total = len(df)
        n_source = int(n_total * source_ratio)
        n_train = int(n_source * source_train_ratio)

        X_train = X.iloc[:n_train].reset_index(drop=True)
        y_train = y.iloc[:n_train].reset_index(drop=True)

        X_cal = X.iloc[n_train:n_source].reset_index(drop=True)
        y_cal = y.iloc[n_train:n_source].reset_index(drop=True)

        X_deploy = X.iloc[n_source:].reset_index(drop=True)
        y_deploy = y.iloc[n_source:].reset_index(drop=True)

        # Write provenance manifest
        if manifest_path is not None:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            manifest = {
                "openml_identifier": 151,
                "dataset_name": "electricity",
                "retrieval_timestamp_utc": retrieval_timestamp,
                "is_real_download": is_real,
                "total_samples": n_total,
                "feature_columns": list(X.columns),
                "target_column": cls.TARGET_COL,
                "class_mapping": {"DOWN": 0, "UP": 1},
                "class_balance_source_train": {
                    "0_DOWN": int((y_train == 0).sum()),
                    "1_UP": int((y_train == 1).sum()),
                },
                "class_balance_deployment": {
                    "0_DOWN": int((y_deploy == 0).sum()),
                    "1_UP": int((y_deploy == 1).sum()),
                },
                "temporal_ordering_policy": "Strict chronological forward ordering without shuffling",
                "source_training_boundary": [0, n_train - 1],
                "source_calibration_boundary": [n_train, n_source - 1],
                "deployment_stream_boundary": [n_source, n_total - 1],
                "preprocessing_configuration": "StandardScaler fitted strictly on source_training split",
                "row_exclusions": 0,
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        return {
            "name": "OpenML_Electricity_151",
            "X_train": X_train,
            "y_train": y_train,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_deploy": X_deploy,
            "y_deploy": y_deploy,
            "feature_names": list(X.columns),
            "target_name": "class_UP",
            "n_total": n_total,
            "is_real": is_real,
        }

    @classmethod
    def _generate_faithful_surrogate(cls, n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
        """Deterministic temporal market surrogate matching Electricity schema and drift patterns."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 10 * np.pi, n_samples)

        day = (np.arange(n_samples) // 48) % 7
        period = (np.arange(n_samples) % 48) / 48.0
        nswdemand = 0.5 + 0.3 * np.sin(t) + 0.1 * rng.standard_normal(n_samples)
        vicdemand = 0.4 + 0.25 * np.sin(t + 0.5) + 0.1 * rng.standard_normal(n_samples)
        transfer = 0.2 * np.cos(t) + 0.05 * rng.standard_normal(n_samples)
        nswprice = 0.05 + 0.03 * (nswdemand**1.5) + 0.02 * rng.standard_normal(n_samples)
        vicprice = 0.04 + 0.03 * (vicdemand**1.5) + 0.02 * rng.standard_normal(n_samples)

        price_drift = 0.02 * np.sin(0.2 * t)
        prob_up = 1.0 / (1.0 + np.exp(-(2.5 * (nswprice - vicprice) + price_drift)))
        y_class = np.where(rng.random(n_samples) < prob_up, "UP", "DOWN")

        df = pd.DataFrame({
            "date": np.linspace(0.0, 1.0, n_samples),
            "day": day,
            "period": period,
            "nswprice": nswprice,
            "nswdemand": nswdemand,
            "vicprice": vicprice,
            "vicdemand": vicdemand,
            "transfer": transfer,
            cls.TARGET_COL: y_class,
        })
        df.attrs["source"] = "_surrogate"
        return df


class CICIDSSchemaHarmonizer:
    """Schema harmonizer for CIC-IDS2017 and CSE-CIC-IDS2018 network security datasets."""

    COLUMN_REPLACEMENTS = {
        "Destination Port": "Dst_Port",
        "Flow Duration": "Flow_Duration",
        "Total Fwd Packets": "Tot_Fwd_Pkts",
        "Total Backward Packets": "Tot_Bwd_Pkts",
        "Total Length of Fwd Packets": "TotLen_Fwd_Pkts",
        "Total Length of Bwd Packets": "TotLen_Bwd_Pkts",
    }

    LABEL_MAPPINGS = {
        "BENIGN": "Benign",
        "BOT": "Botnet",
        "FTP-PATATOR": "Brute-Force FTP",
        "FTP-BRUTEFORCE": "Brute-Force FTP",
        "SSH-PATATOR": "Brute-Force SSH",
        "SSH-BRUTEFORCE": "Brute-Force SSH",
        "DDOS": "DDoS",
        "DOS": "DoS",
        "INFILTRATION": "Web/Infiltration",
        "WEB ATTACK": "Web/Infiltration",
    }

    @classmethod
    def canonicalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize whitespace, case, and divergent exporter column naming."""
        df = df.copy()
        clean_cols = {}
        for c in df.columns:
            name = c.strip()
            name = cls.COLUMN_REPLACEMENTS.get(name, name)
            name = name.replace(" ", "_").replace("/", "_").replace(".", "_")
            clean_cols[c] = name
        df = df.rename(columns=clean_cols)
        return df

    @classmethod
    def map_attack_family(cls, raw_label: pd.Series) -> pd.Series:
        """Map raw vendor attack strings into the 7 canonical attack families."""
        cleaned = raw_label.astype(str).str.strip().str.upper()

        mapped = pd.Series("Benign", index=raw_label.index, dtype=str)
        for pattern, family in cls.LABEL_MAPPINGS.items():
            mask = cleaned.str.contains(pattern, na=False)
            mapped[mask] = family

        return mapped
