import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TARGET_COL = "Tau_ms"
PAPER_COL = "Paper_ID"

KNOWN_CATEGORICAL_COLS = [
    "Paper_ID",
    "Channel",
    "Solvent",
    "Process",
    "Ion_type",
    "wt",
    "polymer",
    "Cation",
    "Anion",
    "Electrode_type",
]

BASE_INPUT_COLS = [
    "Channel",
    "Solvent",
    "Concentration_mg_ml",
    "Process",
    "Spin_RPM",
    "Annealing_temp_C",
    "Annealing_time_h",
    "Ion_type",
    "wt",
    "polymer",
    "Ion_diffusion",
    "Ion_viscosity",
    "Anion_radius",
    "Cation_radius",
    "Cation",
    "Anion",
    "Gate_voltage_V",
    "Drain_voltage_V",
    "Gate_pulse_width_ms",
    "Pulse_number",
    "Electrode_type",
    "Operating_temp_C",
]

DISPLAY_COLS = [
    "Channel", "polymer", "Cation", "Anion", "Ion_type", "Gate_voltage_V",
    "Gate_pulse_width_ms", "Pulse_number", "Annealing_temp_C", "Predicted_Tau_ms", "Tau_ms"
]


def force_string_keep_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.nan if pd.isna(x) else str(x))


def sanitize_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean raw spreadsheet-like input while preserving categorical missing values."""
    df = df.copy()

    if PAPER_COL not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": PAPER_COL})

    if PAPER_COL in df.columns:
        df[PAPER_COL] = df[PAPER_COL].astype(str)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    elif require_target:
        raise ValueError(f"'{TARGET_COL}' column is required for training.")

    for col in df.columns:
        if col == TARGET_COL:
            continue

        if col in KNOWN_CATEGORICAL_COLS:
            df[col] = force_string_keep_nan(df[col])
            continue

        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            original_notna = df[col].notna().sum()
            converted = pd.to_numeric(df[col], errors="coerce")
            converted_notna = converted.notna().sum()

            if original_notna > 0 and converted_notna / original_notna >= 0.70:
                df[col] = converted
            else:
                df[col] = force_string_keep_nan(df[col])

    return df.replace([np.inf, -np.inf], np.nan)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the physics-informed features used in the notebook/presentation."""
    df = df.copy()

    if PAPER_COL in df.columns:
        df["paper_year"] = pd.to_numeric(
            df[PAPER_COL].astype(str).str.extract(r"(20\d{2})")[0],
            errors="coerce",
        )

    numeric_like_cols = [
        "Concentration_mg_ml",
        "Spin_RPM",
        "Annealing_temp_C",
        "Annealing_time_h",
        "Ion_diffusion",
        "Ion_viscosity",
        "Anion_radius",
        "Cation_radius",
        "Gate_voltage_V",
        "Drain_voltage_V",
        "Gate_pulse_width_ms",
        "Pulse_number",
        "Vth_V",
        "On_off_ratio",
        "Vth_window_V",
        "Operating_temp_C",
    ]

    for col in numeric_like_cols:
        if col in df.columns:
            value = pd.to_numeric(df[col], errors="coerce")
            df[f"abs_{col}"] = value.abs()
            df[f"log1p_abs_{col}"] = np.log1p(value.abs())

    if {"Gate_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()

        df["gate_pulse_dose"] = gate_v * pulse_w
        df["log1p_gate_pulse_dose"] = np.log1p(df["gate_pulse_dose"].clip(lower=0))

        if "Pulse_number" in df.columns:
            pulse_n = pd.to_numeric(df["Pulse_number"], errors="coerce").abs()
            df["total_gate_dose"] = gate_v * pulse_w * pulse_n
            df["log1p_total_gate_dose"] = np.log1p(df["total_gate_dose"].clip(lower=0))

    if {"Drain_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        drain_v = pd.to_numeric(df["Drain_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()

        df["drain_pulse_dose"] = drain_v * pulse_w
        df["log1p_drain_pulse_dose"] = np.log1p(df["drain_pulse_dose"].clip(lower=0))

    if {"Gate_voltage_V", "Drain_voltage_V"}.issubset(df.columns):
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        drain_v = pd.to_numeric(df["Drain_voltage_V"], errors="coerce").abs()
        df["voltage_ratio"] = gate_v / (drain_v + 1e-9)

    if {"Ion_diffusion", "Ion_viscosity"}.issubset(df.columns):
        diff = pd.to_numeric(df["Ion_diffusion"], errors="coerce")
        visc = pd.to_numeric(df["Ion_viscosity"], errors="coerce")

        df["ion_mobility_proxy"] = diff / (visc + 1e-9)
        df["log1p_ion_diffusion"] = np.log1p(diff.clip(lower=0))
        df["log1p_ion_viscosity"] = np.log1p(visc.clip(lower=0))

    if {"Anion_radius", "Cation_radius"}.issubset(df.columns):
        anion_r = pd.to_numeric(df["Anion_radius"], errors="coerce")
        cation_r = pd.to_numeric(df["Cation_radius"], errors="coerce")

        df["radius_sum"] = anion_r + cation_r
        df["radius_diff_abs"] = (anion_r - cation_r).abs()
        df["radius_ratio"] = anion_r / (cation_r + 1e-9)

    if {"Concentration_mg_ml", "Gate_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        conc = pd.to_numeric(df["Concentration_mg_ml"], errors="coerce")
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()

        df["concentration_gate_dose"] = conc * gate_v * pulse_w
        df["log1p_concentration_gate_dose"] = np.log1p(df["concentration_gate_dose"].clip(lower=0))

    if {"Annealing_temp_C", "Annealing_time_h"}.issubset(df.columns):
        temp = pd.to_numeric(df["Annealing_temp_C"], errors="coerce")
        time_h = pd.to_numeric(df["Annealing_time_h"], errors="coerce")

        df["annealing_thermal_budget"] = temp * time_h
        df["log1p_annealing_thermal_budget"] = np.log1p(df["annealing_thermal_budget"].clip(lower=0))

    return df.replace([np.inf, -np.inf], np.nan)


class KFoldTargetEncoderDF(BaseEstimator, TransformerMixin):
    """K-fold target encoding while retaining original categorical columns."""

    def __init__(self, cols=None, smoothing=10, n_splits=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = X.copy()
        y = np.asarray(y)
        self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist() if self.cols is None else list(self.cols)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}

        for col in self.cols_:
            tmp = pd.DataFrame({"cat": X[col].astype(str).fillna("Missing"), "target": y})
            stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            X[f"{col}_te"] = (
                X[col].astype(str).fillna("Missing").map(self.maps_.get(col, {})).fillna(self.global_mean_)
            )
        return X

    def fit_transform(self, X, y=None, **fit_params):
        X = X.copy()
        y = np.asarray(y)
        self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist() if self.cols is None else list(self.cols)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}

        for col in self.cols_:
            new_col = f"{col}_te"
            X[new_col] = self.global_mean_
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            for train_idx, valid_idx in kf.split(X):
                train_part = X.iloc[train_idx]
                tmp = pd.DataFrame({"cat": train_part[col].astype(str).fillna("Missing"), "target": y[train_idx]})
                stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
                smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
                X.loc[X.index[valid_idx], new_col] = (
                    X.iloc[valid_idx][col].astype(str).fillna("Missing").map(smooth).fillna(self.global_mean_).values
                )

            tmp_all = pd.DataFrame({"cat": X[col].astype(str).fillna("Missing"), "target": y})
            stats_all = tmp_all.groupby("cat")["target"].agg(["mean", "count"])
            smooth_all = (stats_all["mean"] * stats_all["count"] + self.global_mean_ * self.smoothing) / (stats_all["count"] + self.smoothing)
            self.maps_[col] = smooth_all.to_dict()

        return X


def prepare_training_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    df = sanitize_dataframe(df_raw, require_target=True)
    df = add_engineered_features(df)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()

    tau_floor = df[TARGET_COL].quantile(0.01)
    df = df[df[TARGET_COL] >= tau_floor].copy()

    y = np.log1p(df[TARGET_COL].values.astype(float))
    X = df.drop(columns=[TARGET_COL])
    if PAPER_COL in X.columns:
        X = X.drop(columns=[PAPER_COL])
    X = X.replace([np.inf, -np.inf], np.nan)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    target_encoded_cols = [f"{col}_te" for col in categorical_cols]
    numeric_cols = [col for col in X.columns if col not in categorical_cols] + target_encoded_cols

    metadata = {
        "tau_floor_1pct": float(tau_floor),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_columns": X.columns.tolist(),
        "base_input_cols": BASE_INPUT_COLS,
    }
    return X, y, metadata


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str], fast_demo: bool = True) -> Pipeline:
    n_extra, n_rf, n_xgb = (120, 100, 120) if fast_demo else (800, 600, 700)
    xgb_lr = 0.05 if fast_demo else 0.035
    stack_cv = 3 if fast_demo else 5
    parallel_jobs = 1 if fast_demo else -1

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    stack_model = StackingRegressor(
        estimators=[
            ("extra", ExtraTreesRegressor(n_estimators=n_extra, random_state=42, max_features=0.8, min_samples_leaf=1, n_jobs=parallel_jobs)),
            ("rf", RandomForestRegressor(n_estimators=n_rf, random_state=42, max_features=0.8, min_samples_leaf=1, n_jobs=parallel_jobs)),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=n_xgb,
                    learning_rate=xgb_lr,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=parallel_jobs,
                ),
            ),
        ],
        final_estimator=RidgeCV(),
        cv=stack_cv,
        n_jobs=parallel_jobs,
    )

    return Pipeline([
        ("target_encoder", KFoldTargetEncoderDF(cols=categorical_cols, smoothing=10, n_splits=5, random_state=42)),
        ("preprocess", preprocessor),
        ("model", stack_model),
    ])


def build_input_schema(df_raw: pd.DataFrame, feature_columns: List[str]) -> Dict:
    df = sanitize_dataframe(df_raw, require_target=False)
    schema = {"categorical": {}, "numeric": {}}

    for col in BASE_INPUT_COLS:
        if col not in df.columns:
            continue
        if col in KNOWN_CATEGORICAL_COLS:
            values = sorted([str(v) for v in df[col].dropna().unique().tolist()])
            schema["categorical"][col] = values[:200]
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() == 0:
                default = 0.0
                min_v = 0.0
                max_v = 1.0
            else:
                default = float(series.median())
                min_v = float(series.min())
                max_v = float(series.max())
            schema["numeric"][col] = {"min": min_v, "max": max_v, "default": default}

    schema["feature_columns"] = feature_columns
    return schema


def align_features(raw_input_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    df = sanitize_dataframe(raw_input_df, require_target=False)
    df = add_engineered_features(df)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    if PAPER_COL in df.columns:
        df = df.drop(columns=[PAPER_COL])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[feature_columns].replace([np.inf, -np.inf], np.nan)


def predict_tau(model: Pipeline, raw_input_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X_input = align_features(raw_input_df, feature_columns)
    pred_log = np.asarray(model.predict(X_input)).reshape(-1)
    pred_tau = np.expm1(pred_log)
    out = raw_input_df.copy()
    out["Predicted_log1p_Tau"] = pred_log
    out["Predicted_Tau_ms"] = pred_tau
    return out


def train_and_save(data_path: str, model_dir: str, fast_demo: bool = True) -> Dict:
    os.makedirs(model_dir, exist_ok=True)
    raw = pd.read_excel(data_path)
    X, y, metadata = prepare_training_data(raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(metadata["categorical_cols"], metadata["numeric_cols"], fast_demo=fast_demo)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    summary = {
        "model_type": "StackingRegressor(ExtraTrees + RandomForest + XGBoost, RidgeCV final estimator)",
        "target": "log1p(Tau_ms)",
        "n_rows_after_filtering": int(len(X)),
        "n_features": int(X.shape[1]),
        "r2_log_holdout": float(r2_score(y_test, pred)),
        "rmse_log_holdout": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae_log_holdout": float(mean_absolute_error(y_test, pred)),
        "fast_demo_model": bool(fast_demo),
        **metadata,
    }

    # Refit on all available filtered data for the demo app.
    final_pipe = build_pipeline(metadata["categorical_cols"], metadata["numeric_cols"], fast_demo=fast_demo)
    final_pipe.fit(X, y)

    model_path = os.path.join(model_dir, "retention_model.joblib")
    joblib.dump(final_pipe, model_path)

    schema = build_input_schema(raw, metadata["feature_columns"])
    schema_path = os.path.join(model_dir, "input_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(model_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Candidate table for reverse design: predict all cleaned rows.
    base_cols = [col for col in BASE_INPUT_COLS if col in raw.columns]
    candidate_raw = raw[base_cols + ([TARGET_COL] if TARGET_COL in raw.columns else [])].copy()
    pred_candidates = predict_tau(final_pipe, candidate_raw[base_cols], metadata["feature_columns"])
    if TARGET_COL in candidate_raw.columns:
        pred_candidates[TARGET_COL] = candidate_raw[TARGET_COL]
    pred_candidates.to_csv(os.path.join(model_dir, "candidate_predictions.csv"), index=False, encoding="utf-8-sig")

    return summary


def load_artifacts(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "retention_model.joblib"))
    with open(os.path.join(model_dir, "input_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    with open(os.path.join(model_dir, "training_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    return model, schema, summary
