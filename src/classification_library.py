from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import zipfile
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier


# -----------------------------
# 1) Load data
# -----------------------------
def load_beijing_air_quality(
    use_ucimlrepo: bool = True,
    raw_zip_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Preferred: use_ucimlrepo=True (auto fetch UCI #501).
    Optional: if you manually downloaded the ZIP, pass raw_zip_path.
    """
    if use_ucimlrepo:
        ds = fetch_ucirepo(id=501)
        X = ds.data.features
        y = ds.data.targets
        if y is None or (isinstance(y, pd.DataFrame) and y.shape[1] == 0):
            df = X.copy()
        else:
            df = pd.concat([X, y], axis=1)
        return df

    if raw_zip_path is None:
        raise ValueError("If use_ucimlrepo=False you must provide raw_zip_path.")

    raw_zip_path = Path(raw_zip_path)
    if not raw_zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {raw_zip_path}")

    dfs: list[pd.DataFrame] = []
    with zipfile.ZipFile(raw_zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise ValueError("No CSV files found inside the ZIP.")

        for m in members:
            with zf.open(m) as f:
                dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    return df


# -----------------------------
# 2) Clean + datetime + rolling PM2.5(24h)
# -----------------------------
@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def clean_air_quality_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Replace common NA strings with np.nan
    - Parse datetime
    - Coerce numeric columns
    - Keep station / wd as plain object dtype (avoid pandas StringDtype + pd.NA issues)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Replace common missing markers
    df = df.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)

    # build datetime
    required = {"year", "month", "day", "hour"}
    if required.issubset(set(df.columns)):
        df["datetime"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
            errors="coerce",
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError("Cannot find time columns (year/month/day/hour) nor datetime.")

    # Coerce numeric columns
    numeric_candidates = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep station / wd as plain object (NOT pandas 'string')
    for col in ["station", "wd"]:
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(pd.notna(df[col]), np.nan)

    # Sort (important for rolling/lag)
    if "station" in df.columns:
        df = df.sort_values(["station", "datetime"])
    else:
        df = df.sort_values(["datetime"])

    return df


# -----------------------------
# 3) Build classification label from PM2.5 (24h mean)
# -----------------------------
AQI_CLASSES = [
    "Good",
    "Moderate",
    "Unhealthy_for_Sensitive_Groups",
    "Unhealthy",
    "Very_Unhealthy",
    "Hazardous",
]


def pm25_to_aqi_class(pm25_ug_m3: pd.Series) -> pd.Series:
    """
    PM2.5 breakpoints (µg/m³):
      Good:     0.0–9.0
      Moderate: 9.1–35.4
      USG:      35.5–55.4
      Unhealthy:55.5–125.4
      Very Unhealthy:125.5–225.4
      Hazardous:225.5+
    """
    x = pd.to_numeric(pm25_ug_m3, errors="coerce")
    bins = [-np.inf, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
    return pd.cut(
        x,
        bins=bins,
        labels=AQI_CLASSES,
        right=True,
        include_lowest=True,
    )


def add_pm25_24h_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 24-hour mean PM2.5 per station, then map to AQI class.
    """
    if "PM2.5" not in df.columns:
        raise ValueError("PM2.5 column not found; cannot build label.")

    df = df.copy()
    if "station" in df.columns:
        df["pm25_24h"] = (
            df.groupby("station")["PM2.5"]
            .rolling(window=24, min_periods=18)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["pm25_24h"] = df["PM2.5"].rolling(window=24, min_periods=18).mean()

    # keep as object; missing -> np.nan
    df["aqi_class"] = pm25_to_aqi_class(df["pm25_24h"]).astype("object")
    df["aqi_class"] = df["aqi_class"].where(pd.notna(df["aqi_class"]), np.nan)
    return df


# -----------------------------
# 4) Feature engineering (time features + optional lags)
# -----------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    df["dow"] = dt.dt.dayofweek.astype("int16")
    df["month"] = dt.dt.month.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def _coerce_lag_hours(lag_hours) -> tuple[int, ...]:
    """
    Papermill đôi khi truyền tuple/list qua parameters thành string.
    Hàm này nhận: str / list / tuple / int và trả về tuple[int, ...].
    """
    if lag_hours is None:
        return tuple()

    if isinstance(lag_hours, str):
        s = lag_hours.strip()
        try:
            lag_hours = ast.literal_eval(s)
        except Exception:
            s = s.strip("()[]")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            lag_hours = [int(p) for p in parts]

    if isinstance(lag_hours, (int, np.integer)):
        return (int(lag_hours),)

    try:
        return tuple(int(x) for x in lag_hours)
    except TypeError as e:
        raise TypeError(f"Unsupported lag_hours type: {type(lag_hours)}") from e


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: tuple[int, ...] | list[int] | str | int = (1, 3, 24),
    cols: tuple[str, ...] = ("PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"),
) -> pd.DataFrame:
    """
    Adds lag features per station (if station exists).
    """
    df = df.copy()
    base_cols = [c for c in cols if c in df.columns]
    lag_hours = _coerce_lag_hours(lag_hours)

    if not lag_hours or not base_cols:
        return df

    if "station" in df.columns:
        g = df.groupby("station", sort=False)
        for lag in lag_hours:
            for c in base_cols:
                df[f"{c}_lag{lag}"] = g[c].shift(lag)
    else:
        for lag in lag_hours:
            for c in base_cols:
                df[f"{c}_lag{lag}"] = df[c].shift(lag)

    return df


# -----------------------------
# 5) Build train/test split (time-based)
# -----------------------------
def time_split(
    df: pd.DataFrame,
    cutoff: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    train_df = df[df["datetime"] < cutoff_ts].copy()
    test_df = df[df["datetime"] >= cutoff_ts].copy()
    return train_df, test_df


# -----------------------------
# 6) Train classifier
# -----------------------------
def train_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "aqi_class",
) -> dict:
    """
    We exclude direct PM2.5 / pm25_24h from features to avoid leakage.

    IMPORTANT:
    - Convert pandas pd.NA -> np.nan before sklearn transformers.
    - Force numeric cols to float64 to avoid object arrays carrying pd.NA.
    """
    drop_cols = {"PM2.5", "pm25_24h", target_col, "datetime"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype("object")

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype("object")

    # Replace common NA strings early (defensive)
    X_train = X_train.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)
    X_test = X_test.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)

    # Build columns by actual dtype in X_train (NOT train_df)
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # --- Normalize missing values for sklearn ---
    # Numeric: coerce to float64 (pd.NA -> np.nan)
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype("float64")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype("float64")

    # Categorical: force plain object; missing -> np.nan
    for c in cat_cols:
        X_train[c] = X_train[c].astype("object")
        X_test[c] = X_test[c].astype("object")
        X_train[c] = X_train[c].where(pd.notna(X_train[c]), np.nan)
        X_test[c] = X_test[c].where(pd.notna(X_test[c]), np.nan)

    # Final safety sweep: pd.NA -> np.nan (should be no-op now)
    X_train = X_train.replace({pd.NA: np.nan})
    X_test = X_test.replace({pd.NA: np.nan})

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )

    model = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    train_mask = pd.notna(y_train)
    test_mask = pd.notna(y_test)
    model.fit(X_train.loc[train_mask], y_train.loc[train_mask])

    y_pred = model.predict(X_test.loc[test_mask])

    metrics = {
        "cutoff": str(test_df["datetime"].min()),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test.loc[test_mask], y_pred)),
        "f1_macro": float(f1_score(y_test.loc[test_mask], y_pred, average="macro")),
        "report": classification_report(y_test.loc[test_mask], y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test.loc[test_mask], y_pred, labels=AQI_CLASSES).tolist(),
        "labels": AQI_CLASSES,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
    }

    pred_df = pd.DataFrame({
        "datetime": test_df.loc[test_mask, "datetime"].values,
        "station": test_df.loc[test_mask, "station"].values if "station" in test_df.columns else None,
        "y_true": y_test.loc[test_mask].values,
        "y_pred": y_pred,
    })

    return {"model": model, "metrics": metrics, "pred_df": pred_df}


# -----------------------------
# 7) End-to-end pipeline helpers
# -----------------------------
def run_prepare(paths: Paths, use_ucimlrepo: bool = True, raw_zip_path: str | None = None, lag_hours=(1, 3, 24)) -> Path:
    _ensure_dirs(paths.data_raw, paths.data_processed)

    df = load_beijing_air_quality(use_ucimlrepo=use_ucimlrepo, raw_zip_path=raw_zip_path)
    df = clean_air_quality_df(df)
    df = add_pm25_24h_and_label(df)
    df = add_time_features(df)
    df = add_lag_features(df, lag_hours=lag_hours)

    out = paths.data_processed / "cleaned.parquet"
    df.to_parquet(out, index=False)
    return out


def run_train(paths: Paths, cutoff: str = "2017-01-01") -> dict:
    cleaned_path = paths.data_processed / "cleaned.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing: {cleaned_path}. Run prepare first.")

    df = pd.read_parquet(cleaned_path)
    df = df[df["datetime"].notna()].copy()

    snap_path = paths.data_processed / "dataset_for_clf.parquet"
    df.to_parquet(snap_path, index=False)

    train_df, test_df = time_split(df, cutoff=cutoff)
    out = train_classifier(train_df, test_df, target_col="aqi_class")

    metrics_path = paths.data_processed / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out["metrics"], f, ensure_ascii=False, indent=2)

    out["pred_df"].head(5000).to_csv(paths.data_processed / "predictions_sample.csv", index=False)

    return out
