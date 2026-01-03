from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

# Reuse loading/cleaning helpers from the classification pipeline
from .classification_library import (
    Paths,
    _ensure_dirs,
    load_beijing_air_quality,
    clean_air_quality_df,
    add_time_features,
    _coerce_lag_hours,
)


def add_lag_features_for_regression(
    df: pd.DataFrame,
    lag_hours=(1, 3, 24),
    cols=("PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"),
) -> pd.DataFrame:
    """
    Adds lag features per station (if station exists) for regression use-cases.
    Unlike classification_library.add_lag_features, this version can include PM2.5.
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


def make_regression_target(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    horizon: int = 1,
    out_col: str = "y",
) -> pd.DataFrame:
    """
    Create supervised regression label: y(t) = target(t + horizon).
    This turns a time series problem into a tabular regression (one-step / multi-step ahead).

    NOTE: For forecasting, you will later compare this approach vs ARIMA.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Missing target_col={target_col}")

    if "station" in df.columns:
        df[out_col] = df.groupby("station", sort=False)[target_col].shift(-horizon)
    else:
        df[out_col] = df[target_col].shift(-horizon)

    return df


def time_split(
    df: pd.DataFrame,
    cutoff: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split to avoid leakage.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train_df = df[df["datetime"] < cutoff_ts].copy()
    test_df = df[df["datetime"] >= cutoff_ts].copy()
    return train_df, test_df


def _build_preprocess(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a sklearn ColumnTransformer (num + cat).
    Important: normalize missing values to np.nan.
    """
    feature_cols = list(X.columns)

    # Identify numeric/cat by dtype in X
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X = X.copy()

    # Numeric: coerce to float64
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    # Categorical: plain object, missing -> np.nan
    for c in cat_cols:
        X[c] = X[c].astype("object")
        X[c] = X[c].where(pd.notna(X[c]), np.nan)

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
    return pre, num_cols, cat_cols


def train_regressor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str = "y",
    drop_cols: tuple[str, ...] = ("datetime",),
    model_params: dict | None = None,
) -> dict:
    """
    Train a tabular regressor to predict y (future PM2.5) from lagged features + weather + time.
    """
    model_params = model_params or {}

    feature_cols = [c for c in train_df.columns if c not in set(drop_cols) | {y_col}]
    X_train = train_df[feature_cols].copy()
    y_train = pd.to_numeric(train_df[y_col], errors="coerce")

    X_test = test_df[feature_cols].copy()
    y_test = pd.to_numeric(test_df[y_col], errors="coerce")

    # Normalize missing markers
    X_train = X_train.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)
    X_test = X_test.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)

    # Drop rows without label
    train_mask = pd.notna(y_train)
    test_mask = pd.notna(y_test)

    pre, num_cols, cat_cols = _build_preprocess(X_train)

    reg = HistGradientBoostingRegressor(
        max_depth=model_params.get("max_depth", 6),
        learning_rate=model_params.get("learning_rate", 0.06),
        max_iter=model_params.get("max_iter", 400),
        random_state=model_params.get("random_state", 42),
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", reg)])
    pipe.fit(X_train.loc[train_mask], y_train.loc[train_mask])

    y_pred = pipe.predict(X_test.loc[test_mask])

    rmse = float(np.sqrt(mean_squared_error(y_test.loc[test_mask], y_pred)))
    mae = float(mean_absolute_error(y_test.loc[test_mask], y_pred))
    r2 = float(r2_score(y_test.loc[test_mask], y_pred))

    # sMAPE for robustness to small values
    denom = (np.abs(y_test.loc[test_mask].values) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.where(denom == 0, 0.0, np.abs(y_pred - y_test.loc[test_mask].values) / denom)) * 100.0)

    pred_df = pd.DataFrame({
        "datetime": test_df.loc[test_mask, "datetime"].values,
        "station": test_df.loc[test_mask, "station"].values if "station" in test_df.columns else None,
        "y_true": y_test.loc[test_mask].values,
        "y_pred": y_pred,
    })

    metrics = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "smape_pct": smape,
        "r2": r2,
        "feature_cols": feature_cols,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }
    return {"model": pipe, "metrics": metrics, "pred_df": pred_df}


def run_prepare_regression_dataset(
    paths: Paths,
    use_ucimlrepo: bool = True,
    raw_zip_path: str | None = None,
    lag_hours=(1, 3, 24),
    horizon: int = 1,
    target_col: str = "PM2.5",
) -> Path:
    """
    End-to-end: load -> clean -> time features -> lag features -> future target -> save parquet.
    Output: data/processed/dataset_for_regression.parquet
    """
    _ensure_dirs(paths.data_raw, paths.data_processed)

    df = load_beijing_air_quality(use_ucimlrepo=use_ucimlrepo, raw_zip_path=raw_zip_path)
    df = clean_air_quality_df(df)
    df = add_time_features(df)

    # lags include PM2.5 for regression
    df = add_lag_features_for_regression(df, lag_hours=lag_hours)

    # create y = PM2.5(t+h)
    df = make_regression_target(df, target_col=target_col, horizon=horizon, out_col="y")

    out = paths.data_processed / "dataset_for_regression.parquet"
    df.to_parquet(out, index=False)
    return out


def run_train_regression(
    paths: Paths,
    cutoff: str = "2017-01-01",
    model_out: str = "regressor.joblib",
    metrics_out: str = "regression_metrics.json",
    preds_out: str = "regression_predictions_sample.csv",
) -> dict:
    """
    Train regression model using the prepared dataset.
    """
    ds_path = paths.data_processed / "dataset_for_regression.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing: {ds_path}. Run regression prepare first.")

    df = pd.read_parquet(ds_path)
    df = df[df["datetime"].notna()].copy()

    train_df, test_df = time_split(df, cutoff=cutoff)
    out = train_regressor(train_df, test_df, y_col="y", drop_cols=("datetime",))

    # save artifacts
    joblib.dump(out["model"], paths.data_processed / model_out)

    with open(paths.data_processed / metrics_out, "w", encoding="utf-8") as f:
        json.dump(out["metrics"], f, ensure_ascii=False, indent=2)

    out["pred_df"].head(8000).to_csv(paths.data_processed / preds_out, index=False)

    return out
