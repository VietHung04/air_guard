from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier


# ------------------------------------------------------------
# Reuse project primitives if available; otherwise fallback
# ------------------------------------------------------------
try:
    # project uses src.classification_library in notebooks
    from src.classification_library import AQI_CLASSES, time_split  # type: ignore
except Exception:  # pragma: no cover
    AQI_CLASSES = [
        "Good",
        "Moderate",
        "Unhealthy_for_Sensitive_Groups",
        "Unhealthy",
        "Very_Unhealthy",
        "Hazardous",
    ]

    def time_split(df: pd.DataFrame, cutoff: str = "2017-01-01"):
        cutoff_ts = pd.Timestamp(cutoff)
        train_df = df[df["datetime"] < cutoff_ts].copy()
        test_df = df[df["datetime"] >= cutoff_ts].copy()
        return train_df, test_df


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True)
class SemiDataConfig:
    target_col: str = "aqi_class"
    cutoff: str = "2017-01-01"
    random_state: int = 42
    leakage_cols: Tuple[str, ...] = ("PM2.5", "pm25_24h", "datetime")


@dataclass(frozen=True)
class SelfTrainingConfig:
    tau: float = 0.90
    max_iter: int = 10
    min_new_per_iter: int = 20
    val_frac: float = 0.20


@dataclass(frozen=True)
class CoTrainingConfig:
    tau: float = 0.90
    max_iter: int = 10
    max_new_per_iter: int = 500
    min_new_per_iter: int = 20
    val_frac: float = 0.20


# -----------------------------
# Utilities
# -----------------------------
def _normalize_missing(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace(["NA", "N/A", "na", "null", "None", ""], np.nan)
    X = X.replace({pd.NA: np.nan})
    return X


def build_feature_columns(df: pd.DataFrame, cfg: SemiDataConfig) -> List[str]:
    drop = set(cfg.leakage_cols) | {cfg.target_col}
    return [c for c in df.columns if c not in drop]


def _infer_num_cat_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    Xn = _normalize_missing(X.copy())
    num_cols, cat_cols = _infer_num_cat_cols(Xn)

    # numeric to float
    for c in num_cols:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce").astype("float64")

    # categorical to object, keep np.nan
    for c in cat_cols:
        Xn[c] = Xn[c].astype("object")
        Xn[c] = Xn[c].where(pd.notna(Xn[c]), np.nan)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),

        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, num_cols, cat_cols


def make_base_classifier(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=random_state,
    )


def make_pipeline(X: pd.DataFrame, random_state: int = 42) -> Tuple[Pipeline, Dict]:
    pre, num_cols, cat_cols = build_preprocess(X)
    model = make_base_classifier(random_state=random_state)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    info = {"numeric_cols": num_cols, "categorical_cols": cat_cols}
    return pipe, info


def _align_proba_to_labels(proba: np.ndarray, classes_: np.ndarray, labels: List[str]) -> np.ndarray:
    """
    Ensure proba columns align with global AQI_CLASSES, even when some classes
    are missing in training split (common under label scarcity).
    """
    out = np.zeros((proba.shape[0], len(labels)), dtype=float)
    class_to_pos = {str(c): i for i, c in enumerate(classes_)}
    for j, lab in enumerate(labels):
        if lab in class_to_pos:
            out[:, j] = proba[:, class_to_pos[lab]]
        else:
            out[:, j] = 0.0
    # re-normalize to sum=1 (avoid all-zero rows)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return out / row_sum


def mask_labels_time_aware(
    df: pd.DataFrame,
    cfg: SemiDataConfig,
    missing_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Simulate label scarcity: randomly hide a fraction of labels in TRAIN ONLY.
    Keeps test labels intact for evaluation.
    """
    out = df.copy()
    train_df, _ = time_split(out, cutoff=cfg.cutoff)

    labeled_idx = train_df.index[pd.notna(train_df[cfg.target_col])].to_numpy()
    if labeled_idx.size == 0 or missing_fraction <= 0:
        out["is_labeled"] = pd.notna(out[cfg.target_col])
        return out

    rng = np.random.default_rng(cfg.random_state)
    n_mask = int(np.floor(missing_fraction * labeled_idx.size))
    n_mask = max(0, min(n_mask, labeled_idx.size))
    mask_idx = rng.choice(labeled_idx, size=n_mask, replace=False)

    out.loc[mask_idx, cfg.target_col] = np.nan
    out["is_labeled"] = pd.notna(out[cfg.target_col])
    return out


def _split_train_val_labeled(
    train_df: pd.DataFrame,
    cfg: SemiDataConfig,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    labeled_idx = train_df.index[pd.notna(train_df[cfg.target_col])].to_numpy()
    rng = np.random.default_rng(cfg.random_state)
    rng.shuffle(labeled_idx)
    n_val = int(np.floor(val_frac * labeled_idx.size))
    val_idx = labeled_idx[:n_val]
    fit_idx = labeled_idx[n_val:]
    return fit_idx, val_idx


def aqi_severity(aqi_class: str) -> int:
    try:
        return int(AQI_CLASSES.index(aqi_class))
    except Exception:
        return -1


def add_alert_columns(
    df_pred: pd.DataFrame,
    pred_col: str = "y_pred",
    severe_from: str = "Unhealthy",
) -> pd.DataFrame:
    out = df_pred.copy()
    thresh = aqi_severity(severe_from)
    out["severity_rank"] = out[pred_col].map(aqi_severity).astype("int16")
    out["is_alert"] = (out["severity_rank"] >= thresh).astype("int8")
    return out


# -----------------------------
# Self-training
# -----------------------------
class SelfTrainingAQIClassifier:
    """
    Explicit self-training loop (teaching-friendly, easy to inspect).
    """

    def __init__(self, data_cfg: SemiDataConfig, st_cfg: SelfTrainingConfig):
        self.data_cfg = data_cfg
        self.st_cfg = st_cfg
        self.model_: Optional[Pipeline] = None
        self.info_: Dict = {}
        self.history_: List[Dict] = []

    def fit(self, df: pd.DataFrame) -> "SelfTrainingAQIClassifier":
        df = df.copy()
        train_df, _ = time_split(df, cutoff=self.data_cfg.cutoff)

        feat_cols = build_feature_columns(train_df, self.data_cfg)
        X_all = _normalize_missing(train_df[feat_cols].copy())
        y_all = train_df[self.data_cfg.target_col].astype("object")

        fit_idx, val_idx = _split_train_val_labeled(train_df, self.data_cfg, self.st_cfg.val_frac)
        unlabeled_idx = train_df.index[pd.isna(y_all)].to_numpy()

        pipe, info = make_pipeline(X_all, random_state=self.data_cfg.random_state)
        self.info_ = {"feature_cols": feat_cols, **info}

        y_work = y_all.copy()

        for it in range(1, self.st_cfg.max_iter + 1):
            # fit on current labeled pool
            pipe.fit(X_all.loc[fit_idx], y_work.loc[fit_idx])

            # val on real labels only
            y_val_true = y_all.loc[val_idx]
            y_val_pred = pipe.predict(X_all.loc[val_idx])
            val_acc = float(accuracy_score(y_val_true, y_val_pred))
            val_f1 = float(f1_score(y_val_true, y_val_pred, average="macro"))

            # pseudo-label from unlabeled pool
            if unlabeled_idx.size > 0:
                proba_raw = pipe.predict_proba(X_all.loc[unlabeled_idx])
                proba = _align_proba_to_labels(proba_raw, pipe.named_steps["model"].classes_, AQI_CLASSES)
                max_prob = proba.max(axis=1)
                y_hat = np.array(AQI_CLASSES, dtype=object)[proba.argmax(axis=1)]
                pick_mask = max_prob >= float(self.st_cfg.tau)
                picked = unlabeled_idx[pick_mask]
                picked_labels = y_hat[pick_mask]
            else:
                picked = np.array([], dtype=int)
                picked_labels = np.array([], dtype=object)

            n_new = int(picked.size)
            self.history_.append({
                "iter": it,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "unlabeled_pool": int(unlabeled_idx.size),
                "new_pseudo": n_new,
                "tau": float(self.st_cfg.tau),
            })

            if n_new < int(self.st_cfg.min_new_per_iter):
                break

            # add pseudo labels into training pool
            y_work.loc[picked] = picked_labels
            fit_idx = np.unique(np.concatenate([fit_idx, picked]))

            picked_set = set(picked.tolist())
            unlabeled_idx = np.array([i for i in unlabeled_idx if i not in picked_set], dtype=int)

        self.model_ = pipe
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        feat_cols = self.info_["feature_cols"]
        X = _normalize_missing(df[feat_cols].copy())
        return self.model_.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        feat_cols = self.info_["feature_cols"]
        X = _normalize_missing(df[feat_cols].copy())
        proba_raw = self.model_.predict_proba(X)
        return _align_proba_to_labels(proba_raw, self.model_.named_steps["model"].classes_, AQI_CLASSES)


def run_self_training(
    df: pd.DataFrame,
    data_cfg: SemiDataConfig,
    st_cfg: SelfTrainingConfig,
) -> Dict:
    st = SelfTrainingAQIClassifier(data_cfg=data_cfg, st_cfg=st_cfg).fit(df)

    _, test_df = time_split(df.copy(), cutoff=data_cfg.cutoff)
    y_test = test_df[data_cfg.target_col].astype("object")
    mask = pd.notna(y_test)

    feat_cols = build_feature_columns(df, data_cfg)
    X_test = _normalize_missing(test_df.loc[mask, feat_cols].copy())
    y_pred = st.model_.predict(X_test)

    pred_df = pd.DataFrame({
        "datetime": test_df.loc[mask, "datetime"].values,
        "station": test_df.loc[mask, "station"].values if "station" in test_df.columns else None,
        "y_true": y_test.loc[mask].values,
        "y_pred": y_pred,
    })

    test_metrics = {
        "cutoff": str(test_df["datetime"].min()),
        "n_train": int((test_df["datetime"] < pd.Timestamp(data_cfg.cutoff)).sum()),
        "n_test": int(mask.sum()),
        "accuracy": float(accuracy_score(y_test.loc[mask], y_pred)),
        "f1_macro": float(f1_score(y_test.loc[mask], y_pred, average="macro")),
        "report": classification_report(y_test.loc[mask], y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test.loc[mask], y_pred, labels=AQI_CLASSES).tolist(),
        "labels": AQI_CLASSES,
        "feature_cols": st.info_.get("feature_cols", []),
        "categorical_cols": st.info_.get("categorical_cols", []),
        "numeric_cols": st.info_.get("numeric_cols", []),
    }

    return {
        "model": st.model_,
        "history": st.history_,
        "test_metrics": test_metrics,
        "pred_df": pred_df,
        "model_info": st.info_,
    }


# -----------------------------
# Co-training (two views)
# -----------------------------
def make_default_views(feature_cols: Iterable[str]) -> Tuple[List[str], List[str]]:
    cols = list(feature_cols)
    v2_patterns = ("station", "wd", "hour_", "dow", "month", "is_weekend", "year", "day", "hour")
    view2 = [c for c in cols if any(p in c for p in v2_patterns)]
    view1 = [c for c in cols if c not in set(view2)]

    if len(view1) == 0 or len(view2) == 0:
        mid = max(1, len(cols) // 2)
        view1, view2 = cols[:mid], cols[mid:]
    return view1, view2


class CoTrainingAQIClassifier:
    """
    Two-view co-training:
      - train 2 models on 2 feature views
      - each iter: both pseudo-label unlabeled; take confident ones >= tau
      - conflict: prefer agreement; else choose higher confidence
    """

    def __init__(
        self,
        data_cfg: SemiDataConfig,
        ct_cfg: CoTrainingConfig,
        view1_cols: Optional[List[str]] = None,
        view2_cols: Optional[List[str]] = None,
    ):
        self.data_cfg = data_cfg
        self.ct_cfg = ct_cfg
        self.view1_cols = view1_cols
        self.view2_cols = view2_cols

        self.model1_: Optional[Pipeline] = None
        self.model2_: Optional[Pipeline] = None
        self.info_: Dict = {}
        self.history_: List[Dict] = []

    def fit(self, df: pd.DataFrame) -> "CoTrainingAQIClassifier":
        df = df.copy()
        train_df, _ = time_split(df, cutoff=self.data_cfg.cutoff)

        feat_cols = build_feature_columns(train_df, self.data_cfg)
        v1, v2 = make_default_views(feat_cols)
        if self.view1_cols is not None:
            v1 = list(self.view1_cols)
        if self.view2_cols is not None:
            v2 = list(self.view2_cols)

        X1_all = _normalize_missing(train_df[v1].copy())
        X2_all = _normalize_missing(train_df[v2].copy())
        y_all = train_df[self.data_cfg.target_col].astype("object")

        fit_idx, val_idx = _split_train_val_labeled(train_df, self.data_cfg, self.ct_cfg.val_frac)
        unlabeled_idx = train_df.index[pd.isna(y_all)].to_numpy()

        pipe1, info1 = make_pipeline(X1_all, random_state=self.data_cfg.random_state)
        pipe2, info2 = make_pipeline(X2_all, random_state=self.data_cfg.random_state)

        self.info_ = {
            "view1_cols": v1,
            "view2_cols": v2,
            "view1_numeric_cols": info1["numeric_cols"],
            "view1_categorical_cols": info1["categorical_cols"],
            "view2_numeric_cols": info2["numeric_cols"],
            "view2_categorical_cols": info2["categorical_cols"],
        }

        y_work = y_all.copy()

        for it in range(1, self.ct_cfg.max_iter + 1):
            pipe1.fit(X1_all.loc[fit_idx], y_work.loc[fit_idx])
            pipe2.fit(X2_all.loc[fit_idx], y_work.loc[fit_idx])

            # validation via late-fusion, aligned to AQI_CLASSES
            p1_val_raw = pipe1.predict_proba(X1_all.loc[val_idx])
            p2_val_raw = pipe2.predict_proba(X2_all.loc[val_idx])
            p1_val = _align_proba_to_labels(p1_val_raw, pipe1.named_steps["model"].classes_, AQI_CLASSES)
            p2_val = _align_proba_to_labels(p2_val_raw, pipe2.named_steps["model"].classes_, AQI_CLASSES)

            p_avg = (p1_val + p2_val) / 2.0
            y_val_pred = np.array(AQI_CLASSES, dtype=object)[p_avg.argmax(axis=1)]
            y_val_true = y_all.loc[val_idx]

            val_acc = float(accuracy_score(y_val_true, y_val_pred))
            val_f1 = float(f1_score(y_val_true, y_val_pred, average="macro"))

            if unlabeled_idx.size == 0:
                self.history_.append({
                    "iter": it,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                    "unlabeled_pool": 0,
                    "new_pseudo": 0,
                    "tau": float(self.ct_cfg.tau),
                })
                break

            # pseudo-label proposals
            p1_u_raw = pipe1.predict_proba(X1_all.loc[unlabeled_idx])
            p2_u_raw = pipe2.predict_proba(X2_all.loc[unlabeled_idx])
            p1_u = _align_proba_to_labels(p1_u_raw, pipe1.named_steps["model"].classes_, AQI_CLASSES)
            p2_u = _align_proba_to_labels(p2_u_raw, pipe2.named_steps["model"].classes_, AQI_CLASSES)

            max1 = p1_u.max(axis=1)
            max2 = p2_u.max(axis=1)
            y1 = np.array(AQI_CLASSES, dtype=object)[p1_u.argmax(axis=1)]
            y2 = np.array(AQI_CLASSES, dtype=object)[p2_u.argmax(axis=1)]

            cand1 = unlabeled_idx[max1 >= float(self.ct_cfg.tau)]
            cand2 = unlabeled_idx[max2 >= float(self.ct_cfg.tau)]
            picked = np.unique(np.concatenate([cand1, cand2]))

            if picked.size > int(self.ct_cfg.max_new_per_iter):
                rng = np.random.default_rng(self.data_cfg.random_state + it)
                picked = rng.choice(picked, size=int(self.ct_cfg.max_new_per_iter), replace=False)

            n_new = int(picked.size)
            self.history_.append({
                "iter": it,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "unlabeled_pool": int(unlabeled_idx.size),
                "new_pseudo": n_new,
                "tau": float(self.ct_cfg.tau),
            })

            if n_new < int(self.ct_cfg.min_new_per_iter):
                break

            idx_to_pos = {int(idx): pos for pos, idx in enumerate(unlabeled_idx)}
            new_labels: Dict[int, str] = {}
            for idx in picked:
                idx_i = int(idx)
                pos = idx_to_pos[idx_i]
                if y1[pos] == y2[pos]:
                    new_labels[idx_i] = str(y1[pos])
                else:
                    new_labels[idx_i] = str(y1[pos] if max1[pos] >= max2[pos] else y2[pos])

            y_work.loc[list(new_labels.keys())] = list(new_labels.values())
            fit_idx = np.unique(np.concatenate([fit_idx, np.array(list(new_labels.keys()), dtype=int)]))

            picked_set = set(new_labels.keys())
            unlabeled_idx = np.array([i for i in unlabeled_idx if int(i) not in picked_set], dtype=int)

        self.model1_ = pipe1
        self.model2_ = pipe2
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model1_ is None or self.model2_ is None:
            raise RuntimeError("Models are not fitted yet.")

        v1 = self.info_["view1_cols"]
        v2 = self.info_["view2_cols"]
        X1 = _normalize_missing(df[v1].copy())
        X2 = _normalize_missing(df[v2].copy())

        p1_raw = self.model1_.predict_proba(X1)
        p2_raw = self.model2_.predict_proba(X2)
        p1 = _align_proba_to_labels(p1_raw, self.model1_.named_steps["model"].classes_, AQI_CLASSES)
        p2 = _align_proba_to_labels(p2_raw, self.model2_.named_steps["model"].classes_, AQI_CLASSES)

        p = (p1 + p2) / 2.0
        return np.array(AQI_CLASSES, dtype=object)[p.argmax(axis=1)]


def run_co_training(
    df: pd.DataFrame,
    data_cfg: SemiDataConfig,
    ct_cfg: CoTrainingConfig,
    view1_cols: Optional[List[str]] = None,
    view2_cols: Optional[List[str]] = None,
) -> Dict:
    ct = CoTrainingAQIClassifier(
        data_cfg=data_cfg,
        ct_cfg=ct_cfg,
        view1_cols=view1_cols,
        view2_cols=view2_cols,
    ).fit(df)

    _, test_df = time_split(df.copy(), cutoff=data_cfg.cutoff)
    y_test = test_df[data_cfg.target_col].astype("object")
    mask = pd.notna(y_test)

    test_labeled = test_df.loc[mask].copy()
    y_pred = ct.predict(test_labeled)

    pred_df = pd.DataFrame({
        "datetime": test_labeled["datetime"].values,
        "station": test_labeled["station"].values if "station" in test_labeled.columns else None,
        "y_true": y_test.loc[mask].values,
        "y_pred": y_pred,
    })

    test_metrics = {
        "cutoff": str(test_df["datetime"].min()),
        "n_train": int((test_df["datetime"] < pd.Timestamp(data_cfg.cutoff)).sum()),
        "n_test": int(mask.sum()),
        "accuracy": float(accuracy_score(y_test.loc[mask], y_pred)),
        "f1_macro": float(f1_score(y_test.loc[mask], y_pred, average="macro")),
        "report": classification_report(y_test.loc[mask], y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test.loc[mask], y_pred, labels=AQI_CLASSES).tolist(),
        "labels": AQI_CLASSES,
        "feature_cols": list(ct.info_.get("view1_cols", [])) + list(ct.info_.get("view2_cols", [])),
        "categorical_cols": [],
        "numeric_cols": [],
    }

    return {
        "model1": ct.model1_,
        "model2": ct.model2_,
        "history": ct.history_,
        "test_metrics": test_metrics,
        "pred_df": pred_df,
        "model_info": ct.info_,
    }
