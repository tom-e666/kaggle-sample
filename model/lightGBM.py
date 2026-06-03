import os
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

# ---- CONFIG (edit paths here) ----
TRAIN_PATH = r"C:\AI\Kaggle\titanic\data\train.csv"
TEST_PATH = r"C:\AI\Kaggle\titanic\data\test.csv"
OUTPUT_DIR = r"C:\AI\Kaggle\outputs\lightgbm"

TARGET_COL = "Survived"
ID_COL = "PassengerId"
TASK = "binary"  # "binary" | "multiclass" | "regression"

N_SPLITS = 5
SEED = 42

# LightGBM params
LGB_PARAMS = {
	"n_estimators": 800,
	"learning_rate": 0.05,
	"num_leaves": 31,
	"subsample": 0.9,
	"colsample_bytree": 0.9,
	"random_state": SEED,
}


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df = pd.read_csv(TRAIN_PATH)
	test_df = pd.read_csv(TEST_PATH)
	return train_df, test_df


def build_splits(y: pd.Series):
	if TASK in {"binary", "multiclass"}:
		return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def one_hot_align(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
	combined = pd.get_dummies(combined, dummy_na=True)
	X = combined.iloc[: len(train_df)].copy()
	X_test = combined.iloc[len(train_df) :].copy()
	return X, X_test


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
	# LightGBM rejects feature names with JSON special characters.
	safe = df.columns.str.replace(r"[\[\]{}:\\,\"]", "_", regex=True)
	safe = safe.str.replace(r"\s+", "_", regex=True)
	df = df.copy()
	df.columns = safe
	return df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
	if TASK == "regression":
		rmse = mean_squared_error(y_true, y_pred, squared=False)
		return {"rmse": rmse}
	if TASK == "multiclass":
		acc = accuracy_score(y_true, y_pred)
		ll = log_loss(y_true, y_proba)
		return {"accuracy": acc, "logloss": ll}
	auc = roc_auc_score(y_true, y_proba)
	ll = log_loss(y_true, y_proba)
	return {"roc_auc": auc, "logloss": ll}


def log_metrics(path: str, fold: int | None, metrics: dict) -> None:
	prefix = "overall" if fold is None else f"fold_{fold}"
	line = prefix + ": " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
	with open(path, "a", encoding="utf-8") as f:
		f.write(line + "\n")


def main() -> None:
	ensure_dir(OUTPUT_DIR)
	log_path = os.path.join(OUTPUT_DIR, "metrics.txt")
	if os.path.exists(log_path):
		os.remove(log_path)

	train_df, test_df = load_data()
	y = train_df[TARGET_COL]

	X_raw = train_df.drop(columns=[TARGET_COL])
	X_test_raw = test_df.copy()

	if ID_COL in X_raw.columns:
		X_raw = X_raw.drop(columns=[ID_COL])
	if ID_COL in X_test_raw.columns:
		X_test_raw = X_test_raw.drop(columns=[ID_COL])

	X, X_test = one_hot_align(X_raw, X_test_raw)
	X = sanitize_columns(X)
	X_test = sanitize_columns(X_test)
	splitter = build_splits(y)

	fold_metrics = []
	for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
		X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
		y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

		if TASK == "regression":
			model = LGBMRegressor(**LGB_PARAMS)
			model.fit(X_tr, y_tr)
			pred = model.predict(X_va)
			metrics = evaluate(y_va.values, pred, None)
		elif TASK == "multiclass":
			params = {**LGB_PARAMS, "objective": "multiclass", "num_class": y.nunique()}
			model = LGBMClassifier(**params)
			model.fit(X_tr, y_tr)
			proba = model.predict_proba(X_va)
			pred = np.argmax(proba, axis=1)
			metrics = evaluate(y_va.values, pred, proba)
		else:
			params = {**LGB_PARAMS, "objective": "binary"}
			model = LGBMClassifier(**params)
			model.fit(X_tr, y_tr)
			proba = model.predict_proba(X_va)[:, 1]
			pred = (proba >= 0.5).astype(int)
			metrics = evaluate(y_va.values, pred, proba)

		fold_metrics.append(metrics)
		log_metrics(log_path, fold, metrics)

	avg_metrics = {
		k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()
	}
	log_metrics(log_path, None, avg_metrics)

	if TASK == "regression":
		final_model = LGBMRegressor(**LGB_PARAMS)
		final_model.fit(X, y)
		final_model.booster_.save_model(os.path.join(OUTPUT_DIR, "model.txt"))
		test_pred = final_model.predict(X_test)
	elif TASK == "multiclass":
		params = {**LGB_PARAMS, "objective": "multiclass", "num_class": y.nunique()}
		final_model = LGBMClassifier(**params)
		final_model.fit(X, y)
		final_model.booster_.save_model(os.path.join(OUTPUT_DIR, "model.txt"))
		test_proba = final_model.predict_proba(X_test)
		test_pred = np.argmax(test_proba, axis=1)
	else:
		params = {**LGB_PARAMS, "objective": "binary"}
		final_model = LGBMClassifier(**params)
		final_model.fit(X, y)
		final_model.booster_.save_model(os.path.join(OUTPUT_DIR, "model.txt"))
		test_pred = final_model.predict_proba(X_test)[:, 1]

	submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_pred})
	pred_path = os.path.join(OUTPUT_DIR, "pred_test.csv")
	submission.to_csv(pred_path, index=False)


if __name__ == "__main__":
	main()
