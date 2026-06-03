import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

# ---- CONFIG (edit paths here) ----
TRAIN_PATH = r"C:\AI\Kaggle\titanic\data\train.csv"
TEST_PATH = r"C:\AI\Kaggle\titanic\data\test.csv"
OUTPUT_DIR = r"C:\AI\Kaggle\outputs\catboost"

TARGET_COL = "Survived"
ID_COL = "PassengerId"
TASK = "binary"  # "binary" | "multiclass" | "regression"

N_SPLITS = 5
SEED = 42

# CatBoost params
CB_PARAMS = {
	"depth": 6,
	"learning_rate": 0.05,
	"iterations": 800,
	"loss_function": "Logloss",
	"eval_metric": "AUC",
	"random_seed": SEED,
	"verbose": False,
}


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df = pd.read_csv(TRAIN_PATH)
	test_df = pd.read_csv(TEST_PATH)
	return train_df, test_df


def get_cat_features(df: pd.DataFrame) -> List[int]:
	cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
	return [df.columns.get_loc(c) for c in cat_cols]


def build_splits(y: pd.Series):
	if TASK in {"binary", "multiclass"}:
		return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


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

	X = train_df.drop(columns=[TARGET_COL])
	X_test = test_df.copy()

	if ID_COL in X.columns:
		X = X.drop(columns=[ID_COL])
	if ID_COL in X_test.columns:
		X_test = X_test.drop(columns=[ID_COL])

	cat_features = get_cat_features(X)
	splitter = build_splits(y)

	fold_metrics = []
	for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
		X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
		y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

		if TASK == "regression":
			model = CatBoostRegressor(**CB_PARAMS)
			model.fit(X_tr, y_tr, cat_features=cat_features)
			pred = model.predict(X_va)
			metrics = evaluate(y_va.values, pred, None)
		elif TASK == "multiclass":
			params = {**CB_PARAMS, "loss_function": "MultiClass", "eval_metric": "MultiClass"}
			model = CatBoostClassifier(**params)
			model.fit(X_tr, y_tr, cat_features=cat_features)
			proba = model.predict_proba(X_va)
			pred = np.argmax(proba, axis=1)
			metrics = evaluate(y_va.values, pred, proba)
		else:
			params = {**CB_PARAMS, "loss_function": "Logloss", "eval_metric": "AUC"}
			model = CatBoostClassifier(**params)
			model.fit(X_tr, y_tr, cat_features=cat_features)
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
		final_model = CatBoostRegressor(**CB_PARAMS)
		final_model.fit(X, y, cat_features=cat_features)
	elif TASK == "multiclass":
		params = {**CB_PARAMS, "loss_function": "MultiClass", "eval_metric": "MultiClass"}
		final_model = CatBoostClassifier(**params)
		final_model.fit(X, y, cat_features=cat_features)
	else:
		params = {**CB_PARAMS, "loss_function": "Logloss", "eval_metric": "AUC"}
		final_model = CatBoostClassifier(**params)
		final_model.fit(X, y, cat_features=cat_features)

	model_path = os.path.join(OUTPUT_DIR, "model.cbm")
	final_model.save_model(model_path)

	if TASK == "regression":
		test_pred = final_model.predict(X_test)
	elif TASK == "multiclass":
		test_proba = final_model.predict_proba(X_test)
		test_pred = np.argmax(test_proba, axis=1)
	else:
		test_pred = final_model.predict_proba(X_test)[:, 1]

	submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_pred})
	pred_path = os.path.join(OUTPUT_DIR, "pred_test.csv")
	submission.to_csv(pred_path, index=False)


if __name__ == "__main__":
	main()
