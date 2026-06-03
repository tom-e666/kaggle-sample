import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

# ---- CONFIG (edit paths here) ----
PRED_PATH = r"C:\AI\Kaggle\outputs\catboost\pred_test.csv"
GT_PATH = r"C:\AI\Kaggle\titanic\data\gender_submission.csv"
ID_COL = "PassengerId"
TARGET_COL = "Survived"
TASK = "binary"  # "binary" | "multiclass" | "regression"
METRIC = "roc_auc"  # "roc_auc" | "accuracy" | "logloss" | "rmse"
MODEL = "catboost"  # "lightgbm" | "xgboost" | "catboost" | "tree"

def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
	pred_df = pd.read_csv(PRED_PATH)
	gt_df = pd.read_csv(GT_PATH)
	return pred_df, gt_df


def align_on_id(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
	merged = gt_df[[ID_COL, TARGET_COL]].merge(pred_df, on=ID_COL, how="inner")
	y_true = merged[TARGET_COL + "_x"].values
	y_pred = merged[TARGET_COL + "_y"].values
	return y_true, y_pred


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	if TASK == "regression":
		return mean_squared_error(y_true, y_pred, squared=False)
	if TASK == "multiclass":
		if METRIC == "logloss":
			return log_loss(y_true, y_pred)
		return accuracy_score(y_true, y_pred)
	if METRIC == "accuracy":
		return accuracy_score(y_true, (y_pred >= 0.5).astype(int))
	if METRIC == "logloss":
		return log_loss(y_true, y_pred)
	return roc_auc_score(y_true, y_pred)


def main() -> None:
	if not os.path.exists(PRED_PATH):
		raise FileNotFoundError(PRED_PATH)
	if not os.path.exists(GT_PATH):
		raise FileNotFoundError(GT_PATH)

	pred_df, gt_df = load_frames()
	y_true, y_pred = align_on_id(pred_df, gt_df)
	value = score(y_true, y_pred)
	print(f"metric={METRIC} value={value:.6f}")


if __name__ == "__main__":
	main()
