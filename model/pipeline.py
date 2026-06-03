import os
import subprocess
import sys
from typing import List

# ---- CONFIG (edit paths here) ----
BASE_DIR = r"C:\AI\Kaggle\model"
PYTHON_EXE = sys.executable  # uses current interpreter

MODELS: List[str] = [
	"tree.py",
	"lightGBM.py",
	"XGBoost.py",
	"CatBoost.py",
]


def run_models() -> None:
	for name in MODELS:
		path = os.path.join(BASE_DIR, name)
		if not os.path.exists(path):
			print(f"[skip] {path} not found")
			continue
		print(f"[run] {name}")
		result = subprocess.run([PYTHON_EXE, path], cwd=BASE_DIR)
		if result.returncode != 0:
			raise RuntimeError(f"Model failed: {name} (exit {result.returncode})")
		print(f"[done] {name}")


if __name__ == "__main__":
	run_models()
