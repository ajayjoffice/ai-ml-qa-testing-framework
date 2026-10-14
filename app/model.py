from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python -m training.train` first."
        )
    return joblib.load(MODEL_PATH)
