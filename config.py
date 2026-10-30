import os
from pathlib import Path

FORECAST_DAYS = (1, 7)
TRAIN_RATIO = 0.8
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
