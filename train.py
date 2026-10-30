import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
from pydantic import BaseModel, Field, field_validator

from config import MODEL_DIR
from model import Sample, train_series
from server import SampleIn


class TrainingData(BaseModel):
    batch_id: int = Field(gt=0)
    samples: dict[str, list[SampleIn]]

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, value):
        if not value:
            raise ValueError("samples wajib berisi minimal satu parameter")
        return value


def train_file(dataset: Path, output_dir: Path = MODEL_DIR) -> Path:
    data = TrainingData.model_validate_json(dataset.read_text())
    artifact = {
        "batch_id": data.batch_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: train_series([Sample(item.t, item.value) for item in items])
            for key, items in data.samples.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"batch-{data.batch_id}.joblib"
    temporary = target.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(train_file(args.dataset, args.output_dir))
