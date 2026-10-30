import unittest
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi import HTTPException
from model import Sample, forecast_series, train_series
from server import PredictRequest, load_model, predict
from train import train_file

START = datetime(2026, 10, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def linear_samples(count: int, base: float = 40, step: float = 0.1) -> list[Sample]:
    return [
        Sample(t=START + index * HOUR, value=base + index * step)
        for index in range(count)
    ]


class ForecastTest(unittest.TestCase):
    def test_linear_trend_for_both_horizons(self):
        now = START + 47 * HOUR
        h1, h7 = forecast_series(train_series(linear_samples(48)), (1, 7), now)

        self.assertEqual(h1["nilai"], 47.1)
        self.assertEqual(h7["nilai"], 61.5)

    def test_metrics_use_holdout(self):
        now = START + 47 * HOUR
        forecast = forecast_series(train_series(linear_samples(48)), (1,), now)[0]

        self.assertEqual(forecast["metrik"]["mae"], 0.0)
        self.assertEqual(forecast["metrik"]["rmse"], 0.0)
        self.assertEqual(forecast["metrik"]["r2"], 1.0)
        self.assertEqual(forecast["metrik"]["n"], 10)

    def test_insufficient_training_data_rejected(self):
        for samples in ([], [Sample(START, 41)], [Sample(START, 41)] * 2):
            with self.assertRaises(ValueError):
                train_series(samples)

    def test_flat_series_has_no_r2(self):
        now = START + 9 * HOUR
        samples = [Sample(t=START + index * HOUR, value=41) for index in range(10)]
        forecast = forecast_series(train_series(samples), (1,), now)[0]

        self.assertEqual(forecast["nilai"], 41.0)
        self.assertIsNone(forecast["metrik"]["r2"])

    def test_saved_model_predicts_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.json"
            dataset.write_text(json.dumps({
                "batch_id": 1,
                "samples": {"suhu": [
                    {"t": sample.t.isoformat(), "value": sample.value}
                    for sample in reversed(linear_samples(48))
                ]},
            }))
            target = train_file(dataset, root)
            self.assertEqual(joblib.load(target)["parameters"]["suhu"]["sample_count"], 48)
            dataset.unlink()
            load_model.cache_clear()
            self.addCleanup(load_model.cache_clear)
            with patch("server.MODEL_DIR", root), patch(
                "model.LinearRegression.fit", side_effect=AssertionError("Inference must not train")
            ):
                request = PredictRequest(batch_id=1, now=START + 47 * HOUR)
                result = predict(request)
                self.assertEqual(result["predictions"]["suhu"][0]["nilai"], 47.1)
                with patch("server.joblib.load", side_effect=AssertionError("Must reuse cached model")):
                    self.assertEqual(predict(request), result)
                with self.assertRaises(HTTPException) as error:
                    predict(PredictRequest(batch_id=2))
                self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
