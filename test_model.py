import unittest
from datetime import datetime, timedelta, timezone

from model import Sample, forecast_series

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
        h1, h7 = forecast_series(linear_samples(48), (1, 7), now)

        self.assertEqual(h1["nilai"], 47.1)
        self.assertEqual(h7["nilai"], 61.5)

    def test_metrics_use_holdout(self):
        now = START + 47 * HOUR
        forecast = forecast_series(linear_samples(48), (1,), now)[0]

        self.assertEqual(forecast["metrik"]["mae"], 0.0)
        self.assertEqual(forecast["metrik"]["rmse"], 0.0)
        self.assertEqual(forecast["metrik"]["r2"], 1.0)
        self.assertEqual(forecast["metrik"]["n"], 10)

    def test_fewer_than_two_samples_returns_empty(self):
        self.assertEqual(forecast_series([], (1,), START), [])
        self.assertEqual(forecast_series([Sample(t=START, value=41)], (1,), START), [])

    def test_flat_series_has_no_r2(self):
        now = START + 9 * HOUR
        samples = [Sample(t=START + index * HOUR, value=41) for index in range(10)]
        forecast = forecast_series(samples, (1,), now)[0]

        self.assertEqual(forecast["nilai"], 41.0)
        self.assertIsNone(forecast["metrik"]["r2"])


if __name__ == "__main__":
    unittest.main()
