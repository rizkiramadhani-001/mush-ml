"""Regresi linear per parameter sensor untuk MUSH.

Data diurutkan kronologis, 80% awal untuk latih dan 20% akhir untuk uji.
Model final dilatih ulang pada seluruh data, lalu diekstrapolasi ke horizon
satu dan tujuh hari.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FORECAST_DAYS, TRAIN_RATIO


@dataclass(frozen=True)
class Sample:
    t: datetime
    value: float


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float | None
    n: int


def _evaluate(xs: np.ndarray, ys: np.ndarray, model: LinearRegression) -> Metrics | None:
    n = len(ys)
    if n < 2:
        return None

    predicted = model.predict(xs)
    r2 = None if float(np.var(ys)) <= 1e-12 else round(float(r2_score(ys, predicted)), 3)

    return Metrics(
        mae=round(float(mean_absolute_error(ys, predicted)), 2),
        rmse=round(float(np.sqrt(mean_squared_error(ys, predicted))), 2),
        r2=r2,
        n=n,
    )


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def train_series(samples: list[Sample]) -> dict:
    if len(samples) < 2:
        raise ValueError("Training membutuhkan minimal 2 sampel per parameter")

    ordered = sorted(samples, key=lambda sample: sample.t)
    origin = ordered[0].t
    if ordered[-1].t == origin:
        raise ValueError("Training membutuhkan minimal 2 waktu berbeda")

    xs = np.array(
        [[(sample.t - origin).total_seconds() / 3600] for sample in ordered],
        dtype=float,
    )
    ys = np.array([sample.value for sample in ordered], dtype=float)

    split = max(2, int(len(ordered) * TRAIN_RATIO))
    train = LinearRegression().fit(xs[:split], ys[:split])
    metrics = _evaluate(xs[split:], ys[split:], train)

    model = LinearRegression().fit(xs, ys)
    return {
        "estimator": model,
        "origin": origin,
        "last_sample": ordered[-1].t,
        "sample_count": len(ordered),
        "metrics": None if metrics is None else asdict(metrics),
    }


def forecast_series(
    trained: dict,
    days: tuple[int, ...] = FORECAST_DAYS,
    now: datetime | None = None,
) -> list[dict]:
    at = now or datetime.now(timezone.utc)
    model = trained["estimator"]
    origin = trained["origin"]

    forecasts = []
    for day in days:
        target = at + timedelta(days=day)
        hours = (target - origin).total_seconds() / 3600
        value = float(model.predict(np.array([[hours]]))[0])
        forecasts.append(
            {
                "hari": day,
                "waktu": _iso(target),
                "nilai": round(value, 1),
                "metrik": trained["metrics"],
            }
        )
    return forecasts
