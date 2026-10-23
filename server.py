"""HTTP service untuk inferensi MUSH (FastAPI + scikit-learn)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import FORECAST_DAYS
from model import Sample, forecast_series


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SampleIn(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    t: datetime
    value: float

    @field_validator("t")
    @classmethod
    def _normalize(cls, value: datetime) -> datetime:
        return _to_utc(value)


class PredictRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    samples: dict[str, list[SampleIn]]
    horizons: list[Annotated[int, Field(gt=0)]] | None = None
    now: datetime | None = None

    @field_validator("samples")
    @classmethod
    def _not_empty(cls, value: dict[str, list[SampleIn]]) -> dict[str, list[SampleIn]]:
        if not value:
            raise ValueError("samples wajib berisi minimal satu parameter")
        return value

    @field_validator("now")
    @classmethod
    def _normalize_now(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)


app = FastAPI(title="MUSH ML", version="0.1.0")


@app.exception_handler(RequestValidationError)
async def validation_error(_request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "Payload tidak valid", "issues": jsonable_encoder(exc.errors())},
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    days = tuple(sorted(set(payload.horizons))) if payload.horizons else FORECAST_DAYS
    now = payload.now or datetime.now(timezone.utc)

    predictions = {
        key: forecast_series(
            [Sample(t=item.t, value=item.value) for item in items],
            days,
            now,
        )
        for key, items in payload.samples.items()
    }

    return {
        "model": "regresi_linear",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "predictions": predictions,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8081")),
    )
