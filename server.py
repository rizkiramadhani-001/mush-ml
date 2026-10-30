"""HTTP service untuk inferensi MUSH (FastAPI + scikit-learn)."""

from __future__ import annotations

import os
from functools import lru_cache
from datetime import datetime, timezone
from typing import Annotated

import uvicorn
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import FORECAST_DAYS, MODEL_DIR
from model import forecast_series


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

    batch_id: int = Field(gt=0)
    horizons: list[Annotated[int, Field(gt=0)]] | None = None
    now: datetime | None = None

    @field_validator("now")
    @classmethod
    def _normalize_now(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)


app = FastAPI(title="MUSH ML", version="0.1.0")


@lru_cache(maxsize=32)
def load_model(batch_id: int) -> dict:
    path = MODEL_DIR / f"batch-{batch_id}.joblib"
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Model batch {batch_id} belum dilatih")
    artifact = joblib.load(path)
    if artifact["batch_id"] != batch_id:
        raise HTTPException(status_code=503, detail="Identitas model tidak sesuai batch")
    return artifact


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
    artifact = load_model(payload.batch_id)

    predictions = {
        key: forecast_series(trained, days, now)
        for key, trained in artifact["parameters"].items()
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
