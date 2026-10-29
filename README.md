# MUSH ML

Forecasting service for sensor data in MUSH fermentation monitoring.

**Stack:** Python, FastAPI, scikit-learn

## Portfolio snapshot

- Validated `POST /predict` API for sensor samples and forecast horizons
- Linear regression model with chronological train and test splits
- MAE, RMSE, and R² metrics in the response
- Unit tests for model behavior and edge cases

**Related repositories:** [MUSH Server](https://github.com/rizkiramadhani-001/mush-server) · [MUSH React](https://github.com/rizkiramadhani-001/mush-react)

Layanan inferensi untuk dashboard MUSH: regresi linear per parameter sensor
dengan horizon 1 hari dan 7 hari. Memakai FastAPI + scikit-learn, tidak
menyentuh database.

## Jalankan

Kebutuhan: Python 3.11+.

```bash
cd mush-ml
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python server.py            # default http://localhost:8081
```

Atau langsung dengan uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8081
```

`HOST` (default `0.0.0.0`) dan `PORT` (default `8081`) dibaca dari environment.

## `POST /predict`

Body:

```json
{
  "samples": {
    "suhu": [{ "t": "2026-10-22T00:00:00Z", "value": 41.1 }],
    "gas": [{ "t": "2026-10-22T00:00:00Z", "value": 418 }],
    "tekanan": [{ "t": "2026-10-22T00:00:00Z", "value": 12.4 }]
  },
  "horizons": [1, 7],
  "now": "2026-10-23T06:00:00Z"
}
```

- `samples` wajib; setiap parameter minimal 2 titik agar menghasilkan prediksi.
- `horizons` opsional, default `[1, 7]`.
- `now` opsional, default waktu server.
- Waktu tanpa zona dianggap UTC.

Response:

```json
{
  "model": "regresi_linear",
  "generated_at": "2026-10-23T06:00:00Z",
  "predictions": {
    "suhu": [
      {
        "hari": 1,
        "waktu": "2026-10-24T06:00:00Z",
        "nilai": 41.2,
        "metrik": { "mae": 0.22, "rmse": 0.24, "r2": -0.003, "n": 68 }
      }
    ]
  }
}
```

Parameter dengan kurang dari 2 sampel menghasilkan array kosong. `metrik`
bernilai `null` bila data uji kurang dari 2; `r2` bernilai `null` bila variansi
data uji nol.

Payload tidak valid mengembalikan `400 {"error":"Payload tidak valid","issues":[...]}`.

## Metode

- `sklearn.linear_model.LinearRegression` dengan `x` = jam sejak sampel pertama.
- Data diurutkan kronologis, 80% awal untuk latih dan 20% akhir untuk uji.
- Metrik dari `sklearn.metrics`: MAE, RMSE, R².
- Model final dilatih ulang pada seluruh data, lalu diekstrapolasi ke
  `now + 1 hari` dan `now + 7 hari`.

## Tes

```bash
. .venv/bin/activate
python -m unittest -v test_model.py
```

## `GET /health`

```json
{ "ok": true }
```

## Struktur

```
config.py       horizon dan rasio latih/uji
model.py        regresi linear + metrik (scikit-learn)
server.py       FastAPI + validasi Pydantic
test_model.py   unit test model
requirements.txt
```