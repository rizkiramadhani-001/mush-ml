# MUSH ML

Layanan inferensi untuk MUSH: regresi linear per parameter sensor dengan
horizon 1 dan 7 hari. Dibuat dengan FastAPI + scikit-learn, tidak menyentuh
database. Model dilatih di luar proses HTTP, lalu dimuat per batch saat
prediksi.

**Stack:** Python 3.11+ · FastAPI · scikit-learn · joblib

## Ringkasan

- Satu artifact model per batch (`models/batch-<id>.joblib`).
- Prediksi dari model tersimpan; request tidak mengirim ulang sampel.
- Artifact berisi estimator, origin waktu, jumlah sampel, dan metrik uji.
- Model di-cache di memori per batch; prediksi tidak pernah melatih ulang.
- Validasi payload dengan Pydantic; data tanpa zona waktu dianggap UTC.

## Alur

```
dataset JSON ──train.py──► models/batch-1.joblib
                                   │
mush-server ──POST /predict───────►│──► prediksi 1 & 7 hari
                                   │
                              GET /health
```

## Menjalankan

Kebutuhan: Python 3.11+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python server.py            # http://localhost:8081
```

Atau dengan uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8081
```

Health check:

```bash
curl http://localhost:8081/health
# {"ok":true}
```

## Melatih model

Model harus ada sebelum `/predict` bisa dipakai. Dataset berisi satu batch dan
sampel per parameter:

```json
{
  "batch_id": 1,
  "samples": {
    "suhu": [
      { "t": "2026-10-01T00:00:00Z", "value": 40.0 },
      { "t": "2026-10-01T01:00:00Z", "value": 40.1 }
    ],
    "tekanan": [
      { "t": "2026-10-01T00:00:00Z", "value": 12.4 }
    ],
    "gas": [
      { "t": "2026-10-01T00:00:00Z", "value": 418 }
    ]
  }
}
```

```bash
python train.py dataset.json
# models/batch-1.joblib
```

Opsi output direktori:

```bash
python train.py dataset.json --output-dir /var/lib/mush-ml/models
```

Aturan training:

- `batch_id` > 0 dan `samples` tidak kosong.
- Setiap parameter minimal 2 sampel dengan minimal 2 waktu berbeda.
- Data diurutkan kronologis; `x` = jam sejak sampel pertama.
- 80% awal untuk latih, 20% akhir untuk uji.
- Model final dilatih ulang pada seluruh data, lalu diekstrapolasi ke
  `now + 1 hari` dan `now + 7 hari`.
- Direktori `models/` tidak di-commit (lihat `.gitignore`).

## `POST /predict`

Body:

```json
{
  "batch_id": 1,
  "horizons": [1, 7],
  "now": "2026-10-23T06:00:00Z"
}
```

| Field | Wajib | Keterangan |
|---|---|---|
| `batch_id` | ya | Model yang dimuat: `models/batch-<id>.joblib` |
| `horizons` | tidak | Daftar hari positif; default `[1, 7]`; duplikat dibuang |
| `now` | tidak | Titik acuan prediksi; default waktu server |

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
      },
      {
        "hari": 7,
        "waktu": "2026-10-30T06:00:00Z",
        "nilai": 44.8,
        "metrik": { "mae": 0.22, "rmse": 0.24, "r2": -0.003, "n": 68 }
      }
    ],
    "tekanan": [],
    "gas": []
  }
}
```

- Key `predictions` mengikuti parameter yang ada di artifact.
- `nilai` dibulatkan 1 desimal; `mae`/`rmse` 2 desimal; `r2` 3 desimal.
- `metrik` bernilai `null` bila data uji kurang dari 2 titik.
- `r2` bernilai `null` bila variansi data uji nol.

### Status error

| Status | Kondisi |
|---|---|
| `400` | Payload tidak valid |
| `503` | Model batch belum dilatih atau identitas artifact tidak cocok |

```json
{ "error": "Payload tidak valid", "issues": [] }
```

```json
{ "detail": "Model batch 2 belum dilatih" }
```

## Konfigurasi

| Variabel | Default | Keterangan |
|---|---|---|
| `HOST` | `0.0.0.0` | Alamat bind uvicorn |
| `PORT` | `8081` | Port HTTP |
| `MODEL_DIR` | `./models` | Lokasi artifact `batch-<id>.joblib` |

`MODEL_DIR` dibaca saat modul dimuat. Model di-cache hingga 32 batch; setelah
melatih ulang artifact, restart proses agar versi baru dipakai.

## Integrasi `mush-server`

`mush-server` memanggil `POST /predict` dengan:

```json
{ "batch_id": 1, "now": "2026-10-23T06:00:00.000Z" }
```

Timeout 8 detik. Server memetakan prediksi `suhu` (°C), `tekanan` (Pa), dan
`gas` (ADC) ke response `/api/predict`. Bila layanan mati, model belum ada, atau
timeout, API mengembalikan `503 {"error":"Layanan ML tidak tersedia"}`.

## Pengujian

```bash
. .venv/bin/activate
python -m unittest -v test_model.py
```

Tes mencakup tren linear untuk kedua horizon, metrik holdout, penolakan data
kurang, `r2` untuk deret datar, serta inference dari model tersimpan tanpa
melatih ulang.

## Struktur

```
server.py       FastAPI, validasi Pydantic, cache model, endpoint
model.py        training, evaluasi, dan forecast regresi linear
train.py        CLI untuk melatih dataset JSON menjadi artifact joblib
config.py       horizon dan rasio latih/uji
test_model.py   unit test model dan server
requirements.txt
```

## Batasan

- Regresi linear cenderung mengekstrapolasi lurus; bukan model musiman.
- Model harus dilatih manual per batch; belum ada retraining otomatis.
- Belum ada autentikasi; batasi akses lewat jaringan privat.
- Belum ada Random Forest atau Isolation Forest.

## Repo terkait

- [MUSH Server](https://github.com/rizkiramadhani-001/mush-server) — API dan PostgreSQL
- [MUSH React](https://github.com/rizkiramadhani-001/mush-react) — dashboard Ionic React
- [Landing page](https://github.com/rizkiramadhani-001/landing)
