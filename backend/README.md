# CrisisLens — Backend

FastAPI server hosting the full ML pipeline: five components, several REST endpoints, and one background task.

---

## Overview

The backend is a Python 3.11 FastAPI application that:

- Loads four ML models into RAM once at startup
- Exposes REST endpoints for the React frontend to call
- Runs the full pipeline on incoming post batches
- Maintains an in-memory cluster store that the frontend polls every 3 seconds
- Generates a BART situation summary roughly every 15 minutes as a background task

All ML models run locally except BART, which is called via the HuggingFace Inference API.

---

## Project Structure

```
backend/
├── api/
│   ├── main.py             # FastAPI app, lifespan, CORS, model loading
│   └── routes/
│       └── pipeline.py     # All REST endpoints + in-memory state
├── components/             # One file per pipeline stage
│   ├── bouncer.py          # TF-IDF + SVM noise filter (runtime)
│   ├── deduplicator.py     # MiniLM embeddings + DBSCAN clustering
│   ├── detective.py        # BERT NER location extraction
│   ├── geocoder.py         # Chennai gazetteer coordinate lookup
│   └── editor.py           # BART situation summary via HuggingFace API
├── scripts/                # Offline setup scripts (run once)
│   ├── prepare_data.py     # Merge Kaggle + CrisisLexT26 into a unified CSV
│   ├── train_bouncer.py    # Train and save TF-IDF vectoriser + SVM
│   ├── populate_gazetteer.py # Query Nominatim to build the Chennai JSON
│   └── download_models.py  # Pre-cache MiniLM + BERT from HuggingFace
├── models/                 # Trained .pkl files (tracked via Git LFS for deployment)
│   ├── tfidf_vectoriser.pkl
│   └── svm_classifier.pkl
├── data/
│   └── gazetteers/
│       └── chennai_gazetteer.json  # 171 Chennai locations with coordinates
├── requirements.txt
└── Dockerfile              # HuggingFace Spaces deployment (see below)
```

---

## Pipeline Components

### Bouncer (`bouncer.py`)

**What it does:** filters non-disaster posts, passing on only those classified as disaster-related.

**Technology:** TF-IDF vectoriser + LinearSVC, trained on a combination of the Kaggle Disaster Tweets dataset and CrisisLexT26.

**Speed:** under 1 ms per post — it handles 100% of incoming traffic as the first gate.

**Key detail:** both the fitted `tfidf_vectoriser.pkl` and `svm_classifier.pkl` must be loaded at runtime. The vectoriser holds the vocabulary mapping (word → column index) that the SVM's decision rules depend on. A fresh vectoriser at runtime would reassign column indices and produce garbage predictions.

---

### Deduplicator (`deduplicator.py`)

**What it does:** groups semantically similar posts about the same incident into clusters, returning one representative per unique incident.

**Technology:** `sentence-transformers/all-MiniLM-L6-v2` for embeddings, DBSCAN for clustering.

**Rolling window:** only posts from the last 30 minutes participate in clustering. Older posts are retired and their embeddings explicitly deleted to prevent memory growth.

**DBSCAN parameters:**

- `eps=0.40` (cosine distance threshold)
- `min_samples=2`
- `metric='cosine'`

**Noise handling:** DBSCAN noise points (posts with no close neighbours) are **not** discarded. Each becomes a singleton cluster representing a unique incident — potentially the first report of a new emergency.

**Urgency formula:**

```
urgency_score = 0.4 × keyword_weight
              + 0.4 × cluster_density
              + 0.2 × recency_weight

CRITICAL: score ≥ 0.7
HIGH:     score ≥ 0.4
MODERATE: score < 0.4
```

---

### Detective (`detective.py`)

**What it does:** extracts location names from the representative post of each cluster.

**Technology:** `dslim/bert-base-NER` from HuggingFace.

**Key parameters:**

- `aggregation_strategy="simple"` — reconstructs subword tokens into complete words. Without it, "Koyambedu" becomes `["Ko", "##yam", "##bed", "##u"]` and the geocoder receives "Ko".
- Runs in a `ThreadPoolExecutor` — BERT inference is synchronous and takes 200–500 ms. Running it directly in an async route would block the event loop; the thread pool keeps FastAPI responsive.

---

### Geocoder (`geocoder.py`)

**What it does:** converts location strings to `(lat, lng)` coordinates.

**Technology:** local JSON dictionary lookup — zero network calls at runtime.

**Lookup strategy:**

1. Exact match after lowercasing (handles BERT capitalisation variance)
2. Fuzzy match via `difflib.get_close_matches()` (handles typos and partial names)
3. Null coordinates if not found — no dot placed on the map

**Gazetteer coverage:** 171 Chennai locations including neighbourhoods, hospitals, railway stations, bridges, waterways, major roads, and flood-prone areas. Built once by `populate_gazetteer.py`.

---

### Editor (`editor.py`)

**What it does:** generates a natural-language situation summary from all active clusters.

**Technology:** `facebook/bart-large-cnn` via the HuggingFace Inference API.

**Schedule:** runs roughly every 15 minutes as a background asyncio task — not per post.

**Input construction:** sorts clusters by urgency descending and truncates input at 900 tokens (BART's limit is 1,024; the buffer prevents silent truncation of high-priority incidents).

**Why the API and not local:** BART-large is ~1.6 GB. Loading it alongside BERT NER + MiniLM would add significant RAM. Offloading it to the Inference API costs a few seconds of latency per summary — acceptable for a 15-minute cycle.

---

## API Endpoints

| Method   | Endpoint             | Description                                       |
| -------- | -------------------- | ------------------------------------------------- |
| `POST`   | `/api/process`       | Run the pipeline on a list of posts               |
| `GET`    | `/api/clusters`      | Get all active clusters (the frontend polls this) |
| `DELETE` | `/api/clusters/{id}` | Remove a resolved incident                        |
| `DELETE` | `/api/clusters`      | Clear all incidents                               |
| `GET`    | `/api/report`        | Get the latest situation summary                  |
| `GET`    | `/api/health`        | Per-model load status                             |
| `GET`    | `/docs`              | Swagger interactive API docs                      |

### `POST /api/process` — request body

```json
{
  "tweets": [
    "Family trapped near Velachery bridge, water rising fast",
    "Flooding at Koyambedu bus stand, people stranded"
  ]
}
```

### `GET /api/clusters` — response format

```json
[
  {
    "cluster_id": "c_000",
    "representative_tweet": "Family trapped near Velachery bridge...",
    "tweet_count": 4,
    "source_tweets": ["...", "...", "...", "..."],
    "first_seen": "2024-01-01T10:00:00",
    "last_seen": "2024-01-01T10:22:00",
    "urgency_score": 0.87,
    "urgency_label": "CRITICAL",
    "locations": ["Velachery bridge"],
    "lat": 12.9802,
    "lng": 80.2229,
    "resolved_location": "Velachery, Chennai, Tamil Nadu, India"
  }
]
```

---

## Local Setup

### 1. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Download the ML models

```bash
python backend/scripts/download_models.py
```

Pre-caches MiniLM (~90 MB) and BERT NER (~400 MB) locally so the server never downloads at runtime.

### 3. Prepare training data and train the classifier

Download the datasets first (see the root README), then:

```bash
python backend/scripts/prepare_data.py
python backend/scripts/train_bouncer.py
```

Two `.pkl` files are written to `backend/models/`.

### 4. Build the gazetteer

```bash
python backend/scripts/populate_gazetteer.py
```

Takes ~8–10 minutes. Enforces a delay between Nominatim calls per their terms of service, and is crash-safe — re-running resumes where it stopped.

### 5. Configure the environment

```bash
echo "HF_API_TOKEN=hf_your_token_here" > backend/.env
```

### 6. Start the server

```bash
uvicorn backend.api.main:app --reload --port 8000
```

---

## Running with Docker

A `Dockerfile` is included — the same one used for the HuggingFace deployment. To build and run the container locally:

```bash
docker build -t crisislens-backend .
docker run -p 7860:7860 crisislens-backend
```

The image installs **CPU-only PyTorch** (to avoid the multi-gigabyte CUDA build) and bakes MiniLM + BERT into the image at build time, so the container starts cleanly with no runtime model download. The server listens on port `7860`.

---

## Deployment (HuggingFace Spaces)

The backend runs on a free HuggingFace **Docker** Space (`crisislens-backend-deployment`), which provides 16 GB of RAM — comfortable headroom for the ~1.1 GB the models require.

| Platform           | Free RAM | Verdict              |
| ------------------ | -------- | -------------------- |
| Render             | 512 MB   | Crashes on BERT load |
| Railway            | 512 MB   | Crashes on BERT load |
| HuggingFace Spaces | 16 GB    | Comfortable headroom |

**How it is deployed:**

1. The Space is created with the **Docker** SDK on CPU Basic (free), set to public.
2. The two `.pkl` files are tracked with **Git LFS** and pushed from a clean history (HuggingFace rejects plain binary blobs).
3. The `Dockerfile` installs CPU-only PyTorch and runs `download_models.py` at build time, baking MiniLM + BERT into the image. Because the models live in the image layer, they survive every sleep/wake restart without re-downloading.
4. Two settings are configured on the Space: `HF_API_TOKEN` (secret, for the BART summary) and `FRONTEND_URL` (variable, used to restrict CORS to the deployed frontend).

The Space sleeps after periods of inactivity and wakes automatically on the first incoming request, so no manual start/stop is needed.

---

## Environment Variables

| Variable       | Required | Description                                                       |
| -------------- | -------- | ----------------------------------------------------------------- |
| `HF_API_TOKEN` | Yes      | HuggingFace token for BART inference                              |
| `FRONTEND_URL` | No       | Frontend origin allowed by CORS (defaults to permissive if unset) |

---

## Requirements

See `requirements.txt` for pinned versions. Key dependencies:

```
fastapi==0.111.0              # Web framework
uvicorn[standard]==0.29.0     # ASGI server
scikit-learn==1.4.2           # TF-IDF + SVM
sentence-transformers==3.0.0  # MiniLM embeddings
transformers==4.41.0          # BERT NER
torch==2.3.0                  # PyTorch backend (CPU-only in deployment)
geopy==2.4.1                  # Nominatim (setup only)
python-dotenv==1.0.1          # .env file loading
```
