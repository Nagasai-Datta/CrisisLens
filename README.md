# CrisisLens

**Real-time crisis intelligence from social media for disaster response.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![React](https://img.shields.io/badge/React-18-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

CrisisLens is an end-to-end pipeline that reads social-media posts during a natural disaster, filters out the noise, groups duplicate reports into distinct incidents, pins them to a map, and summarises the overall situation — turning a flood of raw posts into a short, structured view of what is happening and where.

---

## Live Demo

**▶ Access the live dashboard here: https://mango-desert-052139500.4.azurestaticapps.net**

| Resource           | Link                                                             |
| ------------------ | ---------------------------------------------------------------- |
| Live dashboard     | https://mango-desert-052139500.4.azurestaticapps.net             |
| Backend API        | https://nagasaidatta-crisislens-backend-deployment.hf.space      |
| API docs (Swagger) | https://nagasaidatta-crisislens-backend-deployment.hf.space/docs |

> **Note on first load:** the backend runs on a free HuggingFace Space that sleeps after periods of inactivity. If the dashboard shows its components as offline when you first open it, the backend is simply waking up — this takes about a minute, after which the status turns green automatically. No action is required.

---

## Why

Chennai floods in some years badly enough to disrupt the city. In events like the 2015 floods, the bottleneck was not a lack of information — social media carried thousands of real-time reports: roads underwater, people stranded, requests for help. The hard part was the manual effort of separating actionable signal from the sheer volume of noise, fast enough to act on it.

CrisisLens is a prototype that automates that filtering and aggregation step, so that a stream of unstructured posts becomes a short list of distinct, located incidents.

---

## What It Does

- **Filters** irrelevant posts using a trained SVM classifier (sub-millisecond per post)
- **Groups** semantically similar reports about the same incident using MiniLM sentence embeddings and DBSCAN clustering
- **Extracts** location names from incident reports using BERT Named Entity Recognition
- **Resolves** those locations to coordinates via an offline, pre-built Chennai gazetteer
- **Summarises** the active situation every 15 minutes using BART abstractive summarisation
- **Displays** everything on a live dashboard — an incident feed, a map view, and a summary report

---

## How It Works

Data flows in one direction. Each stage does one job and passes its enriched output to the next.

```
Raw posts
    │
[Bouncer]        TF-IDF + LinearSVC        filters ~90% noise in < 1 ms/post
    │
[Deduplicator]   MiniLM + DBSCAN           groups duplicate reports into incidents
    │
[Detective]      dslim/bert-base-NER       extracts location names
    │
[Geocoder]       Chennai gazetteer (JSON)  maps location names to lat/lng
    │
[Editor]         facebook/bart-large-cnn   writes a situation summary (every 15 min)
    │
Dashboard        React + Leaflet, polling the API every 3 seconds
```

Each stage is a self-contained Python module; no stage reaches backwards.

---

## Results

Measured on the trained noise-filter (the Bouncer) and the geocoding layer:

| Metric                           | Value                |
| -------------------------------- | -------------------- |
| Disaster-class F1 (noise filter) | 0.91                 |
| ROC-AUC                          | 0.92                 |
| 5-fold cross-validation accuracy | ~86.5%               |
| Classifier throughput            | 75,000+ posts/second |
| Trained classifier size on disk  | 3.5 MB               |
| Chennai gazetteer coverage       | 171 locations        |

The classifier is trained on a combination of the Kaggle Disaster Tweets dataset and CrisisLexT26, which together provide both diverse, metaphorical examples and authentic crisis language from real events.

---

## Technology Stack

| Layer            | Technology                                                      | Purpose                               |
| ---------------- | --------------------------------------------------------------- | ------------------------------------- |
| Frontend         | React 18 + Vite                                                 | Dashboard UI                          |
| Map              | react-leaflet + Leaflet.js                                      | Incident map                          |
| Backend          | FastAPI + Uvicorn                                               | API server and pipeline orchestration |
| Noise filter     | scikit-learn TF-IDF + LinearSVC                                 | Sub-millisecond post classification   |
| Deduplication    | sentence-transformers MiniLM-L6-v2 + DBSCAN                     | Semantic incident clustering          |
| Location NER     | dslim/bert-base-NER (HuggingFace)                               | Named-entity extraction               |
| Geocoding        | Custom Chennai gazetteer (JSON)                                 | Offline coordinate resolution         |
| Summarisation    | facebook/bart-large-cnn (HuggingFace Inference API)             | Situation summary generation          |
| Containerisation | Docker                                                          | Reproducible backend image            |
| Hosting          | HuggingFace Spaces (backend) + Azure Static Web Apps (frontend) | Free cloud hosting                    |

---

## Architecture & Deployment

The backend is containerised with Docker and deployed to a free **HuggingFace Space**. The MiniLM and BERT models are baked into the image at build time, so there is no model download at runtime — the container starts cleanly even on a cold boot. The Space sleeps when idle and wakes automatically on the first incoming request.

The frontend is a static React build hosted on **Azure Static Web Apps**, served from a global CDN and deployed automatically from this repository. It polls the backend every three seconds and reflects the live state of the incident store. The two halves communicate over REST; CORS on the backend is restricted to the deployed frontend origin.

---

## Repository Structure

```
CrisisLens/
├── backend/
│   ├── api/
│   │   ├── main.py              # App entry point, CORS, model loading
│   │   └── routes/pipeline.py   # /api/clusters, /api/process, /api/report, /api/health
│   ├── components/
│   │   ├── bouncer.py           # TF-IDF + LinearSVC noise filter
│   │   ├── deduplicator.py      # MiniLM + DBSCAN clustering
│   │   ├── detective.py         # BERT NER location extraction
│   │   ├── geocoder.py          # Gazetteer lookup + fuzzy matching
│   │   └── editor.py            # BART situation summary (via HF API)
│   ├── scripts/                 # One-time setup: data prep, training, gazetteer, model pre-download
│   ├── models/                  # Trained .pkl classifier files
│   ├── data/gazetteers/         # Chennai gazetteer JSON
│   └── requirements.txt
├── frontend/                    # Vite + React dashboard
├── figures/                     # Evaluation plots
└── results/                     # Metrics and evaluation output
```

---

## Running Locally

**Backend**

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m scripts.download_models          # caches MiniLM + BERT once
export HF_API_TOKEN=your_hf_token          # for the BART summary step
uvicorn api.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev                                # serves on http://localhost:5173
```

The frontend reads its backend URL from `VITE_API_URL` (defaults to `http://localhost:8000`).

---
