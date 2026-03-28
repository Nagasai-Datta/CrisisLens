# CrisisLens

**Real-time crisis intelligence for natural disaster response.**

CrisisLens is an end-to-end decision support system that ingests social media text during natural disasters, filters noise, deduplicates reports, extracts and geocodes locations, and surfaces actionable incident intelligence on a live tactical dashboard for emergency commanders.

![CrisisLens War Room](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![React](https://img.shields.io/badge/React-18-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What It Does

During a flood or disaster, thousands of tweets are posted per minute. Most are noise. A small fraction contain actionable rescue requests. CrisisLens automatically:

- **Filters** irrelevant tweets using a trained SVM classifier (under 1ms per tweet)
- **Groups** semantically similar reports about the same incident using MiniLM embeddings and DBSCAN clustering
- **Extracts** location names from incident reports using BERT Named Entity Recognition
- **Geocodes** locations to GPS coordinates via a pre-built Chennai gazetteer
- **Generates** an executive situation report every 5 minutes using BART summarisation
- **Displays** everything on a live War Room dashboard — tactical map, incident feed, and commander's report

---

## Live Demo

| Component          | URL                                                         |
| ------------------ | ----------------------------------------------------------- |
| War Room Dashboard | [crisislens.vercel.app](https://crisislens.vercel.app)      |
| Backend API        | [huggingface.co/spaces/...](https://huggingface.co/spaces/) |
| API Documentation  | `<backend-url>/docs`                                        |

---

## Pipeline Architecture

```
Raw Tweets
    ↓
[Bouncer]        TF-IDF + LinearSVC      — filters ~90% noise in < 1ms/tweet
    ↓
[Deduplicator]   MiniLM + DBSCAN         — groups duplicate reports into clusters
    ↓
[Detective]      dslim/bert-base-NER     — extracts location names from tweets
    ↓
[Geocoder]       Chennai Gazetteer JSON  — maps location names to lat/lng
    ↓
[Editor]         facebook/bart-large-cnn — generates situation report (every 5 min)
    ↓
War Room Dashboard (React + Leaflet + 3-second polling)
```

Each stage is a self-contained Python module. Data flows in one direction only. No stage reaches backwards.

---

## Technology Stack

| Layer         | Technology                                       | Purpose                               |
| ------------- | ------------------------------------------------ | ------------------------------------- |
| Frontend      | React 18 + Vite                                  | War Room dashboard                    |
| Maps          | react-leaflet + Leaflet.js                       | Tactical incident map                 |
| Backend       | FastAPI + Uvicorn                                | API server and pipeline orchestration |
| Noise Filter  | scikit-learn TF-IDF + LinearSVC                  | Sub-millisecond tweet classification  |
| Deduplication | sentence-transformers MiniLM-L6-v2 + DBSCAN      | Semantic incident clustering          |
| Location NER  | dslim/bert-base-NER (HuggingFace)                | Named entity extraction               |
| Geocoding     | Custom Chennai Gazetteer JSON                    | Offline coordinate resolution         |
| Summarisation | facebook/bart-large-cnn (HuggingFace API)        | Situation report generation           |
| Deployment    | HuggingFace Spaces (backend) + Vercel (frontend) | Free cloud hosting                    |

---

## Repository Structure

```
crisislens/
├── backend/                    # FastAPI server + ML pipeline
│   ├── api/
│   │   ├── main.py             # App entry point, CORS, lifespan
│   │   └── routes/
│   │       └── pipeline.py     # REST endpoints
│   ├── components/             # Pipeline components (one file = one job)
│   │   ├── bouncer.py          # Runtime noise filter
│   │   ├── deduplicator.py     # Semantic clustering
│   │   ├── detective.py        # BERT NER location extraction
│   │   ├── geocoder.py         # Gazetteer coordinate lookup
│   │   └── editor.py           # BART situation report generation
│   ├── scripts/                # One-time setup scripts (run offline)
│   │   ├── prepare_data.py     # Merge Kaggle + CrisisLexT26 datasets
│   │   ├── train_bouncer.py    # Train and save TF-IDF + SVM
│   │   ├── populate_gazetteer.py # Build Chennai coordinate JSON
│   │   └── download_models.py  # Pre-cache HuggingFace models
│   ├── data/
│   │   └── gazetteers/
│   │       └── chennai_gazetteer.json  # 171 Chennai locations with coordinates
│   ├── requirements.txt
│   └── Dockerfile              # HuggingFace Spaces deployment
├── frontend/                   # React War Room dashboard
│   ├── src/
│   │   ├── App.jsx             # Root component, shared state, polling
│   │   ├── components/
│   │   │   ├── ControlSidebar  # Model health + file upload
│   │   │   ├── ClusterFeed     # Incident card list (Zone A)
│   │   │   ├── ClusterCard     # Individual incident card
│   │   │   ├── TacticalMap     # Leaflet map with incident dots (Zone B)
│   │   │   └── CommandersReport # BART summary display (Zone C)
│   └── package.json
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11
- Node.js 18+
- ~2GB disk space for ML models

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/crisislens.git
cd crisislens
```

### 2. Set up the backend

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 3. Download ML models (run once)

```bash
python backend/scripts/download_models.py
```

### 4. Train the Bouncer (run once)

You need the training datasets first:

- **Kaggle Disaster Tweets**: Download `train.csv` from [kaggle.com/competitions/nlp-getting-started](https://kaggle.com/competitions/nlp-getting-started) → place in `backend/data/raw/kaggle/`
- **CrisisLexT26**: Download from [crisislex.org](https://crisislex.org/data-collections.html) → place 26 event folders in `backend/data/raw/crisislex/`

```bash
# Merge and clean both datasets
python backend/scripts/prepare_data.py

# Train TF-IDF + SVM classifier
python backend/scripts/train_bouncer.py
```

### 5. Build the Chennai Gazetteer (run once, ~10 minutes)

```bash
python backend/scripts/populate_gazetteer.py
```

### 6. Configure environment

Create `backend/.env`:

```
HF_API_TOKEN=hf_your_token_here
```

Get your free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 7. Start the backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to verify all models loaded.

### 8. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to open the War Room.

---

## Usage

1. Open the War Room at `http://localhost:5173`
2. Verify all four pipeline components show green in the sidebar
3. Upload a `.txt` or `.csv` file of tweets (one per line) using the file picker
4. Click **Run Pipeline**
5. Incident clusters appear on the map and in the feed within seconds
6. The Commander's Report auto-generates every 5 minutes
7. Click **✓ Resolve** on any card to dismiss a handled incident

### Sample Tweet File

A demo CSV is provided at `backend/data/demo/chennai_flood_tweets.csv` with 30 tweets across 10 simulated Chennai flood incidents including realistic noise tweets.

---

## API Reference

| Method   | Endpoint             | Description                        |
| -------- | -------------------- | ---------------------------------- |
| `POST`   | `/api/process`       | Submit tweets, run full pipeline   |
| `GET`    | `/api/clusters`      | Get current active cluster list    |
| `DELETE` | `/api/clusters/{id}` | Resolve (dismiss) a single cluster |
| `DELETE` | `/api/clusters`      | Clear all clusters                 |
| `GET`    | `/api/report`        | Get latest BART situation report   |
| `GET`    | `/api/health`        | Per-model health status            |
| `GET`    | `/docs`              | Interactive Swagger UI             |

---

## Deployment

CrisisLens deploys entirely for free:

- **Backend** → HuggingFace Spaces (Docker, 16GB RAM — required for BERT + MiniLM)
- **Frontend** → Vercel (static build, global CDN)

See [`backend/README.md`](backend/README.md) for the full deployment walkthrough.

---

## Training Data

The Bouncer classifier is trained on a combination of two public datasets:

| Dataset                    | Size                            | Source                                                                                             |
| -------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------- |
| Kaggle NLP Disaster Tweets | ~7,600 tweets                   | [kaggle.com/competitions/nlp-getting-started](https://kaggle.com/competitions/nlp-getting-started) |
| CrisisLexT26               | ~27,000 tweets across 26 events | [crisislex.org](https://crisislex.org)                                                             |

Training data is not included in this repository. Both datasets are publicly available and free to download.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) — BERT NER model
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Sentence embeddings
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) — Summarisation
- [CrisisLex](https://crisislex.org) — Crisis tweet dataset
- [Leaflet.js](https://leafletjs.com) + [OpenStreetMap](https://www.openstreetmap.org) — Mapping
