# =============================================================================
# DOCKERFILE — CrisisLens Backend (HuggingFace Spaces)
#
# PURPOSE:
#   Builds the FastAPI backend container for deployment on HuggingFace Spaces.
#   All ML models (MiniLM, BERT NER) are pre-downloaded during build so the
#   server starts instantly without needing internet at runtime.
#
# HUGGINGFACE SPACES REQUIREMENTS:
#   - Must use port 7860 (HF's default)
#   - Must create user with UID 1000 (HF's security requirement)
#   - .dockerignore must exclude frontend/ and other bloat
#
# BUILD STRATEGY — WHY THIS ORDER MATTERS:
#   Docker caches each layer. We order from "least likely to change"
#   to "most likely to change" so rebuilds are fast:
#
#   Layer 1: System packages        → almost never changes
#   Layer 2: Python dependencies    → changes when requirements.txt changes
#   Layer 3: Model downloads        → changes only if we switch models
#   Layer 4: Application code       → changes on every code push
#
#   If you only change code (Layer 4), Docker reuses Layers 1-3 from cache.
#   A full rebuild from scratch takes ~10-15 minutes.
#   A code-only rebuild takes ~30 seconds.
#
# PYTORCH CPU-ONLY FIX:
#   The default `pip install torch` downloads the CUDA-enabled version (~2.3GB).
#   HuggingFace Spaces free tier is CPU-only — no GPU available.
#   We install the CPU-only version (~200MB) from PyTorch's CPU index.
#   This saves ~2GB of download time and disk space.
#
# TO DEPLOY:
#   1. Create a HuggingFace Space with Docker SDK
#   2. Push this repo to the Space's git repo
#   3. HF auto-builds and deploys from this Dockerfile
# =============================================================================

FROM python:3.11-slim

# ── System setup ──────────────────────────────────────────────────────────
# Create non-root user (HuggingFace Spaces requirement: UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# ── Layer 2: Python dependencies ──────────────────────────────────────────
# Copy requirements first (Docker cache: only re-runs if requirements change)
COPY --chown=user backend/requirements.txt requirements.txt

# Install PyTorch CPU-only FIRST, then the rest of requirements.
# WHY SEPARATE:
#   requirements.txt has torch==2.3.0 which pulls the CUDA version (~2.3GB).
#   By installing CPU-only torch first from PyTorch's CPU index,
#   the subsequent pip install sees torch is already satisfied and skips it.
#   This saves ~2GB and prevents build timeouts on HuggingFace.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Layer 3: Pre-download ML models ──────────────────────────────────────
# Downloads MiniLM (~90MB) and BERT NER (~400MB) into the image.
# Without this, first startup would stall for 5-10 minutes downloading models.
# Models are cached in /home/user/.cache/huggingface inside the image.
COPY --chown=user backend/scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py && rm /tmp/download_models.py

# ── Layer 4: Application code ─────────────────────────────────────────────
# This layer changes most often — any code edit triggers only this layer.
# .dockerignore excludes frontend/, .git/, venv/, notebooks/, raw data.
COPY --chown=user . /home/user/app

# ── Runtime configuration ─────────────────────────────────────────────────
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Tell HuggingFace libraries where to find cached models.
    # During build, models were downloaded to /home/user/.cache/huggingface.
    # At runtime, this ensures transformers/sentence-transformers find them.
    HF_HOME=/home/user/.cache/huggingface \
    # Disable tokenizer parallelism warning (noisy in logs, not useful)
    TOKENIZERS_PARALLELISM=false

EXPOSE 7860

# ── Start the server ──────────────────────────────────────────────────────
# --host 0.0.0.0: listen on all interfaces (required for Docker)
# --port 7860: HuggingFace Spaces default port
# --workers 1: single worker — our models are loaded into process memory
#              and shared state (active_clusters) wouldn't sync across workers
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]