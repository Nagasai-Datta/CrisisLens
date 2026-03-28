# =============================================================================
# FILE: backend/scripts/download_models.py
#
# PURPOSE:
#   Pre-downloads and caches all HuggingFace models so the server
#   doesn't need internet access at runtime.
#
# WHY THIS EXISTS (Bug 16 fix):
#   MiniLM (~90MB) and BERT NER (~400MB) download on first import.
#   If there's no internet at demo time (or if download is slow),
#   the server crashes or hangs on startup.
#
#   This script is called DURING the Docker build, so models are
#   baked into the image. First startup is instant.
#
# WHEN TO RUN:
#   - Automatically during Docker build (called in Dockerfile)
#   - Manually during local setup: python -m backend.scripts.download_models
#
# MODELS DOWNLOADED:
#   1. sentence-transformers/all-MiniLM-L6-v2  (~90MB)  → Deduplicator
#   2. dslim/bert-base-NER                     (~400MB) → Detective
#
# NOTE: BART (facebook/bart-large-cnn) is NOT downloaded here because
#   we use it via the HuggingFace Inference API — it never runs locally.
# =============================================================================

import os
import sys


def download_all_models():
    """
    Downloads and caches both transformer models.
    
    The models are saved to wherever HF_HOME points:
    - During Docker build: /home/user/.cache/huggingface
    - At runtime on HF Spaces: /data/.huggingface (persistent storage)
    
    We download during build so the image contains the models.
    At runtime, HF_HOME=/data/.huggingface may also cache them
    for even faster restarts on HF Spaces.
    """
    
    print("=" * 50)
    print("CrisisLens — Pre-downloading ML Models")
    print("=" * 50)
    
    cache_dir = os.environ.get("HF_HOME", None)
    if cache_dir:
        print(f"   Cache directory: {cache_dir}")
    else:
        print("   Cache directory: default (~/.cache/huggingface)")
    
    # ── 1. MiniLM (Deduplicator) ─────────────────────────────────────
    print("\n[1/2] Downloading sentence-transformers/all-MiniLM-L6-v2...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Quick sanity check — encode a test sentence
        test_embedding = model.encode(["test sentence"])
        assert test_embedding.shape[1] == 384, "MiniLM should produce 384-dim vectors" # type: ignore
        
        print("   ✅ MiniLM downloaded and verified (384-dim embeddings)")
        del model  # free memory
    except Exception as e:
        print(f"   ❌ MiniLM download failed: {e}")
        sys.exit(1)
    
    # ── 2. BERT NER (Detective) ──────────────────────────────────────
    print("\n[2/2] Downloading dslim/bert-base-NER...")
    try:
        from transformers import pipeline
        ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1  # CPU — we're just downloading, not running inference
        )
        
        # Quick sanity check — run NER on a test sentence
        test_result = ner("Chennai is in Tamil Nadu")
        print(f"   ✅ BERT NER downloaded and verified ({len(test_result)} entities found in test)") # type: ignore
        del ner  # free memory
    except Exception as e:
        print(f"   ❌ BERT NER download failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All models downloaded successfully!")
    print("=" * 50)


if __name__ == "__main__":
    download_all_models()