"""Central configuration: paths and LLM settings loaded from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPOS_DIR = ROOT / "repos"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "llm_cache"

for d in (DATA_RAW, DATA_PROCESSED, REPOS_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip()
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
