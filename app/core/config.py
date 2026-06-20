# Configure CPU threading limits to minimize memory overhead in production
import os
from pathlib import Path
from dotenv import load_dotenv
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")


LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 500
LLM_RETRIES = 3


DB_DIR = PROJECT_ROOT / "database"
DB_PATH = DB_DIR / "beaver_dw.db"
