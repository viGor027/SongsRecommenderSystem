from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SongsRecommenderSystem/

# data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_READY_DIR = DATA_DIR / "model_ready"

DOWNLOAD_DIR = RAW_DATA_DIR / "downloaded_songs"
LABELS_DIR = RAW_DATA_DIR / "labels"
LABELS_PATH = LABELS_DIR / "labels.json"
