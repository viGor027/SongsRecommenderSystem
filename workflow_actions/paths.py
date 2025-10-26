from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SongsRecommenderSystem/

"""DATA PATHS"""
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
FRAGMENTED_DATA_DIR = DATA_DIR / "02_fragmented"
MODEL_READY_DATA_DIR = DATA_DIR / "03_model_ready"

DOWNLOAD_DIR = RAW_DATA_DIR / "downloaded_songs"
LABELS_DIR = RAW_DATA_DIR / "labels"

LABELS_PATH = LABELS_DIR / "labels.json"
LABEL_MAPPING_PATH = LABELS_DIR / "label_mapping.json"
SCRAPE_STAMP_PATH = RAW_DATA_DIR / "scrape_stamp.json"
FRAGMENTATION_STAMP_PATH = FRAGMENTED_DATA_DIR / "fragmentation_stamp.json"

"""CONFIG PATHS"""
WORKFLOWS_ROOT = PROJECT_ROOT / "workflow_actions"
DATASET_PREPROCESSOR_CONFIG_PATH = (
    WORKFLOWS_ROOT / "dataset_preprocessor" / "dataset_preprocessor_config.json"
)
TRAIN_CONFIG_PATH = WORKFLOWS_ROOT / "train" / "train_config.json"
DOWNLOAD_SONGS_CONFIG_PATH = (
    WORKFLOWS_ROOT / "download_songs" / "download_songs_config.json"
)

TRAINED_MODELS_DIR = PROJECT_ROOT / "models"

_DIRS_TO_INIT = [
    DATA_DIR,
    RAW_DATA_DIR,
    FRAGMENTED_DATA_DIR,
    FRAGMENTED_DATA_DIR / "train",
    FRAGMENTED_DATA_DIR / "valid",
    MODEL_READY_DATA_DIR,
    MODEL_READY_DATA_DIR / "train",
    MODEL_READY_DATA_DIR / "valid",
    DOWNLOAD_DIR,
    LABELS_DIR,
    TRAINED_MODELS_DIR,
]
