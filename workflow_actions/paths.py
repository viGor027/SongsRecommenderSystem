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

"""CONFIG PATHS"""
WORKFLOWS_ROOT = PROJECT_ROOT / "workflow_actions"
DATASET_PREPROCESSOR_CONFIG_PATH = (
    WORKFLOWS_ROOT / "dataset_preprocessor" / "dataset_preprocessor_config.json"
)
