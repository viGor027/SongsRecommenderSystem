from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SongsRecommenderSystem/

"""DATA PATHS"""
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
FRAGMENTED_DATA_DIR = DATA_DIR / "02_fragmented"
MODEL_READY_DATA_DIR = DATA_DIR / "03_model_ready"

MODEL_READY_TRAIN_DIR = MODEL_READY_DATA_DIR / "train"
MODEL_READY_VALID_DIR = MODEL_READY_DATA_DIR / "valid"

DOWNLOAD_DIR = RAW_DATA_DIR / "downloaded_songs"
LABELS_DIR = RAW_DATA_DIR / "labels"

LABELS_PATH = LABELS_DIR / "labels.json"
LABEL_MAPPING_PATH = LABELS_DIR / "label_mapping.json"
SCRAPE_STAMP_PATH = RAW_DATA_DIR / "scrape_stamp.json"
FRAGMENTATION_STAMP_PATH = FRAGMENTED_DATA_DIR / "fragmentation_stamp.json"
FRAGMENTATION_INDEX_PATH = FRAGMENTED_DATA_DIR / "fragmentation_index.json"

GLOBAL_TRAIN_INDEX_PATH = FRAGMENTED_DATA_DIR / "global_train_index.json"
GLOBAL_VALID_INDEX_PATH = FRAGMENTED_DATA_DIR / "global_valid_index.json"

PIPELINE_RUN_RECORD_PATH = MODEL_READY_DATA_DIR / "pipeline_run_record.json"

"""CONFIG PATHS"""
WORKFLOWS_ROOT = PROJECT_ROOT / "workflow_actions"
DATASET_PREPROCESSOR_CONFIG_PATH = (
    WORKFLOWS_ROOT / "dataset_preprocessor" / "dataset_preprocessor_config.json"
)
TRAIN_CONFIG_PATH = WORKFLOWS_ROOT / "train" / "train_config.json"
DOWNLOAD_SONGS_CONFIG_PATH = (
    WORKFLOWS_ROOT / "download_songs" / "download_songs_config.json"
)

EVALUATE_MODEL_CONFIG_PATH = (
    WORKFLOWS_ROOT / "evaluate_model" / "evaluate_model_confing.json"
)

TRAINED_MODELS_DIR = PROJECT_ROOT / "models"

TRAINED_MODELS_CONFIG_PATHS = (
    DATA_DIR / "trained_model_configs" / "trained_model_configs.json"
)

_DIRS_TO_INIT = [
    DATA_DIR,
    RAW_DATA_DIR,
    FRAGMENTED_DATA_DIR,
    MODEL_READY_DATA_DIR,
    MODEL_READY_TRAIN_DIR,
    MODEL_READY_VALID_DIR,
    DOWNLOAD_DIR,
    LABELS_DIR,
    TRAINED_MODELS_DIR,
]
