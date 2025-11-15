from workflow_actions.dataset_preprocessor.dataset_preprocessor import (
    DatasetPreprocessor,
)
from workflow_actions.paths import DATASET_PREPROCESSOR_CONFIG_PATH
from workflow_actions.json_handlers import read_json_to_dict

prepare_dataset_cfg = read_json_to_dict(DATASET_PREPROCESSOR_CONFIG_PATH)
dp = DatasetPreprocessor(**prepare_dataset_cfg)
# Put code doing workflow job here
dp.run_pipeline()
