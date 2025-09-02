from workflow_actions.prepare_dataset.source.prepare_dataset import PrepareDataset
from workflow_actions.paths import PREPARE_DATASET_CONFIG_PATH
from workflow_actions.json_handlers import read_json_to_dict

prepare_dataset_cfg = read_json_to_dict(PREPARE_DATASET_CONFIG_PATH)
pd = PrepareDataset(**prepare_dataset_cfg)
# Put code doing workflow job here
from pathlib import Path
pd._empty_folder(
    Path(
        "D:\\Nauka\\Projekty\\SongsRecommenderSystem\\test"
    )
)
