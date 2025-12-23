from workflow_actions.dataset_preprocessor.dataset_preprocessor import (
    DatasetPreprocessor,
)
from workflow_actions.paths import DATASET_PREPROCESSOR_CONFIG_PATH
from workflow_actions.json_handlers import read_json_to_dict


def make_dp():
    prepare_dataset_cfg = read_json_to_dict(DATASET_PREPROCESSOR_CONFIG_PATH)
    dp = DatasetPreprocessor(**prepare_dataset_cfg)
    return dp


if __name__ == "__main__":
    dp = make_dp()
    dp.prepare_model_ready_data(for_sanity_check=True)
    # dp.sample_packer.pack()
