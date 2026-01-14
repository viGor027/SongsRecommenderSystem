from workflow_actions.evaluate_model.source import AccuraccyTest, RandomizedABXTest
from workflow_actions.paths import (
    TRAINED_MODELS_CONFIG_PATHS,
    GLOBAL_TRAIN_INDEX_PATH,
    DATA_DIR,
)
from workflow_actions.train.source import ModelInitializer
from workflow_actions.json_handlers import read_json_to_dict, write_dict_to_json
import torch


class EvaluateModel:
    def __init__(self):
        """IMPORTANT:
        - Prepare c) type index before evaluating.
        - TRAINED_MODELS_CONFIG_PATHS is source of truth for which models will be evaluated
        """
        self.tests = [AccuraccyTest, RandomizedABXTest]
        self.configs = read_json_to_dict(TRAINED_MODELS_CONFIG_PATHS)

        self.mi = ModelInitializer()
        self.index = read_json_to_dict(GLOBAL_TRAIN_INDEX_PATH)

    @torch.no_grad()
    def evaluate_all(self, k_triplets: int):
        summary = {}
        kwargs = {"randomized_abx_test_k_triplets": k_triplets}
        for ckpt_fname, cfg in self.configs.items():
            model_summary = []

            assembly = self.mi.get_model_assembly(cfg)
            model = self.mi.load_assembly_weights(
                model_assembly=assembly, model_ckpt_filename=ckpt_fname
            )
            for test_cls in self.tests:
                test = test_cls(model, ckpt_fname, self.index)
                test_result = test(**kwargs)
                description = repr(test)
                model_summary.append([description, test_result])
            summary[ckpt_fname] = model_summary
        write_dict_to_json(
            data=summary, file_path=DATA_DIR / "all_models_evaluation_summary.json"
        )
