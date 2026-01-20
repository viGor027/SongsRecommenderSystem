from workflow_actions.evaluate_model.source import (
    AccuraccyTest,
    RandomizedABXTest,
    AccuraccyTestFullSongs,
)
from workflow_actions.paths import (
    TRAINED_MODELS_CONFIG_PATHS,
    GLOBAL_TRAIN_INDEX_PATH,
    DATA_DIR,
)
from workflow_actions.train.source import ModelInitializer
from workflow_actions.json_handlers import read_json_to_dict, write_dict_to_json
from workflow_actions.paths import MODEL_READY_TRAIN_DIR
import torch
import tqdm


class EvaluateModel:
    def __init__(
        self,
        model_ckpt_filenames: list[str],
        tests_to_perform: list[str],
    ):
        """
        IMPORTANT:
        - Prepare c) type index before evaluating.
        - if some item from `model_ckpt_filenames` is not present in TRAINED_MODELS_CONFIG_PATHS as a key,
          then procedure will fail.
        """
        str2test = {
            "AccuraccyTest": AccuraccyTest,
            "RandomizedABXTest": RandomizedABXTest,
            "AccuraccyTestFullSongs": AccuraccyTestFullSongs,
        }

        self.tests = [str2test[test_str] for test_str in tests_to_perform]
        self.configs = read_json_to_dict(TRAINED_MODELS_CONFIG_PATHS)

        self.mi = ModelInitializer()
        self.index = read_json_to_dict(GLOBAL_TRAIN_INDEX_PATH)

        self.model_ckpt_filenames = model_ckpt_filenames

        paths = sorted(
            MODEL_READY_TRAIN_DIR.glob("X_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        self.Xs = [torch.load(x_path) for x_path in tqdm.tqdm(paths)]
        print("EvaluateModel.__init__: Wczytano Xs")

    @torch.inference_mode()
    def evaluate(self, k_triplets: int):
        summary = {}
        kwargs = {"randomized_abx_test_k_triplets": k_triplets}
        for ckpt_fname in self.model_ckpt_filenames:
            if ckpt_fname not in self.configs:
                raise ValueError(
                    f"TRAINED_MODELS_CONFIG_PATHS doesn't contain {ckpt_fname}"
                )
            cfg = self.configs[ckpt_fname]
            print(
                f"EvaluateModel.evaluate: currently evaluated model is {ckpt_fname.split('-epoch')[0]}"
            )

            model_summary = []

            assembly = self.mi.get_model_assembly(cfg)
            model = self.mi.load_assembly_weights(
                model_assembly=assembly, model_ckpt_filename=ckpt_fname
            )
            for test_cls in self.tests:
                test = test_cls(model, ckpt_fname, self.index, Xs=self.Xs)
                test_result = test(**kwargs)
                description = repr(test)
                model_summary.append([description, test_result])
            summary[ckpt_fname] = model_summary
            write_dict_to_json(
                data=summary, file_path=DATA_DIR / "all_models_evaluation_summary.json"
            )
