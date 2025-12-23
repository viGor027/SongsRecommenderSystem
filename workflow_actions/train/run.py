if __name__ == "__main__":
    from workflow_actions.train.train import Train
    from workflow_actions.paths import TRAIN_CONFIG_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from multiprocessing import freeze_support

    BASE_RUN_NAME = "_IMAGENET1K_V1_no_aug"

    freeze_support()
    training_cfg = read_json_to_dict(TRAIN_CONFIG_PATH)
    for backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
        training_cfg["run_single_training_config"]["architecture"]["backbone_name"] = (
            backbone
        )
        training_cfg["run_single_training_config"]["run_name"] = (
            backbone + BASE_RUN_NAME
        )
        train = Train(**training_cfg)
        train.run_single_training()
