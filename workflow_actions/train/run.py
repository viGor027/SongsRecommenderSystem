if __name__ == "__main__":
    from workflow_actions.train.train import Train
    from workflow_actions.paths import TRAIN_CONFIG_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from multiprocessing import freeze_support

    freeze_support()
    training_cfg = read_json_to_dict(TRAIN_CONFIG_PATH)
    train = Train(**training_cfg)
    train.run_single_training()
