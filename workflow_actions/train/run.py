if __name__ == "__main__":
    from workflow_actions.train.train import Train
    from workflow_actions.paths import TRAIN_CONFIG_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from multiprocessing import freeze_support

    freeze_support()

    for aggregator_type, trainable_aggregator in [
        ("lstm", False),
        ("lstm", True),
        ("average", False),
    ]:
        training_cfg = read_json_to_dict(TRAIN_CONFIG_PATH)
        training_cfg["run_single_training_config"]["architecture"][
            "aggregator_type"
        ] = aggregator_type
        training_cfg["run_single_training_config"]["architecture"][
            "trainable_aggregator"
        ] = trainable_aggregator
        training_cfg["run_single_training_config"]["run_name"] = (
            "Aggregator_" + aggregator_type + str(trainable_aggregator)
        )
        train = Train(**training_cfg)
        train.run_single_training()
