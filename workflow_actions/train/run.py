if __name__ == "__main__":
    from workflow_actions.train.train import Train
    from workflow_actions.paths import TRAIN_CONFIG_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from multiprocessing import freeze_support

    freeze_support()
    training_cfg = read_json_to_dict(TRAIN_CONFIG_PATH)
    schedules = {
        "IMAGENET1K_V1": {
            "optimizer": "Adam",
            "optimizer_params": {"lr": 5e-4},
            "lr_schedule": "warmup_cosine",
            "lr_schedule_params": {
                "start_factor": 0.1,
                "warmup_iters": 10,
                "T_max": 90,
                "eta_min": 1e-6,
            },
        },
        "IMAGENET1K_V2": {
            "optimizer": "Adam",
            "optimizer_params": {"lr": 5e-4},
            "lr_schedule": "warmup_cosine",
            "lr_schedule_params": {
                "start_factor": 0.1,
                "warmup_iters": 10,
                "T_max": 90,
                "eta_min": 1e-6,
            },
        },
        None: {
            "optimizer": "Adam",
            "optimizer_params": {"lr": 1e-3},
            "lr_schedule": None,
            "lr_schedule_params": None,
        },
    }

    training_cfg["run_single_training_config"]["architecture"]["freeze_backbone"] = (
        False
    )
    for weights in ["IMAGENET1K_V1", "IMAGENET1K_V2", None]:
        training_cfg["run_single_training_config"]["optimization"] = schedules[weights]
        training_cfg["run_single_training_config"]["architecture"]["weights"] = weights

        for backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            if backbone not in ["resnet50", "resnet101"] and weights == "IMAGENET1K_V2":
                continue

            training_cfg["run_single_training_config"]["architecture"][
                "backbone_name"
            ] = backbone
            training_cfg["run_single_training_config"]["run_name"] = (
                f"{backbone}_{weights}"
            )

            train = Train(**training_cfg)
            train.run_single_training()
