from workflow_actions.paths import _DIRS_TO_INIT


for directory in _DIRS_TO_INIT:
    directory.mkdir(parents=True, exist_ok=True)
