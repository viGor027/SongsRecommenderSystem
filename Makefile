prepare_dirs:
	uv run python workflow_actions/prepare_dirs.py

train:
	uv run python workflow_actions/train/run.py