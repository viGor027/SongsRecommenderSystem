prepare_dirs:
	uv run python workflow_actions/prepare_dirs.py

train:
	uv run python workflow_actions/train/run.py

train_detached:
	tmux new-session -d -s train "uv run python workflow_actions/train/run.py"

pack_data:
	tar -cf data/03_model_ready.tar -C data 03_model_ready

unpack_data:
	tar -xf /app/data/03_model_ready.tar -C /app/data

init_venv:
	source /app/.venv/bin/activate

connect:
	ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) -L 8080:localhost:8080

send_data:
	type "D:\Nauka\Projekty\SongsRecommenderSystem\data\03_model_ready.tar" | ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) "cat > /app/data/03_model_ready.tar"
