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

delete_packed:
	rm app/data/03_model_ready.tar

init_venv:
	source /app/.venv/bin/activate

connect:
	ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) -L 8080:localhost:8080

send_data:
	type "D:\Nauka\Projekty\SongsRecommenderSystem\data\03_model_ready.tar" | ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) "cat > /app/data/03_model_ready.tar"

pack_raw_fragmented:
	tar -cf data/raw_and_fragmented.tar -C data 01_raw 02_fragmented

send_raw_fragmented:
	type "D:\Nauka\Projekty\SongsRecommenderSystem\data\raw_and_fragmented.tar" | ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) "cat > /app/data/raw_and_fragmented.tar"

unpack_raw_fragmented:
	tar -xf /app/data/raw_and_fragmented.tar -C /app/data

get_models:
	scp -r -i C:\Users\Wiktor\.ssh\vast_key -P $(PORT) root@$(IP):/app/models "D:\Nauka\Projekty\SongsRecommenderSystem"

send_model_checkpoint:
	type "D:\Nauka\Projekty\SongsRecommenderSystem\models\$(CKPT_FNAME)" | ssh -i C:\Users\Wiktor\.ssh\vast_key -p $(PORT) root@$(IP) "cat > /app/models/$(CKPT_FNAME)"
