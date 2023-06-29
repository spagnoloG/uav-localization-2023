push-ml-node:
	rsync -rav --progress \
		--exclude='.venv/' \
		--exclude='.git/' \
		--exclude='checkpoints/' \
		--exclude='__pycache__/' \
		--include='utils/**' \
		--include='conf/**' \
		--include='*/' \
		--include='*.py' \
		--include='*.csv' \
		--include="*.txt" \
		--exclude='*' \
		. ml-node:/home/ml-node/Documents/uav-localization-2023

val:
	python3 val.py

train:
	python3 train.py

train-ml-node: push-ml-node
	ssh ml-node "tmux new-session -d -s training && \
		tmux send-keys -t training 'cd /home/ml-node/Documents/uav-localization-2023 && \
		. .venv/bin/activate && python3 train.py --config configuration-ml-node' C-m && \
		tmux split-window -h -t training && \
		tmux send-keys -t training 'nvtop' C-m"
