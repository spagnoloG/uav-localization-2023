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

push-vicos:
	find . \
    -path './.venv' -prune -o \
    -path './.git' -prune -o \
    -path './checkpoints' -prune -o \
    -path './__pycache__' -prune -o \
    -type f \( -name '*.py' -o -name '*.csv' -o -name '*.txt' -o -name '*.yaml' \) -print0 | \
    tar cf - --null -T - | \
    ssh vicos 'dir="/home/gasper/uav-localization-2023_`date +%Y%m%d_%H%M%S`"; mkdir -p "$$dir" && cd "$$dir" && tar xf -'

init_venv:
	python3 -m venv .venv

install-requirements:
	pip3 install -r requirements.txt

lint:
	black *.py
	black utils/*.py
	black cnn_backbone/*.py
	black sat/*.py

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
