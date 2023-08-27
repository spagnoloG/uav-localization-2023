push-ml-node:
	@echo "Pushing code to ml-node"
	rsync -rav --progress \
		--exclude='.venv/' \
		--exclude='.git/' \
		--exclude='vis/*' \
		--exclude='checkpoints/' \
		--exclude='__pycache__/' \
		--exclude='castral_dataset' \
		--include='utils/**' \
		--include='conf/**' \
		--include='*/' \
		--include='*.py' \
		--include='*.csv' \
		--include="*.txt" \
		--exclude='*' \
		. ml-node:/home/ml-node/Documents/uav-localization-2023

push-vicos:
	@echo "Pushing code to vicos"
	find . \
    	-path './.venv' -prune -o \
    	-path './.git' -prune -o \
    	-path './checkpoints' -prune -o \
    	-path './__pycache__' -prune -o \
    	-type f \( -name '*.py' -o -name '*.csv' -o -name '*.txt' -o -name '*.yaml' \) -print0 | \
    tar cf - --null -T - | \
    	ssh vicos 'dir="/home/gasper/uav-localization-2023_`date +%Y%m%d_%H%M%S`"; mkdir -p "$$dir" && cd "$$dir" && tar xf -'

init_venv:
	@echo "Initializing virtual environment"
	python3 -m venv .venv

install-requirements:
	@echo "Installing requirements"
	pip3 install -r requirements.txt

lint:
	@echo "Linting python files"
	black *.py
	black utils/*.py
	black cnn_backbone/*.py
	black sat/*.py
	@echo "Linting configuration files"
	prettier --write conf/*.yaml # npm i -g prettier

add_results:
	@echo "Adding results to git"
	@for dir in checkpoints/*; do [ -d "$$dir" ] && (mkdir -p results/$$(basename $$dir) && cp "$$dir/train.log" "$$dir/config.json" results/$$(basename $$dir)/); done

val:
	@echo "Running validation"
	python3 val.py

train:
	@echo "Running training"
	python3 train.py

train-ml-node: push-ml-node
	@echo "Training on ml-node"
	ssh ml-node "tmux new-session -d -s training && \
		tmux send-keys -t training 'cd /home/ml-node/Documents/uav-localization-2023 && \
		. .venv/bin/activate && python3 train.py --config configuration-ml-node' C-m && \
		tmux split-window -h -t training && \
		tmux send-keys -t training 'nvtop' C-m"

download-vis:
	@echo  "Downloading plots from ml-node"
	mkdir -p vis
	rsync -rav ml-node:/home/ml-node/Documents/uav-localization-2023/vis/ ./vis/

download-weights:
	@echo  "Downloading weights from ml-node"
	mkdir -p checkpoints
	rsync -rav ml-node:/home/ml-node/Documents/uav-localization-2023/checkpoints/ ./checkpoints/

search-heatmap:
	@test -n "$(HEATMAP_VALUE)" || (echo "HEATMAP_VALUE is not set"; exit 1)
	@find ./results -type f -iname "config.json" -printf '%T@ %p\n' | sort -n | cut -d' ' -f2- | while read -r file; do \
		value=$$(jq ".dataset | .heatmap_kernel_size" "$$file"); \
		if [ "$$value" == "$(HEATMAP_VALUE)" ]; then \
			echo "$$file: $$value"; \
		fi \
	done

move3dplots:
	@CSV_FILE="./utils/res/hann_kers.csv"; \
	TARGET_DIR="./utils/res/heatmaps3d"; \
	\
	if [ ! -f "$$CSV_FILE" ]; then \
		echo "$$CSV_FILE not found!"; \
		exit 1; \
	fi; \
	\
	mkdir -p "$$TARGET_DIR"; \
	\
	tail -n +2 "$$CSV_FILE" | while IFS=", " read -r size hash; do \
		SOURCE_FILE="./vis/$$hash/validation_3d_hm_$$hash-3-1.png"; \
		if [ -f "$$SOURCE_FILE" ]; then \
			cp "$$SOURCE_FILE" "$$TARGET_DIR/"; \
			echo "Copied $$SOURCE_FILE to $$TARGET_DIR/"; \
		else \
			echo "File $$SOURCE_FILE not found!"; \
		fi; \
	done
