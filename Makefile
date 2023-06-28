push-ml-node:
	rsync -rav --progress \
	--exclude='.venv/' \
	--exclude='.git/' \
	--exclude='checkpoints/' \
	--exclude='__pycache__/' \
	--include='*/' \
	--include='*.py' \
	--include='*.csv' \
	--include="*.txt" \
	--exclude='*' \
	. ml-node:/home/ml-node/Documents/uav-localization-2023
