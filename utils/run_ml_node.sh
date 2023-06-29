#!/bin/bash
tmux new-session -d -s my_session 'cd ../ && python3 train.py --config configuration-ml-node'
