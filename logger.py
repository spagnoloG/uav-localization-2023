import logging
import os
from datetime import datetime
import sys

log_dir = "/tmp/uav-localization-2023-log/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a custom logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler(stream=sys.stdout)
f_handler = logging.FileHandler(
    f'{log_dir}/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

format_str = "[%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s] -> %(message)s"
c_format = logging.Formatter(format_str)
f_format = logging.Formatter(format_str)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)
