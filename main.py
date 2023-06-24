#!/usr/bin/env python3
import argparse
import resource
import logging


logging.basicConfig(level=logging.INFO)


# -------------
# MEMORY SAFETY
# -------------
memory_limit_gb = 24
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (memory_limit_gb * 1024**3, hard))


# ----------------
# ARGUMENT PARSING
# ----------------
parser = argparse.ArgumentParser(description="Run the localization model")
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="test the model")
parser.add_argument(
    "--realtime", action="store_true", help="test the model in realtime"
)

args = parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
