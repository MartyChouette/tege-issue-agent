"""Thin wrapper so `python eval/run_eval.py` works from a checkout. Same as `enjin-triage eval`."""

import sys

from triage.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
