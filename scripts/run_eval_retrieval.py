#!/usr/bin/env python3
"""CLI wrapper for generic retrieval evaluation pipeline."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.eval_retrieval import run_eval_retrieval


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval evaluation pipeline")
    parser.add_argument("--profile", default="qwen_v2", help="Pipeline profile name")
    args = parser.parse_args(argv)
    run_eval_retrieval(args.profile)


if __name__ == "__main__":
    main()
