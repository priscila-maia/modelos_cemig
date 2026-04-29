#!/usr/bin/env python3
"""Run MCQ evaluation pipeline from a YAML config file."""

import argparse
import os
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.eval_mcq import run_eval_mcq


def _to_env_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping/object at top-level.")

    profile = data.get("profile", "qwen_v2")
    env_block = data.get("env", {})
    if not isinstance(env_block, dict):
        raise ValueError("Field 'env' must be a mapping/object.")

    return profile, env_block


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run MCQ evaluation pipeline from YAML config"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    profile, env_block = _load_config(config_path)

    for key, value in env_block.items():
        if value is None:
            continue
        os.environ[str(key)] = _to_env_value(value)

    print(f"Loaded YAML config: {config_path}")
    print(f"Profile: {profile}")
    run_eval_mcq(profile)


if __name__ == "__main__":
    main()
