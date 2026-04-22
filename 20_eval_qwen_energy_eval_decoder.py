#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for Qwen v2 energy_eval with decoder."""

from scripts.run_eval_mcq import main


if __name__ == "__main__":
    main(["--profile", "qwen_v2"])
