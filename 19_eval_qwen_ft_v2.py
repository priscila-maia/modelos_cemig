#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for Qwen v2 retrieval evaluation."""

from scripts.run_eval_retrieval import main


if __name__ == "__main__":
    main(["--profile", "qwen_v2"])
