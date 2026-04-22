#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for Qwen v2 encoder training."""

from scripts.run_train_encoder import main


if __name__ == "__main__":
    main(["--profile", "qwen_v2"])
