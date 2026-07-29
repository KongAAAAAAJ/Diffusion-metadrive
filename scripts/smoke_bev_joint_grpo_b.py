#!/usr/bin/env python3
"""Run one diagnostic Variant-B joint GRPO update on a packed real sample."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.bev_joint_grpo_diagnostic import diagnostic_main


if __name__ == "__main__":
    diagnostic_main("B")
