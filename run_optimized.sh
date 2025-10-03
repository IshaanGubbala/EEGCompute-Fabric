#!/bin/bash

# Use project temp dir to avoid disk full issues
export TMPDIR="$(pwd)/.tmp"
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

# MPS memory optimization flags
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Aggressive memory recycling
export PYTORCH_ENABLE_MPS_FALLBACK=0  # Fail fast on unsupported ops

# Run the pipeline
python scripts/pipeline.py --config configs/v3.yaml --full 0
