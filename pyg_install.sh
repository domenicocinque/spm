#!/usr/bin/env bash

DEFAULTVALUE='cu102'
CUDA="${1:-$DEFAULTVALUE}"

echo "Selected CUDA version: $CUDA"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install class_resolver