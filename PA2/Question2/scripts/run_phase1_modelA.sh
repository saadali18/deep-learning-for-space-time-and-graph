#!/bin/zsh
# Phase 1: train the baseline (model A) with seeds 0 and 1.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
for s in 0 1; do $PY -m mm.train --name A --arch A --seed $s | tee logs/train_A_s$s.log; done
