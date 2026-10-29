#!/bin/zsh
# Phase 3 (optional D+B): model D's architecture with speed AND symmetry augmentation, seeds 0 and 1.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
ARCH=$(python3 -c "import json; print(json.load(open('results/model_D_selection.json'))['chosen'])")
for s in 0 1; do
  if [ -f checkpoints/DB_s${s}.pt ]; then echo "skip DB_s${s} (exists)"; continue; fi
  $PY -m mm.train --name DB --arch $ARCH --speed-aug --sym-aug --seed $s | tee logs/train_DB_s$s.log
done
