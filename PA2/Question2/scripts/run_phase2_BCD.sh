#!/bin/zsh
# Phase 2 (after the Q2.11 predictions are committed): models B, C, the model-D candidates and D.
# Runs whose checkpoint already exists are skipped, so the script can be resumed.
# The optional D+B model is trained separately by run_phase3_DB.sh.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python

run() {  # run <name> <seed> <train args...>
  local name=$1 seed=$2; shift 2
  if [ -f checkpoints/${name}_s${seed}.pt ]; then echo "skip ${name}_s${seed} (exists)"; return; fi
  $PY -m mm.train --name $name --seed $seed "$@" | tee logs/train_${name}_s${seed}.log
}

for s in 0 1; do run B $s --arch A --sym-aug; done
for s in 0 1; do run C $s --arch A --speed-aug; done
# model D: candidate designs trained with the full D protocol (seed 0), chosen on the validation split
for c in dil12 k55h46 p8; do run Dcand_$c 0 --arch $c --speed-aug; done
BEST=$($PY -m scripts.select_D | tail -1)
echo "chosen D design: $BEST"
run D 1 --arch $BEST --speed-aug
