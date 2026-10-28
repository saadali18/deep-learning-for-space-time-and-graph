#!/bin/zsh
# Phase 2 (after the Q2.11 predictions are committed): models B, C, D candidates, D, D+B.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
for s in 0 1; do $PY -m mm.train --name B --arch A --sym-aug --seed $s | tee logs/train_B_s$s.log; done
for s in 0 1; do $PY -m mm.train --name C --arch A --speed-aug --seed $s | tee logs/train_C_s$s.log; done
# model D: candidate designs trained with the full D protocol (seed 0), chosen on the validation split
for c in dil12 k55h46 p8; do $PY -m mm.train --name Dcand_$c --arch $c --speed-aug --seed 0 | tee logs/train_Dcand_${c}_s0.log; done
BEST=$($PY -m scripts.select_D | tail -1)
echo "chosen D design: $BEST"
$PY -m mm.train --name D --arch $BEST --speed-aug --seed 1 | tee logs/train_D_s1.log
# optional D+B: same architecture as D, speed and symmetry augmentation
for s in 0 1; do $PY -m mm.train --name DB --arch $BEST --speed-aug --sym-aug --seed $s | tee logs/train_DB_s$s.log; done
