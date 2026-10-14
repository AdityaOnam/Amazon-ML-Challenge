#!/bin/bash
# R10 listwise select cross-encoder: stage runner for the remote GPU server (48 GB profile by default).
#   bash run_select.sh <stage> [gpu profile: 24|48|80]
#   stages: smoke | train_B | score_B | train_A | score_A | all | all1 (second seed)
#   model_B (trained on H_B contests) scores oof_A + test; model_A (trained on H_A) scores oof_B + test.
# Expects $W/research/gpu_exp/{train_select,score_select,sel_model,common}.py, bundle_sel/, and
# $W/models/bge-reranker-v2-m3/ (BAAI base model, Apache-2.0, 567,755,777 params).
# Returns: results/sel_sel_{A,B}_{oof_A|oof_B,test}.parquet + logs/sel_*.log (+ ckpt/sel_{A,B}, bf16 ~1.1 GB each).
set -euo pipefail
W=${W:-$(cd "$(dirname "$0")/../.." && pwd)}
G=$W/research/gpu_exp
PROFILE=${2:-48}
export HF_HOME=$W/.hf_cache TORCH_HOME=$W/.torch_cache TMPDIR=$W/.tmp PIP_CACHE_DIR=$W/.pip_cache
mkdir -p "$TMPDIR" "$G/logs"
PY=${PY:-$W/.venv/bin/python}
BASE=${BASE:-models/bge-reranker-v2-m3}
DEVICE=${DEVICE:-cuda}   # DEVICE=cpu bash run_select.sh smoke  -> CPU check
cd "$G"

# contests: <= 7 records x ~26 tokens (p99 ~380) -> max_len 384. amp = fp32 master weights + bf16 autocast.
case "$PROFILE" in
  24) TR="--bs 8 --grad-accum 4 --grad-ckpt"; SC="--bs 64" ;;
  48) TR="--bs 32";                           SC="--bs 256" ;;
  *)  TR="--bs 64";                           SC="--bs 512" ;;
esac
COMMON="--device $DEVICE --model $BASE --max-len 384 --lr 2e-5 --epochs 2 --eval-every 500"
# second seed (averaged with the first by avg_select_seeds.py): same settings, 3 epochs, seed 1
COMMON1="--device $DEVICE --model $BASE --max-len 384 --lr 2e-5 --epochs 3 --eval-every 500 --seed 1"

train() {  # $1 = half
  $PY -u train_select.py --half "$1" --out "ckpt/sel_$1" $COMMON $TR 2>&1 | tee "logs/sel_train_$1.log"
}
score() {  # $1 = model half, $2 = OOF set it is allowed to score
  $PY -u score_select.py --ckpt "ckpt/sel_$1" --set "$2" --tag "sel_$1" --device $DEVICE $SC 2>&1 | tee "logs/sel_score_$1_$2.log"
  $PY -u score_select.py --ckpt "ckpt/sel_$1" --set test --tag "sel_$1" --device $DEVICE $SC 2>&1 | tee "logs/sel_score_$1_test.log"
}

case "$1" in
smoke)  # ~1-2 min on GPU: every script end-to-end on a few contests, nothing kept
  S=${SMOKE_DIR:-$G/smoke_sel}
  $PY -u train_select.py --half B --out "$S/ckpt" $COMMON --bs 8 --max-rows 512 --max-steps ${SMOKE_STEPS:-20} --eval-every 10 --holdout 0.1
  $PY -u score_select.py --ckpt "$S/ckpt" --set oof_A --tag smoke --results "$S/results" --limit 300 --chunk 100 --device $DEVICE
  $PY -u score_select.py --ckpt "$S/ckpt" --set test  --tag smoke --results "$S/results" --limit 300 --chunk 100 --device $DEVICE
  $PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  rm -rf "$S"
  echo SMOKE OK
  ;;
train_B) train B ;;
score_B) score B oof_A ;;
train_A) train A ;;
score_A) score A oof_B ;;
all) train B; score B oof_A; train A; score A oof_B ;;
all1)  # seed 1 -> ckpt/sel_{A,B}1, results/sel_sel_{A,B}1_*
  for h in B A; do
    $PY -u train_select.py --half $h --out "ckpt/sel_${h}1" $COMMON1 $TR 2>&1 | tee "logs/sel1_train_$h.log"
    o=$([ $h = B ] && echo oof_A || echo oof_B)
    $PY -u score_select.py --ckpt "ckpt/sel_${h}1" --set $o  --tag "sel_${h}1" --device $DEVICE $SC 2>&1 | tee "logs/sel1_score_${h}_$o.log"
    $PY -u score_select.py --ckpt "ckpt/sel_${h}1" --set test --tag "sel_${h}1" --device $DEVICE $SC 2>&1 | tee "logs/sel1_score_${h}_test.log"
  done ;;
*) echo "unknown stage $1"; exit 1 ;;
esac
