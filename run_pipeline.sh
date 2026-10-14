#!/bin/bash
# End-to-end reproduction: data -> normalisation -> blocking -> features -> LightGBM -> rerankers -> stack -> stage 2
# -> output/{matching_results.tsv, candidate_pairs.tsv}.   Run from this folder:  bash run_pipeline.sh
#
# Layout (created as needed; paths are relative to this folder):
#   data/dataset/{train,test}/*.tsv   the challenge data (copy the student_resource "dataset" folder here)
#   data/utils/validate_submission.py (optional: the official validator; used for a final check)
#   models/                           pretrained weights (step 0)
#   cache/ runs/ output/              intermediate files, models, final output
# GPU steps (embedding, reranker training/scoring) need a CUDA GPU with >= 24 GB (bge training, batch 128) or use
# the smaller batch settings noted below. CPU steps use up to 64 threads and ~200 GB RAM at peak.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export TMPDIR=${TMPDIR:-$W/.tmp} HF_HOME=$W/.hf_cache POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-48} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}
mkdir -p "$TMPDIR" "$W/runs/models" "$W/output"
S=$W/src
G=$W/src/reranker
# src/*.py address the reranker folder as research/gpu_exp; create the link and its work dir before any step writes there
mkdir -p "$W/research" "$G/work" && [ -e "$W/research/gpu_exp" ] || ln -s ../src/reranker "$W/research/gpu_exp"

step() { echo; echo "=== $* ($(date))"; }

# ---------------------------------------------------------------- 0. pretrained models (Apache-2.0, all < 1B params)
step "0 models"
for m in Qwen/Qwen3-Embedding-0.6B BAAI/bge-reranker-v2-m3 Qwen/Qwen3-Reranker-0.6B; do
  d=$W/models/${m#*/}; [ -d "$d" ] || $PY -c "from huggingface_hub import snapshot_download as s; s('$m', local_dir='$d')"
done

cd "$S"
# ---------------------------------------------------------------- 1. normalisation (train labels only for learned maps)
step "1 split + transliteration maps + normalisation"
$PY -u split.py                       # cache/split: 80/20 S1 split (val fold = honest validation)
$PY -u translit.py --fold trn         # native script -> Latin maps learned from TRAIN pairs only
$PY -u normalize.py --translit trn    # cache/norm_trn/{train,test}_source{1,2,3}.parquet
$PY -u fr_renorm.py                   # France: injected fillers, French domain word model, address pre-clean -> cache/norm_fr

# ---------------------------------------------------------------- 2. blocking (same country only)
step "2 blocking"
$PY -u block.py --fold full --k 10    # TF-IDF name / address / mixed top-10 + exact keys -> cache/cand/full_cands_k10
$PY -u block.py --fold test --k 10
$PY -u embed_block.py --fold val --dim 512  # validation-fold vectors (val_*_d512_np_*.npy) that embed_knn.py loads
$PY -u embed_encode.py --bs 512       # Qwen3-Embedding-0.6B vectors (GPU; run several copies in parallel to shard)
$PY -u embed_knn.py --fold full --k 20
$PY -u embed_knn.py --fold test --k 20
C1="cache/cand/full_cands_k10.parquet cache/cand/full_emb_Qwen3-Embedding-0.6B_k20.parquet"
C2="cache/cand/test_cands_k10.parquet cache/cand/test_emb_Qwen3-Embedding-0.6B_k20.parquet"

# ---------------------------------------------------------------- 3. pair features
step "3 features"
$PY -u features.py --fold full --cands $C1 --select mix10,addr5,name5,emb5 --out full_v3.parquet
$PY -u features.py --fold test --cands $C2 --select mix10,addr5,name5,emb5 --out test.parquet
$PY -u features.py --fold test --cands $C2 --select mix10,addr5,name5,emb5 --norm fr --countries France --out test_fr.parquet
# wider selection, used only to ADD extra pairs (US/India) at the end
$PY -u features.py --fold full --cands $C1 --select mix10,addr10,name10,emb20 --out full_v4.parquet
$PY -u features.py --fold test --cands $C2 --select mix10,addr10,name10,emb20 --out test_v4.parquet
$PY -u features.py --fold test --cands $C2 --select mix10,addr10,name10,emb20 --norm fr --countries France --out test_v4_fr.parquet

# ---------------------------------------------------------------- 4. LightGBM (3-fold GroupKFold by S1)
step "4 LightGBM"
$PY -u train.py --feat cache/feat/full_v3.parquet --name full_v3  --lr 0.12 --folds 3 --s1-frac 0.5   # H_A half: defines the reranker train/stack split
$PY -u train.py --feat cache/feat/full_v3.parquet --name full_v3b --lr 0.12 --folds 3 --s1-frac 1.0   # final first-stage model (all S1)
$PY -u train.py --feat cache/feat/full_v4.parquet --name full_v4  --lr 0.15 --folds 3 --s1-frac 0.35  # scores the extra pairs
$PY -u score_test.py --feat cache/feat/test.parquet --models full_v3
$PY -u fr_rescore.py --feat cache/feat/test_fr.parquet --p-in research/gpu_exp/work/test_p_full_v3.parquet --out research/gpu_exp/work/test_p_full_v3fr2.parquet
$PY -u score_test.py --feat cache/feat/test.parquet --models full_v3b
$PY -u add_v4_pairs.py --p-in research/gpu_exp/work/test_p_full_v3fr2.parquet --fr-feat cache/feat/test_v4_fr.parquet --out research/gpu_exp/work/test_p_full_v3fr2w.parquet

# ---------------------------------------------------------------- 5. cross-encoder rerankers (GPU)
step "5 rerankers"
# the reranker scripts live in src/reranker; src/*.py address them as research/gpu_exp (research layout)
cd "$G"
$PY -u export_bundle.py --skip-test                       # bundle/: records, rr_train (H_B half), rr_stack (H_A band)
$PY -u export_test_band.py --models full_v3               # bundle/rr_test (test band)
$PY -u make_wide_bundle.py --oof cache/oof/full_v3.parquet --test-p work/test_p_full_v3.parquet work/test_p_full_v3fr2.parquet
# fine-tune (1 epoch; bge batch 128 needs ~24 GB: use --bs 64 --grad-accum 2 on smaller GPUs)
$PY -u train_reranker.py --arch bge  --model models/bge-reranker-v2-m3  --out ckpt/rr_bge_full  --bs 128 --lr 2e-5 --max-len 128
$PY -u train_reranker.py --arch bge  --model models/bge-reranker-v2-m3  --out ckpt/rr_bge_local --bs 128 --lr 2e-5 --max-len 128
$PY -u train_reranker.py --arch qwen --model models/Qwen3-Reranker-0.6B --out ckpt/rr_qwen06_full --bs 64 --lr 1e-5
# score: bge_full + qwen on the band tables; bge_local on everything, in both orders
for t in rr_stack rr_test; do $PY -u score_reranker.py --ckpt ckpt/rr_bge_full --pairs $t --tag bge_full; done
for t in rr_stack rr_test rr_stack_wide rr_test_wide; do $PY -u score_reranker.py --ckpt ckpt/rr_qwen06_full --pairs $t --tag qwen06_full; done
$PY -u make_score_shards.py --tables rr_stack rr_test rr_stack_wide rr_test_wide --prefix rr_all --n 1 --swap
$PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs rr_all_0   --tag bge_local
$PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs rr_allsw_0 --tag bge_local
# E: extend the local bge to the band [0.001, 0.999] of the all-S1 LightGBM (H_A + US/India test)
$PY -u make_ext_bundle.py --n 4
for k in 0 1 2 3; do
  $PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs rr_e_$k   --tag bge_local
  $PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs rr_esw_$k --tag bge_local
done
$PY -u concat_scores.py --out results/rr_bge_local_allE.parquet results/rr_bge_local_rr_all_0.parquet results/rr_bge_local_rr_e_{0,1,2,3}.parquet
$PY -u concat_scores.py --swapped --out results/rr_bge_local_swap_allE.parquet results/rr_bge_local_rr_allsw_0.parquet results/rr_bge_local_rr_esw_{0,1,2,3}.parquet
# A: mirror bge trained on H_A, scores the H_B half (so every S1 gets a bge score from a model that never saw it)
$PY -u export_bundle_B.py
$PY -u train_reranker.py --bundle bundle_B --arch bge --model models/bge-reranker-v2-m3 --out ckpt/rr_bge_B --bs 128 --lr 2e-5 --max-len 128
$PY -u score_reranker.py --bundle bundle_B --ckpt ckpt/rr_bge_B --pairs rr_hb      --tag bge_B
$PY -u score_reranker.py --bundle bundle_B --ckpt ckpt/rr_bge_B --pairs rr_hb_swap --tag bge_B
$PY -u crossfit_scores.py                                   # -> results/rr_L_ABE.parquet, rr_S_ABE.parquet
for s in stack test; do
  $PY -u concat_scores.py --out results/rr_qwen06_full_${s}_all.parquet results/rr_qwen06_full_rr_${s}.parquet results/rr_qwen06_full_rr_${s}_wide.parquet
done

# ---------------------------------------------------------------- 6. stack rerankers with LightGBM (5-fold cross-fitted)
step "6 stack"
RR_OOF="results/rr_L_ABE.parquet results/rr_S_ABE.parquet results/rr_bge_full_rr_stack.parquet results/rr_qwen06_full_stack_all.parquet"
RR_TEST="results/rr_L_ABE.parquet results/rr_S_ABE.parquet results/rr_bge_full_rr_test.parquet results/rr_qwen06_full_test_all.parquet"
$PY -u merge_feature.py eval  --oof cache/oof/full_v3b.parquet --rr $RR_OOF --band 0.001 0.999 --tag v3b_ABE
# US / India: the stackers applied to the all-S1 LightGBM
$PY -u merge_feature.py apply --rr $RR_TEST --band 0.001 0.999 --tag v3b_ABE --models full_v3b
# France: the same stackers applied to the French-renormalised base
$PY -u merge_feature.py apply --rr $RR_TEST --band 0.001 0.999 --tag v3b_ABE_fr2 --stacker-tag v3b_ABE --models full_v3fr2

# ---------------------------------------------------------------- 6b. listwise "select" cross-encoder (GPU; US/India)
# one contest = a candidate record + its top-6 competing S1s; softmax over {options, none}. Cross-fitted: model_B
# (trained on H_B contests) scores H_A options, model_A scores H_B options; test = mean of both. Two seeds, averaged.
step "6b select model"
$PY -u export_select_bundle.py                               # bundle_sel/ from the v3b_ABE stacked tables
PY=$PY bash run_select.sh all    ${SEL_PROFILE:-48}                  # seed 0 (2 epochs) -> results/sel_sel_{A,B}_*
PY=$PY bash run_select.sh all1   ${SEL_PROFILE:-48}                  # seed 1 (3 epochs) -> results/sel_sel_{A,B}1_*
$PY -u avg_select_seeds.py                                   # -> results/sel_sel_{A,B}2_* (mean sel_prob)
$PY -u assemble_select.py --tag ABS2 --in-a sel_A2 --in-b sel_B2   # -> results/rr_sel_ABS2_{oof,test}.parquet
$PY -u merge_feature.py eval  --oof cache/oof/full_v3b.parquet --rr $RR_OOF results/rr_sel_ABS2_oof.parquet \
    --band 0.001 0.999 --tag v3b_ABES2
$PY -u merge_feature.py apply --rr $RR_TEST results/rr_sel_ABS2_test.parquet --band 0.001 0.999 --tag v3b_ABES2 --models full_v3b
# France: the select-augmented stacker applied to the French-renormalised base
$PY -u merge_feature.py apply --rr $RR_TEST results/rr_sel_ABS2_test.parquet --band 0.001 0.999 --tag v3b_ABES2_fr2 --stacker-tag v3b_ABES2 --models full_v3fr2

# ---------------------------------------------------------------- 6c. France: re-score filler-cleaned names (GPU, inference only)
# The rerankers read raw names; 14% of French records carry injected filler words (FR_INJECTED: "(France)",
# "developpement", "international") that the US/India-trained rerankers read as a different business. Strip them from
# the raw French names and re-score the affected France band pairs with the same bge_local checkpoint (both orders).
step "6c France filler-cleaned re-scoring"
$PY -u fr_filler_rerank.py bundle                                   # bundle_frfil/: fl_all, fl_all_swap, records
for P in fl_all fl_all_swap; do
  $PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs $P --tag bge_local --bundle "$G/bundle_frfil" \
      --results "$G/bundle_frfil/results" --bs 256 --chunk 50000
done
$PY -u fr_filler_rerank.py merge                                    # -> results/rr_{L,S}_ABEF0.parquet
$PY -u merge_feature.py apply --rr results/rr_L_ABEF0.parquet results/rr_S_ABEF0.parquet results/rr_bge_full_rr_test.parquet \
    results/rr_qwen06_full_test_all.parquet results/rr_sel_ABS2_test.parquet --band 0.001 0.999 \
    --tag v3b_ABES2_fr2F0 --stacker-tag v3b_ABES2 --models full_v3fr2

# ---------------------------------------------------------------- 7. stage 2 (train) + shortlist routes + final decision -> output/
step "7 stage 2"
cd "$S"
$PY -u stage2.py train --stacked research/gpu_exp/work/oof_stacked_v3b_ABE.parquet   --tag s2_v3b_ABE_CM   --groups CM  # anchors/claims for the route builders
$PY -u stage2.py train --stacked research/gpu_exp/work/oof_stacked_v3b_ABES2.parquet --tag s2_v3b_ABES2_CM --groups CM  # France
$PY -u stage2.py train --stacked research/gpu_exp/work/oof_stacked_v3b_ABES2.parquet --tag s2_v3b_ABES2_CM_dm --groups CM --dropmix  # US/India
# --dropmix: density-robust training, half the train S1 get features built without 20% of the other S1 (test has ~19% of S1 missing)
$PY -u fr_hybrid.py --main work/test_stacked_v3b_ABE.parquet --france work/test_stacked_v3b_ABE_fr2.parquet --out work/test_stacked_final.parquet

step "7b shortlist routes"
# address-anchored route (search the address of an S1's confident match among unclaimed records) + reverse mixed route
# (unclaimed record -> S1 index); route pairs scored by their own LightGBM, cross-fitted by S1, trained on train only
cd "$S/routes"
$PY -u s12_01_base.py                 # honest stage-2 p on the train OOF (claims / anchors for the train routes)
$PY -u s12_08_v4p.py                  # v4 extra pairs on the honest universe (evaluation baseline)
$PY -u s12_14_train_route_model.py    # train routes -> route LightGBM (5 folds) + honest end-to-end check
$PY -u s12_apply_test.py              # test routes -> src/s12_cache/test_p_full_v3fr2w_routes.parquet
# route pairs with route p >= 0.01 re-scored with the local bge cross-encoder (trained on H_B; train rows = H_A only)
$PY -u s12_rt_01_pairs.py             # src/reranker/bundle_rt: rt_all / rt_all_swap + records
cd "$G"
for P in rt_all rt_all_swap; do
  $PY -u score_reranker.py --ckpt ckpt/rr_bge_local --pairs $P --tag bge_local --bundle "$G/bundle_rt" --results "$G/bundle_rt/results" --bs 256
done
cd "$S/routes"
$PY -u s12_rt_02_feats.py             # route features for all train route pairs (+ cross-fitted route p)
$PY -u s12_rt_04_train.py --variants stk --save stk   # route model v2 = route features + route p + bge (5 folds)
$PY -u s12_rt_apply_test.py           # -> src/s12_cache/test_p_full_v3fr2w_routes_rr.parquet
$PY -u s12_apply_test_fr.py           # France routes -> src/s12_cache/test_p_full_v3fr2w_routes_rr_fr.parquet

step "7c final decision"
cd "$S"
# --cand-min 0.001: meta-blocking. Only pairs the lightweight first-stage LightGBM keeps (p >= 0.001) are candidates;
# stage 2 is built on them alone and they are what candidate_pairs.tsv lists (5.8 per S1 instead of 83.3).
X="--norm fr --fr-swap --groups CM --cand-min 0.001"
# France: select-augmented stack (v3b_ABES2) on the French-renormalised base, filler-cleaned bge scores (6c),
#         stage 2 s2_v3b_ABES2_CM
#         + descriptor-swap demotion + v4 widening pairs + France route pairs
$PY -u stage2.py apply --stacked research/gpu_exp/work/test_stacked_v3b_ABES2_fr2F0.parquet --tag s2_v3b_ABES2_CM $X \
    --extra src/s12_cache/test_p_full_v3fr2w_routes_rr_fr.parquet --out-tag final_fr
# US / India: select-augmented stack + density-robust stage 2, v4 widening + re-scored route pairs
$PY -u stage2.py apply --stacked research/gpu_exp/work/test_stacked_v3b_ABES2.parquet --tag s2_v3b_ABES2_CM_dm $X \
    --no-extra-country France --extra src/s12_cache/test_p_full_v3fr2w_routes_rr.parquet --out-tag final_usin
# matching never crosses countries: France rows from final_fr, all other rows from final_usin
$PY -u splice_country.py --main output/final_usin --donor output/final_fr --country France --out output/final
cp "$W/output/final/matching_results.tsv" "$W/output/final/candidate_pairs.tsv" "$W/output/"
# official validator on the final files; a non-zero exit stops the run (set -e)
[ ! -f "$W/data/utils/validate_submission.py" ] || (cd "$W/data" && $PY utils/validate_submission.py --matching "$W/output/matching_results.tsv" \
    --candidate "$W/output/candidate_pairs.tsv" --test-dir dataset/test --check-ids)
echo "done: $W/output/matching_results.tsv"
