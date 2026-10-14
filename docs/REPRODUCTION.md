# Reproduction guide

This regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data, end to end:

normalisation → blocking → pair features → LightGBM → cross-encoder rerankers + listwise select model → LightGBM stack → stage-2 re-scorer → second-pass shortlist routes → one-to-one decision

The methodology is described in [METHODOLOGY.md](METHODOLOGY.md).

## 1. Environment
- Python 3.12, Linux.
- One CUDA GPU with at least 24 GB for the reranker fine-tunes. 20 GB works with the smaller batch settings noted in `run_pipeline.sh`.
- Up to 64 CPU threads and about 200 GB RAM at peak.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA index of your driver
pip install -r requirements.txt
```

## 2. Data layout (relative to the repository root)
```
data/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
data/dataset/test/test_source{1,2,3}.tsv
data/utils/validate_submission.py        (optional, the official validator; used for the final check)
```

## 3. Run
```bash
bash run_pipeline.sh            # from the repository root, or PY=/path/to/python bash run_pipeline.sh
```
- **Step 0** downloads the three pretrained models from Hugging Face into `models/`. This is the only network access: pretrained weights, no data.
- Everything the pipeline creates goes to `cache/`, `runs/`, `src/reranker/{bundle*,ckpt,results,work}` and `output/`.
- The final files are copied to `output/`, and the official validator is run on them.

**Run time on one A100/L40S-class GPU plus 64 cores:** about 20–30 h. Most of it is:
- LightGBM training, 3 fold models × 3 variants;
- reranker fine-tuning, 4 models of about 1 epoch each, plus 4 select models (2 halves × 2 seeds, 2–3 epochs; `SEL_PROFILE=24|48|80` picks the batch settings for the GPU memory);
- reranker scoring, about 15M pairs in total.

The GPU scoring steps can be sharded across several GPUs: `make_score_shards.py` and `make_ext_bundle.py --n`.

## 4. Code map
| Step | Script (`src/`) | What it does |
|---|---|---|
| 1 | `split.py` | 80/20 split of the train S1 records (the val fold is the honest validation) |
| 1 | `translit.py` | native script → Latin maps, learned from TRAIN pairs only |
| 1 | `normalize.py` | name and address normalisation (legal forms, abbreviations, house numbers, city/state) |
| 1 | `fr_renorm.py` | France only: injected-filler removal; a French word model (built from the test set's own French names) for run-together domain names; `N°`/`bis`/dash address clean-up |
| 2 | `block.py`, `embed_encode.py`, `embed_knn.py`, `embed_block.py`, `text_tables.py` | candidate generation: TF-IDF top-k on name, address and mixed text, exact keys, and Qwen3-Embedding-0.6B nearest neighbours; always within the same country |
| 3 | `features.py` | 69 pair features (name, address and house-number similarities, token rarity, blocker ranks, candidate competition) |
| 4 | `train.py`, `score_test.py`, `fr_rescore.py`, `add_v4_pairs.py` | LightGBM (3-fold GroupKFold by S1), test scoring, the French re-scored base, and the extra pairs from the wider "v4" shortlist (all countries) |
| 5 | `reranker/export_bundle.py`, `export_test_band.py`, `make_wide_bundle.py`, `make_ext_bundle.py`, `export_bundle_B.py` | pair tables for the rerankers (two S1 halves H_A/H_B; train on one half, score the other plus test) |
| 5 | `reranker/train_reranker.py`, `score_reranker.py`, `rr_model.py`, `common.py` | cross-encoder fine-tuning and scoring (bge-reranker-v2-m3, Qwen3-Reranker-0.6B) |
| 5 | `reranker/concat_scores.py`, `make_score_shards.py`, `crossfit_scores.py` | shard handling, and the cross-fitted bge scores (each S1 is scored by a model that never trained on its half) |
| 6b | `reranker/export_select_bundle.py`, `train_select.py`, `score_select.py`, `sel_model.py`, `run_select.sh`, `avg_select_seeds.py`, `assemble_select.py` | listwise select cross-encoder: contests (one candidate + up to 6 competing S1s), cross-fitted A/B, two seeds averaged |
| 6c | `reranker/fr_filler_rerank.py`, `score_reranker.py` | France: strip injected filler words (FR_INJECTED) from raw French names and re-score the affected band pairs with bge_local (both orders); France stack re-applied (tag v3b_ABES2_fr2F0) |
| 6 | `reranker/merge_feature.py` | LightGBM stacker over the first-stage p plus the reranker scores (+ the select score for US/India; 5-fold cross-fitted) |
| 7b | `routes/s12_*.py` | second-pass shortlist routes (address-anchored + reverse mixed search among unclaimed records), route LightGBM, route pairs re-scored with the bge cross-encoder (`s12_rt_*`); France routes (`s12_apply_test_fr.py`) |
| 7 | `stage2.py`, `decide.py`, `fr_hybrid.py`, `fr_swap.py`, `splice_country.py` | stage-2 re-scorer (house-number edit types, agreement with the S1's confident matches, competition between S1s, raw features); meta-blocking cut (`--cand-min 0.001`: only first-stage p >= 0.001 pairs are candidates, 5.80 per S1); one-to-one per candidate; threshold 0.75; France combination and the descriptor-swap rule |

`io_utils.py` holds the shared paths.

`run_pipeline.sh` links `research/gpu_exp` to `src/reranker`, because the scripts address the reranker work folder by that research-layout path.

## 5. Honest validation
- Every model used for the validation number is cross-fitted by S1: no S1 is scored by a model that saw its labels.
- `stage2.py train` prints the honest macro F0.5 on the held-out S1 (val fold ∩ first-stage OOF sample).
- Core recipe (select-augmented stack + stage 2): **0.99093** at threshold 0.75 (India 0.99097, US 0.99090).
- Full recipe including the v4 widening pairs and the re-scored routes: **0.99180** (India 0.99227, US 0.99149); `routes/s12_rt_04_train.py` prints the end-to-end number.
