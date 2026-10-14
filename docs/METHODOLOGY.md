# Methodology: Business Entity Resolution

**Team Name:** 01BIT
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
The solution is a cascade.
1. **Recall-oriented blocking.** TF-IDF on name, address and mixed text, exact keys, and Qwen3-Embedding-0.6B nearest neighbours, always within the same country.
2. **LightGBM pair model** over 69 engineered string, address, house-number and competition features.
3. **Fine-tuned cross-encoder rerankers** (bge-reranker-v2-m3 ×3, Qwen3-Reranker-0.6B), combined by a cross-fitted LightGBM stacker.
4. **Listwise "select" cross-encoder** (bge-reranker-v2-m3, two seeds averaged) that reads a candidate together with all S1s competing for it and picks one of them or "none"; its score is a fifth stacker input.
5. **Stage-2 re-scorer** that looks at each S1's other confident matches and at competing S1s; for US/India it is trained density-robust (half of the training S1 see a table with 20% of the other S1 removed, because the test has about 19% of its S1 missing).
6. **Second-pass shortlist routes** (all countries): after a first decision, the address of an S1's confident match and a reverse search from still-unclaimed records propose a few new pairs, scored by their own LightGBM; for US/India the route pairs are additionally re-scored with the cross-encoder (France route pairs use the route LightGBM only).
7. **Decision:** a one-to-one assignment per candidate at the F0.5-optimal threshold of 0.75.

France does not appear in training. It is handled by language-aware normalisation built only from the test inputs themselves (no external data), a rule against sibling-business confusions, and (final version) cross-encoder scores recomputed on French names cleaned of generator-injected filler words.

**Scores:** honest held-out macro F0.5 of the full recipe **0.99180** (US/India: India 0.99227, US 0.99149); public leaderboard 0.989467 for the previous best (S3), **final version (F0) 0.989635**. Candidate set: **5.80 pairs per S1** after meta-blocking (Section 3).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Name noise:** legal-form drift (Pvt Ltd / Private Limited / Corp), abbreviations, word transpositions, typos, transliteration (Devanagari and other native scripts ↔ Latin), trade names, and names written as domains or handles (`abcmotors.com`).
- **Address noise:**
  - abbreviations (Rd/Road, St/Street);
  - missing PIN or state, landmarks ("Near SBI ATM"), reordering;
  - house numbers that differ by one digit, which are often a **different** business;
  - many candidate records with an **empty address**.
- **Multi-match S1s:** many S1s have 2–5 true matches across S2 and S3.
  - Most of the residual loss is partial recall on such S1s.
  - Singletons are common, and any prediction on a singleton scores 0.
- **Ambiguity floor:** about 63% of our remaining validation loss is pairs where the candidate has no address and **several S1s share the same name**.
  - Nothing in the records says which S1 owns such a candidate.
  - The model's probabilities there are calibrated coin flips.
- **France (test only):**
  - French names are built from template descriptors (club, école, comité, amicale…);
  - injected filler tokens occur;
  - addresses use `N°`, `bis` and CEDEX, and domain-style names are run together.
  - A model trained on US/India treats a swapped descriptor ("Club X" vs "École X") like a typo, which it is not in France.
- **Metric:** F0.5 is precision-heavy, so a single wrong merge on a singleton costs a full point for that S1. The whole design is tuned for precision first.

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Blocking, then gradient-boosted classifier, then cross-encoder rerankers with stacking, then a set-level re-scorer and one-to-one decision.

**Core Innovations:**
1. **Honest cross-fitting everywhere.** The train S1s are split into halves H_A/H_B.
   - Each reranker is fine-tuned on one half and scores the other half plus test.
   - A mirror model covers the other direction, so every S1 gets a reranker score from a model that never saw its labels.
   - The stacker and stage 2 are therefore trained on realistic, test-like scores.
2. **Stage-2 "collective" re-scorer.** Re-scores each pair using:
   - agreement with the S1's other confident matches ("anchors");
   - house-number edit types;
   - competition between S1s for the same candidate.
3. **Candidate-side listwise selection.** Pairwise models score (S1, candidate) independently; the select model sees the whole contest (one candidate, up to 6 competing S1s) and resolves sibling-S1 conflicts, which are 10% of the residual loss.
4. **Second-pass routes.** Confident first-pass matches are used as anchors to search for the S1's remaining matches (same address, unclaimed records), recovering blocking misses in crowded-name groups with 96% precision.
5. **Unseen-country adaptation without external data or test labels.** For France:
   - normalisation and a word-splitting model built from the test set's own French names (unsupervised statistics, disclosed in §5);
   - a descriptor-swap rule.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used.** All routes run within the same country only.
  - TF-IDF (char n-gram and word) cosine top-10 on name, on address, and on mixed name+address.
  - Exact keys: house number + city; the name without spaces.
  - Qwen3-Embedding-0.6B dense top-20 (name | address text, 512-dim Matryoshka vectors, cosine kNN).
  - A per-S1 selection keeps the mixed top-10, address top-5, name top-5, embedding top-5 and all exact-key hits.
  - A wider "v4" selection (top-10/10/10/20) adds extra pairs, scored by a separate LightGBM (all countries).
  - Second-pass routes (all countries): an address-anchored search (top-3 records sharing the address of an S1's confident match that no S1 has claimed) and a reverse mixed-text search (unclaimed record → S1 index, top-3).
- **Meta-blocking (learned candidate pruning).** Retrieval returns 83.3 pairs per S1. Each is scored by a lightweight LightGBM on cheap string-similarity features (the stage-1, v4 or route LightGBM); only pairs with p >= 0.001 become candidates. All heavy models (the cross-encoder rerankers, the select model, the stacker and stage 2) run only on these candidates, and stage 2 builds its competition features from them alone.
- **Candidate pairs generated (test):** 10,049,484 pairs for 1,732,544 S1s (**5.80 per S1**). These are the pairs listed in `candidate_pairs.tsv`. Before meta-blocking, retrieval produced 144,290,036 pairs (83.3 per S1).
- **Meta-blocking cut vs accuracy (honest, US/India, stage 2 rebuilt on the pruned set):**

  | First-stage cut | Pairs per S1 | F0.5 |
  |---|---:|---:|
  | none | 89.1 | 0.99180 |
  | p >= 0.0001 | 7.0 | 0.99180 |
  | **p >= 0.001 (used)** | **5.5** | **0.99180** |
  | p >= 0.005 | 4.8 | 0.99177 |
  | p >= 0.02 | 4.0 | 0.99165 |

  On test the cut leaves `matching_results.tsv` byte-identical to the uncut run.
  The full retrieval output (144,290,036 pairs, 83.3 per S1) is regenerated by the same code without `--cand-min` and can be provided if the first-stage input set is the one to be assessed.
- **Retrieval widening before meta-blocking (honest, US/India).** The core selection alone is 45.8 pairs/S1. The v4 widening doubles the set for +0.00029 F0.5. Narrower widening rules trade pairs for score almost linearly:

  | Widening kept (per-channel rank caps) | Pairs per S1 | Δ F0.5 |
  |---|---:|---:|
  | full (addr10, name10, emb20) | 89.1 | 0 |
  | addr7, name7, emb20 | 79.3 | −0.00003 |
  | addr5, name5, emb15 | 63.9 | −0.00008 |
  | addr5, name5, emb10 | 54.6 | −0.00013 |
  | none | 45.8 | −0.00029 |

  Because the leaderboard margins between the top teams are about 0.00003, we kept the full set. Similarity-threshold pruning and tighter core rank caps were also measured and cost ≥ 0.0008 F0.5 at 44 pairs/S1.
- **How we ensured true matches were not lost:**
  - Recall was measured on the held-out train fold after every change. The final union recall is **98.8%**.
  - The union of complementary views (lexical name, lexical address, semantic embedding, exact keys) recovers different miss types.
  - Wider shortlists were only kept when the end-to-end honest F0.5 improved. For example, extra exact-key routes on sorted or phonetic names added millions of low-precision pairs and **lowered** F0.5 by 0.007, so they were dropped.

---

## 4. Matching Model

**Features used** (69 features in the first-stage LightGBM):
- **Name:**
  - RapidFuzz ratio, token-sort, token-set, partial and full token-set;
  - Jaro-Winkler and Levenshtein on the no-space name; phonetic token-set;
  - alias and legal-form handling, acronym match, core, sorted and no-space equality;
  - token Jaccard, coverage and intersection; lengths; domain-name flag;
  - IDF rarity of the rarest shared token, IDF sums, and name frequency in the pool.
- **Address:**
  - token-set, ratio and partial similarity; phonetic similarity;
  - token and numeric-token Jaccard, coverage and intersection;
  - house-number equality, Levenshtein, digit equality and containment;
  - city equality and "city in address"; empty-address and address-length flags.
- **Other:**
  - blocker ranks and scores (TF-IDF name/address/mixed, embedding);
  - which source the record comes from (S2 or S3); number of candidates per S1 and per source;
  - rank and gap to the top mixed score;
  - **candidate competition:** the number of S1s competing for the candidate, the best other S1's score, and the margin.

**Model type:**
- **Stage 1:** LightGBM binary classifier, 3-fold GroupKFold by S1, trained on all train S1s (93M pairs, 1,500 trees, lr 0.12).
- **Rerankers:** cross-encoders fine-tuned for 1 epoch on band-sampled hard pairs plus sampled easy pairs.
  - BAAI/bge-reranker-v2-m3 ×3: "bge_full", and "bge_local" + "bge_B", the cross-fitted pair.
  - bge_local is scored in both input orders.
  - Qwen/Qwen3-Reranker-0.6B (yes/no logit).
  - The rerankers score every pair whose stage-1 p is in [0.001, 0.999].
- **Select model (listwise):** bge-reranker-v2-m3 with a contest head. Input = the candidate record followed by up to 6 competing S1s (stacked p > 0.05), max 384 tokens; output = softmax over {S1_1..S1_k, none}. Cross-fitted like the rerankers (model_B trained on H_B contests scores H_A options, model_A the reverse; test = mean of both). Two seeds (2 and 3 epochs) are averaged.
- **Stacker:** LightGBM over the stage-1 logit, the four reranker scores and the select score (all countries; France applies the same stacker to its French-renormalised stage-1 base). Per score, it uses rank, gap and max within the S1 and within the candidate. It is 5-fold cross-fitted by S1.
- **Stage 2:** LightGBM over all pairs with p ≥ 0.001. Its features are:
  - house-number edit type;
  - agreement with the S1's confident anchors;
  - competition between S1s on the stacked p;
  - raw stage-1 features and all reranker scores.
  - It is 5-fold cross-fitted by S1.
- **Route model:** LightGBM (5 folds by S1) over the route features (anchor agreement, address/name similarity, claim state); route pairs with p ≥ 0.01 are re-scored by a second LightGBM that adds the bge cross-encoder scores in both input orders.
- **France:**
  - the French-normalised base features, and the four-score stacker and its stage-2 model;
  - a descriptor-swap demotion: a candidate that replaces one frequent French descriptor word with another, e.g. club → école, is a sibling business and is set to p = 0;
  - filler-cleaned cross-encoder scores (final version): the rerankers read raw names, and 14% of French records carry generator-injected filler words ("(France)", "Développement", "International"; the list whose removal from the LightGBM name features gave +0.001 on the leaderboard in v6). Trained on US/India only, the rerankers read them as a different business: at stage-1 p ≥ 0.99, pairs with a filler on one side only got a mean bge logit of 0.31 vs 5.79 and were accepted 59% vs 93% of the time. The filler words are stripped from the raw French names and the 197,520 affected France band pairs are re-scored with the same bge_local checkpoint in both input orders (inference only; `src/reranker/fr_filler_rerank.py`, pipeline step 6c). Effect on test: 4,236 French S1 rows change (+3,896 / −413 pairs; 3.2858 → 3.2992 matches per S1); US/India rows are unchanged;
  - no wider v4 pairs.

**Threshold selection method:**
- Macro F0.5 is maximised on the held-out S1 fold. The grid 0.65–0.85 is flat around the best value, **0.75**.
- A greedy one-to-one assignment gives each candidate to at most one S1 (the highest p). It is equivalent to the Hungarian solution here, because S1s have no capacity limit.

**Models, licences and exact parameter counts** (counted from the safetensors):

| Model | Use | Licence | Parameters |
|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | blocking embeddings (not fine-tuned) | Apache-2.0 | 595,776,512 |
| BAAI/bge-reranker-v2-m3 | 3 fine-tuned cross-encoders + 4 listwise select models (2 halves × 2 seeds) | Apache-2.0 | 567,755,777 each |
| Qwen/Qwen3-Reranker-0.6B | 1 fine-tuned cross-encoder | Apache-2.0 | 595,776,512 (tied embeddings) |
| LightGBM 4.7.0 | stage 1, v4 model, stacker, stage 2, route models | MIT | tree ensembles |

- Every model is ≤ 8B parameters.
- No API models, no external data, no LLM-generated data.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), honest held-out validation (US/India), full recipe:** **0.99180** (India 0.99227, US 0.99149). The core selection with stage 2 alone gives 0.99093; the v4 widening and the routes add the rest.

| Version | Change | Honest val (core / full recipe) | Public LB |
|---|---|---:|---:|
| v3 | blocking + LightGBM | 0.98085 | 0.976 |
| v5 | + bge reranker stack | 0.98850 | 0.985 |
| v6/v7 | + French normalisation | – | 0.986 |
| v12 | + stage 2, wide band, Qwen, all-S1 LightGBM | 0.99049 | 0.98807 |
| v13 | + cross-fitted bge for all S1, band 0.001–0.999, all reranker scores in stage 2 | 0.99079 / 0.99109 | 0.98821 |
| v13HG | v13 + France: wide stack, stage 2, descriptor-swap rule | same (France has no labels) | > v13 |
| v14 | + second-pass shortlist routes (US/India) | 0.99079 / 0.99154 | 0.98912 |
| v15u | + listwise select model (seed 0) | 0.99088 / – | – |
| v18u | + route pairs re-scored with the cross-encoder; select model averaged over two seeds | 0.99093 / 0.99180 | 0.989255 |
| (not adopted) | + density-robust stage 2, France self-training | 0.99091 / – | 0.988958 |
| U2 | v18u + density-robust stage 2 (US/India) + v4 widening and routes for France + meta-blocking | 0.99091 (0.99088 test-like) / 0.99180 | 0.989401 |
| (not adopted) | U2 + France threshold 0.60 | – | 0.989349 |
| (not adopted) | U2 + French-normalised TF-IDF retrieval pairs for France | – | 0.989256 |
| S3 | U2 + select-augmented stack and its stage 2 for France | same (France has no labels) | 0.989467 |
| (not adopted) | S3 + per-country rank-normalised stacker inputs (R9) | 0.99089 / – | 0.98932 |
| **final (F0)** | S3 + France reranker scores recomputed on filler-cleaned names | same (France has no labels) | **0.989635** |

- **Where the remaining 0.0092 loss is (held-out):**

  | Part | Share |
  |---|---:|
  | True pairs scored below the threshold | 44% |
  | Blocking misses | 38% |
  | Sibling S1 took the candidate | 10% |
  | False matches | 6% |

  63% of all remaining loss is the ambiguity floor (empty candidate address + shared S1 name).
- **Common false positives (wrong merges):**
  - near-duplicate records of neighbouring branches whose house number differs by one digit;
  - the wrong owner among same-name sibling S1s;
  - a few singleton S1s with a lookalike candidate.
- **Common false negatives (missed matches):**
  - candidates with no address whose name is shared by several S1s (inherently ambiguous);
  - invented or unrelated trade names at an identical address;
  - native-script records;
  - blocking misses for names shared by more than 20 S1s, where siblings crowd the name-based blockers.

**Disclosures:**
1. **Unsupervised statistics from test inputs** (no labels, no training on test records). These are:
   - TF-IDF vocabularies and token document frequencies for blocking and features (fitted per split, including test);
   - for France: the French word-frequency model used to split run-together domain names, the injected-filler list, and the descriptor vocabulary of the swap rule (words in ≥0.2% of French test S1 names, minus common city words).
2. `wordninja` (MIT) ships a bundled English word-frequency list. It is used to split run-together English names, as a software component, not a dataset.
3. **Hand-written normalisation dictionaries.** `text_tables.py` holds small hand-written lists: legal-form tokens (~40) and phrases (12), street/unit abbreviations, US state and Indian state names/abbreviations, French region names with their départements (used only to compare the region field), and 19 historical spelling variants of Indian city names (e.g. bombay → mumbai, calcutta → kolkata). No gazetteer, postal table or packaged geo data is used.
4. **Leaderboard use for France.** France has no training labels, so the choice between France variants (normalisation fixes, and the final France stack/stage-2/swap combination) was confirmed on the public leaderboard. No test labels were used anywhere else.
5. **No external data.** No pseudo-labelling or self-training in the final submission, and no synthetic records. Pretrained weights are the only downloads. Unsupervised statistics on the test inputs (item 1) were confirmed as allowed by the organisers' FAQ.

---

## 6. Conclusion
The recipe that worked:
- a high-recall, same-country blocking union;
- a feature-rich LightGBM;
- **cross-fitted** fine-tuned cross-encoders stacked on top;
- a set-aware second stage.

Honest cross-fitting was the key lesson. Every stacking layer learns from realistic scores, so validation gains transferred to the leaderboard. We also learned to measure end to end: several "obvious" recall boosts lowered F0.5 because of the precision-heavy metric.

---

## Appendix

### A. Code Artefacts
The repository root contains:
- `run_pipeline.sh`, the single entry point that regenerates both output files;
- [`REPRODUCTION.md`](REPRODUCTION.md): environment, data layout, run time and code map;
- `requirements.txt`, pinned;
- `src/`: normalisation, blocking, features, LightGBM, stage 2, France handling;
- `src/reranker/`: bundles, cross-encoder training and scoring, the listwise select model, stacker;
- `src/routes/`: second-pass shortlist routes and their route models;
- `src/splice_country.py`: combines the France output with the US/India output (matching never crosses countries).

### B. Additional Results
- **Threshold grid (core recipe with select, honest):** 0.70 → 0.99092 · **0.75 → 0.99093** · 0.80 → 0.99090. The stacker alone, before stage 2, gives 0.99067 at 0.75.
- **French self-training, not adopted:** v18u + density-robust stage 2 + one round of French self-training scored 0.988958 (< 0.989255). A row-level diff showed the loss came from France: self-training removed 8,818 French pairs, 53% of them with p >= 0.99. The density-robust stage 2 alone was then adopted (final version, 0.989401), together with the France recall additions (v4 widening + routes: +8,338 / −105 French pairs). The select model was then extended to France (+0.000066 on the LB).
- **Also tried on France, not adopted:** a lower France threshold (0.60: −0.000052 LB); pairs from a French-normalised TF-IDF retrieval (−0.000145 LB); sibling-contrast stage-2 features (no signal on US/India: 0.99078 vs 0.99079); a contest rule (p ≥ 0.95 required when a second S1 has p ≥ 0.5: no effect after stage 2); filling empty S1 rows and count-aware thresholds (no gain on US/India); more select seeds (3 seeds 0.99090, 4 seeds 0.99091 vs 2 seeds 0.99093). Per-country rank-normalised stacker inputs (R9: −0.000147 LB; it added 20k French pairs the rerankers had doubted, which is what pointed to the filler problem fixed in the final version). Extending the reranker band to France's p_lgb < 0.003 / > 0.997 pairs was built but not submitted (it removed 4,151 confident French matches for reasons the filler words did not explain).
- **Select model, third seed:** averaging three seeds gave stage-2 0.99090 vs 0.99093 for two, so two seeds are kept.
- **Test-like density check:** dropping 20% of the non-validation S1 (the test has fewer competing S1 per candidate) keeps every adopted gain: select model +0.00010 (+0.00014 test-like), second select seed +0.00005 (both), routes +0.00043, route re-scoring +0.00014.
- **Leave-one-country-out check of the reranker stack:** India→US +0.0070 and US→India +0.0104 over LightGBM alone. This supports transfer to the unseen country.
