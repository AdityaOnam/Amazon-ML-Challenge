# Output files

`run_pipeline.sh` writes two files here. Both are tab-separated with a header row and one line per source-1 (S1) entity.

| File | Columns | Size | Description |
|---|---|---:|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | ~98 MB | Final submission: S1 ids with their matched S2/S3 ids (comma-separated). 5,843,770 matches in total. |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | ~152 MB | Candidates that survived meta-blocking: 10,049,484 pairs, 5.80 per S1. |

The `.tsv` files are git-ignored because GitHub rejects files over 100 MB. Download them from the repository's **Releases** page, or regenerate them with `bash run_pipeline.sh`.
