# Initial Testing Metadata Findings

This document records the currently public-safe testing metadata available in this repository.

## Verified best observed run

Source files:

- `good_results.tsv`
- `good_best_config.json`
- `reports/good_intent_research_deck.md`

Observed result:

| Field | Value |
| --- | --- |
| Commit | `17c1000` |
| Description | `best-lr-down` |
| Status | `keep` |
| Validation intent loss | `0.023388` |
| Validation accuracy | `0.9906` |
| Validation macro F1 | `0.9754` |
| Memory GB | `0.045` in `good_results.tsv`; rounded to `0.0` in deck output |
| Selection mode | `macro_f1` |

## Training configuration

```json
{
  "description": "best-lr-down",
  "INTENT_AUTORESEARCH_LR": "0.001955",
  "INTENT_AUTORESEARCH_BATCH_SIZE": "16",
  "INTENT_AUTORESEARCH_EMBED_DIM": "128",
  "INTENT_AUTORESEARCH_HIDDEN_DIM": "192",
  "INTENT_AUTORESEARCH_DROPOUT": "0.1",
  "INTENT_AUTORESEARCH_WEIGHT_DECAY": "0.01",
  "INTENT_AUTORESEARCH_MAX_LEN": "48",
  "INTENT_AUTORESEARCH_TIME_BUDGET": "30"
}
```

## Model/backend provenance status

The current public result artifacts record the training configuration and validation metrics, but they do **not** yet clearly record the exact model/backend name used for the run.

Current interpretation:

- Training objective: intent classification
- Primary metrics: `val_intent_loss`, `val_accuracy`, `val_macro_f1`
- Best observed run status: kept
- Exact model/backend name: not yet explicitly captured in public run metadata

## Recommended future metadata fields

Future run artifacts should include:

```text
run_id
lineage_id
parent_run_id
model_name
model_backend
embedding_backend
telemetry_backend
dataset_version
dataset_hash
policy_version
policy_hash
trust_score
trust_decision
attention_direction_score
representation_drift_score
calibration_suite_version
```

## Research status

These findings are early prototype evidence only. They show that the local intent-classification experiment loop produced a kept run with high validation accuracy and macro F1, but they do not yet validate attention-direction governance, representation telemetry, or alignment safety claims.

The next research step is to connect run metadata to calibration outputs and representation telemetry so that drift signals can be compared against observed safe, unsafe, ambiguous, and deceptive prompt behavior.
