# Intent Autoresearch Deck

- Total runs: `2`
- Keep: `1`
- Discard: `1`
- Crash: `0`
- Selection mode: `macro_f1`

## Current Best

- Commit: `a37505f`
- Description: best-lr-down
- Val intent loss: `0.023562`
- Val accuracy: `0.9907`
- Val macro F1: `0.9796`
- Memory GB: `0.0`

## Best Config

```json
{
  "description": "best-lr-down",
  "INTENT_AUTORESEARCH_LR": "0.0017",
  "INTENT_AUTORESEARCH_BATCH_SIZE": "16",
  "INTENT_AUTORESEARCH_EMBED_DIM": "128",
  "INTENT_AUTORESEARCH_HIDDEN_DIM": "192",
  "INTENT_AUTORESEARCH_DROPOUT": "0.1",
  "INTENT_AUTORESEARCH_WEIGHT_DECAY": "0.01",
  "INTENT_AUTORESEARCH_MAX_LEN": "48",
  "INTENT_AUTORESEARCH_TIME_BUDGET": "30"
}
```

## Recent Runs

| Status | Loss | Acc | Macro F1 | Memory GB | Description |
| --- | ---: | ---: | ---: | ---: | --- |
| discard | 0.023738 | 0.9907 | 0.9793 | 0.0 | best-lr-up |
| keep | 0.023562 | 0.9907 | 0.9796 | 0.0 | best-lr-down |
