# Intent Autoresearch Deck

- Total runs: `1`
- Keep: `1`
- Discard: `0`
- Crash: `0`
- Selection mode: `macro_f1`

## Current Best

- Commit: `17c1000`
- Description: best-lr-down
- Val intent loss: `0.023388`
- Val accuracy: `0.9906`
- Val macro F1: `0.9754`
- Memory GB: `0.0`

## Best Config

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

## Recent Runs

| Status | Loss | Acc | Macro F1 | Memory GB | Description |
| --- | ---: | ---: | ---: | ---: | --- |
| keep | 0.023388 | 0.9906 | 0.9754 | 0.0 | best-lr-down |
