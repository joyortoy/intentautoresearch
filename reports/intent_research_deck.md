# Intent Autoresearch Deck

- Total runs: `6`
- Keep: `1`
- Discard: `3`
- Crash: `2`
- Selection mode: `loss`

## Current Best

- Commit: `nogit`
- Description: lower-lr
- Val intent loss: `1.241451`
- Val accuracy: `0.5714`
- Val macro F1: `0.4857`
- Memory GB: `0.0`

## Best Config

```json
{
  "description": "lower-lr",
  "INTENT_AUTORESEARCH_LR": "0.001",
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
| discard | 1.316143 | 0.5714 | 0.4857 | 0.0 | lower-lr |
| discard | 1.316143 | 0.5714 | 0.4857 | 0.0 | lower-lr |
| crash | 0.000000 | 0.0000 | 0.0000 | 0.0 | smaller-batch |
| discard | 7.206530 | 0.3333 | 0.1667 | 0.0 | higher-lr |
| keep | 1.241451 | 0.5714 | 0.4857 | 0.0 | lower-lr |
| crash | 0.000000 | 0.0000 | 0.0000 | 0.0 | baseline-repeat |
