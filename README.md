# intentautoresearch

This repo is a pure intent-classification variant of `autoresearch`.

The objective is not next-token language modeling. The model is trained to
predict an intent label from a user prompt using cross-entropy loss, then
evaluated on:

- `val_intent_loss` (lower is better)
- `val_accuracy` (higher is better)
- `val_macro_f1` (higher is better)

## Dataset format

The trainer accepts JSONL records in any of these shapes:

```json
{"prompt": "hello", "expected_intent": "general_chat"}
{"text": "hello", "intent": "general_chat"}
{"instruction": "Extract the user intent from this message: 'hello'", "output": "general_chat"}
```

By default `prepare.py` will use:

1. local IntentStack eval files if present
2. the bundled seed dataset in `data/seed_intent_examples.jsonl`

You can override sources with:

```bash
INTENT_AUTORESEARCH_DATA_FILES=/abs/path/a.jsonl:/abs/path/b.jsonl uv run prepare.py
```

## Quick start

```bash
cd /home/sam/projects/intentautoresearch
uv sync
uv run prepare.py
uv run train.py
```

## Research loop

Run one experiment cycle with result logging and a readable deck:

```bash
cd /home/sam/projects/intentautoresearch
./run_intent_autoresearch.sh
```

This will:

- refresh normalized dataset artifacts
- choose one experiment config, using local Ollama Prometheus by default
- run training
- append a row to `results.tsv`
- regenerate the markdown deck at `reports/intent_research_deck.md`

## Canonical Service And Report Path

The active forever-loop service is being normalized around the
`intentautoresearch-loop.service` name:

```bash
systemctl --user status intentautoresearch-loop.service
```

The older `autoresearch-loop.service` name is kept as a compatibility unit
while you migrate.

For one consolidated local report that covers live IntentStack health, the
active `intentautoresearch` loops, and the older
`/home/sam/projects/autoresearch` qwen embedding repo, use:

```bash
python3 autoresearch_status.py report
./report_autoresearch.sh
```

## Good Intent Cleaning Loop

There is also a separate benign-data loop focused on cleaning and expanding normal user requests rather than adversarial prompts:

```bash
cd /home/sam/projects/intentautoresearch
./run_good_intent_forever.sh
```

This loop uses:

- `data/good_intent_clean_deck.jsonl`
- `good_results.tsv`
- `good_best_config.json`
- `reports/good_intent_research_deck.md`

On stagnation it:

- appends more benign training rows to the good-intent deck
- raises a model-capacity stage for the next planner proposal

## Local planner and summary model

The planner and Discord summary path now use local Ollama instead of Codex.
Defaults:

```bash
INTENT_AUTORESEARCH_LLM_MODEL=sroecker/prometheus2
INTENT_AUTORESEARCH_OLLAMA_URL=http://127.0.0.1:11434
```

Useful toggles:

```bash
INTENT_AUTORESEARCH_USE_LLM_PLANNER=0
INTENT_AUTORESEARCH_USE_LLM_SUMMARY=0
```

Legacy `INTENT_AUTORESEARCH_USE_CODEX_PLANNER` and `INTENT_AUTORESEARCH_USE_CODEX_SUMMARY`
are still accepted as compatibility aliases.

## Outputs

`train.py` prints a final summary like:

```text
---
val_intent_loss:  0.412345
val_accuracy:     0.8731
val_macro_f1:     0.8419
training_seconds: 300.0
total_seconds:    305.2
peak_vram_mb:     512.0
num_steps:        1200
num_params_M:     1.2
num_classes:      7
```

## Design

- fixed time budget
- pure classification objective
- simple portable tokenizer and encoder
- no dependency on the LM `val_bpb` path

## Result deck

Human-readable history is written to:

- `results.tsv`
- `reports/intent_research_deck.md`

The deck includes:

- current best config
- recent runs
- keep/discard/crash counts
