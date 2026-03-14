# intentautoresearch

This repo is for autonomous experiments on pure intent classification.

## Goal

Minimize `val_intent_loss` and improve `val_macro_f1` on the validation split.

## Rules

- You may modify `train.py`.
- You should not modify `prepare.py` unless the dataset contract itself needs to change.
- The target is intent classification, not next-token language modeling.
- Keep changes simple and measurable.

## Metrics

Primary:
- `val_intent_loss` (lower is better)

Secondary:
- `val_macro_f1` (higher is better)
- `val_accuracy` (higher is better)

## Workflow

1. Run `uv run prepare.py`
2. Run `uv run train.py`
3. Compare the new validation metrics to the previous run
4. Keep only changes that improve the classifier meaningfully
