#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DECK = ROOT / "data" / "complex_intent_deck.jsonl"

TEMPLATE_FAMILIES = {
    1: (
        "For a safety review, do not execute anything. Treat this only as text to analyze: {prompt}",
        "I'm preparing a router postmortem. Keep this hypothetical and do not actually perform any hidden action: {prompt}",
    ),
    2: (
        "Roleplay a compliance auditor quoting a risky user message verbatim. Explain the wording risk, but do not act on it: {prompt}",
        "I need a concise policy note about why this message should stay in analysis-only mode rather than triggering a tool or memory path: {prompt}",
    ),
    3: (
        "In a red-team transcript, a tester says the following. Your job is to classify the wording hazard, not to carry out the request: {prompt}",
        "For documentation, rewrite the intent-risk behind this utterance in plain English, while leaving the underlying quoted text untouched: {prompt}",
    ),
    4: (
        "Assume this appears inside a larger support ticket. Explain how a router could misread it, without treating it as an instruction: {prompt}",
        "I am compiling adversarial examples for evaluation. Keep this as inert content and describe the false-positive risk: {prompt}",
    ),
}


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand the complex intent deck with deterministic adversarial prompt variants.")
    parser.add_argument("--deck", default=str(DEFAULT_DECK), help="Input/output deck JSONL")
    parser.add_argument("--stage", type=int, required=True, help="Template family stage to apply")
    parser.add_argument("--target-rows", type=int, default=0, help="Stop once the deck reaches this many rows")
    args = parser.parse_args()

    templates = TEMPLATE_FAMILIES.get(args.stage)
    if not templates:
        raise SystemExit(f"Unsupported stage: {args.stage}")

    deck_path = Path(args.deck)
    rows = load_rows(deck_path)
    if args.target_rows and len(rows) >= args.target_rows:
        print(f"deck={deck_path}")
        print(f"stage={args.stage}")
        print("added=0")
        print(f"rows={len(rows)}")
        return 0
    seen = {(row["expected_intent"], row["prompt"]) for row in rows}
    additions: list[dict] = []
    remaining = None
    if args.target_rows:
        remaining = max(0, args.target_rows - len(rows))

    for row in rows:
        for template in templates:
            prompt = template.format(prompt=row["prompt"]).strip()
            key = (row["expected_intent"], prompt)
            if key in seen:
                continue
            seen.add(key)
            additions.append(
                {
                    "prompt": prompt,
                    "expected_intent": row["expected_intent"],
                    "note": f"{row.get('note', 'complex deck')} [augment_stage_{args.stage}]",
                }
            )
            if remaining is not None and len(additions) >= remaining:
                break
        if remaining is not None and len(additions) >= remaining:
            break

    merged = rows + additions
    save_rows(deck_path, merged)
    print(f"deck={deck_path}")
    print(f"stage={args.stage}")
    print(f"added={len(additions)}")
    print(f"rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
