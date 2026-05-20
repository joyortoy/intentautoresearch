#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare import default_data_files, iter_examples


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "good_intent_clean_deck.jsonl"
DEFAULT_ALLOWED_INTENTS = (
    "general_chat",
    "tech_support",
    "task_request",
    "plan_project",
)


def parse_allowed(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a benign good-intent training deck from normalized sources.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output JSONL path")
    parser.add_argument(
        "--allowed-intents",
        default=",".join(DEFAULT_ALLOWED_INTENTS),
        help="Comma-separated intents to keep",
    )
    args = parser.parse_args()

    allowed_intents = parse_allowed(args.allowed_intents)
    files = default_data_files()
    if not files:
        raise SystemExit("No dataset files found for good-intent deck build.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for path in files:
        for example in iter_examples(path):
            if example.intent not in allowed_intents:
                continue
            key = (example.intent, example.text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "prompt": example.text,
                    "expected_intent": example.intent,
                    "note": f"good intent seed from {example.source}",
                }
            )

    if not rows:
        raise SystemExit("No good-intent rows found in selected sources.")

    rows.sort(key=lambda row: (row["expected_intent"], row["prompt"]))
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"files={len(files)}")
    print(f"out={out_path}")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
