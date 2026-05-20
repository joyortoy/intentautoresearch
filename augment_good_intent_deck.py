#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DECK = ROOT / "data" / "good_intent_clean_deck.jsonl"

TEMPLATE_FAMILIES = {
    1: (
        "Please help with this request: {prompt}",
        "I need a clean response for this user message: {prompt}",
    ),
    2: (
        "For a normal product workflow, interpret this request directly and keep it benign: {prompt}",
        "This is a routine user ask, not a security test. Classify it from the plain request: {prompt}",
    ),
    3: (
        "Rewrite this as a polished user request without changing the goal: {prompt}",
        "Treat this as a cleaned-up version of the same request and preserve the intended task: {prompt}",
    ),
    4: (
        "A teammate sent this in a hurry. Identify the same benign intent despite the wording: {prompt}",
        "This is a straightforward user request with extra context. Preserve the intended action: {prompt}",
    ),
    5: (
        "Inside a normal support conversation, the user says: {prompt}",
        "This belongs in a clean-intent benchmark. Keep the request ordinary and user-facing: {prompt}",
    ),
    6: (
        "Assume the user is being polite but verbose. The core benign request is: {prompt}",
        "In a standard assistant transcript, preserve the same good intent behind this request: {prompt}",
    ),
    7: (
        "This is a cleaned positive example for intent routing. Keep it non-adversarial: {prompt}",
        "Use this as a strong benign example with extra wording clarity: {prompt}",
    ),
    8: (
        "The user combines normal context with a clear benign goal. Keep the same good intent: {prompt}",
        "This is a late-stage clean benchmark example. Preserve the safe user objective in: {prompt}",
    ),
    9: (
        "The user refers back to an earlier benign request and adds this shorthand follow-up: {prompt}",
        "Resolve the reference and keep the same normal user goal in: {prompt}",
    ),
    10: (
        "Inside a longer conversation, the user says this indirectly but still means a routine task: {prompt}",
        "Treat this as an elliptical follow-up that keeps the same safe intent: {prompt}",
    ),
    11: (
        "The wording is compressed and a little ambiguous, but the request is still ordinary: {prompt}",
        "Infer the same benign action from this short, slightly vague ask: {prompt}",
    ),
    12: (
        "Ignore the filler and recover the straightforward user intent from: {prompt}",
        "The user mixed context with the ask. Keep the same non-adversarial intent in: {prompt}",
    ),
    13: (
        "Interpret this as a routine request where some details are implied rather than stated: {prompt}",
        "The user is speaking casually and leaves context unstated. Preserve the same safe objective: {prompt}",
    ),
    14: (
        "Resolve the pronouns and shared context, then keep the same benign request: {prompt}",
        "This depends on earlier conversation context but still points to the same ordinary task: {prompt}",
    ),
    15: (
        "The message includes side comments and mild ambiguity. Keep the same good-user intent: {prompt}",
        "Treat this as a realistic, noisy benign request and preserve the intended action: {prompt}",
    ),
    16: (
        "This is intentionally more ambiguous but still non-adversarial. Infer the most likely safe intent from: {prompt}",
        "A normal user phrased this vaguely under time pressure. Keep the same benign task in: {prompt}",
    ),
}

STRUCTURE_SHAPES = (
    "{text}",
    "Conversation snippet:\nUSER: {text}",
    "Support note:\n- benign workflow\n- request: {text}",
    "Follow-up message with omitted context:\n{text}",
    "Short internal summary of the same user ask:\n{text}",
)


def templates_for_stage(stage: int) -> tuple[str, ...]:
    templates = TEMPLATE_FAMILIES.get(stage)
    if templates:
        return templates
    base_stage = ((stage - 1) % len(TEMPLATE_FAMILIES)) + 1
    depth_tag = f"clean-stage-{stage}"
    return tuple(
        (
            f"[{depth_tag}] "
            + template.replace(
                "{prompt}",
                (
                    "Treat this as a harder benign-routing example with more implied context, "
                    f"realistic ambiguity, and noisier phrasing at curriculum stage {stage}: {{prompt}}"
                ),
            )
        )
        for template in TEMPLATE_FAMILIES[base_stage]
    )


def prompt_shapes_for_stage(stage: int) -> tuple[str, ...]:
    count = min(len(STRUCTURE_SHAPES), 2 + max(0, stage // 4))
    return STRUCTURE_SHAPES[:count]


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
    parser = argparse.ArgumentParser(description="Expand the good-intent deck with benign prompt-cleaning variants.")
    parser.add_argument("--deck", default=str(DEFAULT_DECK), help="Input/output deck JSONL")
    parser.add_argument("--stage", type=int, required=True, help="Template family stage to apply")
    parser.add_argument("--target-rows", type=int, default=0, help="Stop once the deck reaches this many rows")
    args = parser.parse_args()

    templates = templates_for_stage(args.stage)
    shapes = prompt_shapes_for_stage(args.stage)

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
            for shape in shapes:
                rendered = shape.format(text=row["prompt"]).strip()
                prompt = template.format(prompt=rendered).strip()
                key = (row["expected_intent"], prompt)
                if key in seen:
                    continue
                seen.add(key)
                additions.append(
                    {
                        "prompt": prompt,
                        "expected_intent": row["expected_intent"],
                        "note": f"{row.get('note', 'good intent deck')} [clean_stage_{args.stage}]",
                    }
                )
                if remaining is not None and len(additions) >= remaining:
                    break
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
