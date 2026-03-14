from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_TSV = Path(os.getenv("INTENT_AUTORESEARCH_RESULTS_TSV", str(ROOT / "results.tsv")))
DECK_MD = Path(os.getenv("INTENT_AUTORESEARCH_DECK_MD", str(ROOT / "reports" / "intent_research_deck.md")))
BEST_JSON = Path(os.getenv("INTENT_AUTORESEARCH_BEST_CONFIG_JSON", str(ROOT / "best_config.json")))
SELECTION_MODE = os.getenv("INTENT_AUTORESEARCH_SELECTION_MODE", "loss")


def load_rows() -> list[dict]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def best_row(rows: list[dict]) -> dict | None:
    keep_rows = []
    for row in rows:
        try:
            loss = float(row.get("val_intent_loss") or "inf")
            acc = float(row.get("val_accuracy") or "0")
            f1 = float(row.get("val_macro_f1") or "0")
        except ValueError:
            continue
        if row.get("status") == "keep":
            if SELECTION_MODE == "macro_f1":
                keep_rows.append(((f1, acc, -loss), row))
            else:
                keep_rows.append(((-loss,), row))
    if not keep_rows:
        return None
    keep_rows.sort(key=lambda item: item[0], reverse=True)
    return keep_rows[0][1]


def fmt_float(value: str, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return value or "-"


def build_deck(rows: list[dict]) -> str:
    total_runs = len(rows)
    crash_runs = sum(1 for row in rows if row.get("status") == "crash")
    keep_runs = sum(1 for row in rows if row.get("status") == "keep")
    discard_runs = sum(1 for row in rows if row.get("status") == "discard")
    champion = best_row(rows)
    lines = [
        "# Intent Autoresearch Deck",
        "",
        f"- Total runs: `{total_runs}`",
        f"- Keep: `{keep_runs}`",
        f"- Discard: `{discard_runs}`",
        f"- Crash: `{crash_runs}`",
        f"- Selection mode: `{SELECTION_MODE}`",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Commit: `{champion.get('commit', '-')}`",
                f"- Description: {champion.get('description', '-')}",
                f"- Val intent loss: `{fmt_float(champion.get('val_intent_loss', ''), 6)}`",
                f"- Val accuracy: `{fmt_float(champion.get('val_accuracy', ''), 4)}`",
                f"- Val macro F1: `{fmt_float(champion.get('val_macro_f1', ''), 4)}`",
                f"- Memory GB: `{fmt_float(champion.get('memory_gb', ''), 1)}`",
                "",
            ]
        )
    if BEST_JSON.exists():
        best_cfg = json.loads(BEST_JSON.read_text(encoding="utf-8"))
        lines.extend(["## Best Config", "", "```json", json.dumps(best_cfg, indent=2, ensure_ascii=False), "```", ""])
    lines.extend(
        [
            "## Recent Runs",
            "",
            "| Status | Loss | Acc | Macro F1 | Memory GB | Description |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows[-12:][::-1]:
        lines.append(
            "| {status} | {loss} | {acc} | {f1} | {mem} | {desc} |".format(
                status=row.get("status", "-"),
                loss=fmt_float(row.get("val_intent_loss", ""), 6),
                acc=fmt_float(row.get("val_accuracy", ""), 4),
                f1=fmt_float(row.get("val_macro_f1", ""), 4),
                mem=fmt_float(row.get("memory_gb", ""), 1),
                desc=row.get("description", "-").replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = load_rows()
    DECK_MD.parent.mkdir(parents=True, exist_ok=True)
    DECK_MD.write_text(build_deck(rows), encoding="utf-8")
    print(DECK_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
