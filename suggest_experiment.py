from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

from ollama_llm import chat, env_flag, provider_label


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "description": "baseline",
    "INTENT_AUTORESEARCH_LR": "0.002",
    "INTENT_AUTORESEARCH_BATCH_SIZE": "16",
    "INTENT_AUTORESEARCH_EMBED_DIM": "128",
    "INTENT_AUTORESEARCH_HIDDEN_DIM": "192",
    "INTENT_AUTORESEARCH_DROPOUT": "0.1",
    "INTENT_AUTORESEARCH_WEIGHT_DECAY": "0.01",
    "INTENT_AUTORESEARCH_MAX_LEN": "48",
    "INTENT_AUTORESEARCH_TIME_BUDGET": "30",
}
ALLOWED_KEYS = [
    "description",
    "INTENT_AUTORESEARCH_LR",
    "INTENT_AUTORESEARCH_BATCH_SIZE",
    "INTENT_AUTORESEARCH_EMBED_DIM",
    "INTENT_AUTORESEARCH_HIDDEN_DIM",
    "INTENT_AUTORESEARCH_DROPOUT",
    "INTENT_AUTORESEARCH_WEIGHT_DECAY",
    "INTENT_AUTORESEARCH_MAX_LEN",
    "INTENT_AUTORESEARCH_TIME_BUDGET",
]


def load_best(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: str(v) for k, v in data.items()})
    return merged


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def heuristic_candidate(best: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    run_idx = len(rows)

    def clone(desc: str, **updates: str) -> dict[str, str]:
        cfg = dict(best)
        cfg.update(updates)
        cfg["description"] = desc
        return cfg

    candidates = [
        clone("baseline-repeat"),
        clone("lower-lr", INTENT_AUTORESEARCH_LR="0.001"),
        clone("higher-lr", INTENT_AUTORESEARCH_LR="0.004"),
        clone("smaller-batch", INTENT_AUTORESEARCH_BATCH_SIZE="8"),
        clone("larger-batch", INTENT_AUTORESEARCH_BATCH_SIZE="24"),
        clone("wider-hidden", INTENT_AUTORESEARCH_HIDDEN_DIM="256"),
        clone("smaller-hidden", INTENT_AUTORESEARCH_HIDDEN_DIM="128"),
        clone("larger-embed", INTENT_AUTORESEARCH_EMBED_DIM="192"),
        clone("lower-dropout", INTENT_AUTORESEARCH_DROPOUT="0.05"),
        clone("higher-dropout", INTENT_AUTORESEARCH_DROPOUT="0.2"),
        clone("stronger-weight-decay", INTENT_AUTORESEARCH_WEIGHT_DECAY="0.03"),
        clone(
            "low-lr-strong-reg",
            INTENT_AUTORESEARCH_LR="0.0005",
            INTENT_AUTORESEARCH_DROPOUT="0.2",
            INTENT_AUTORESEARCH_WEIGHT_DECAY="0.03",
        ),
        clone("shorter-context", INTENT_AUTORESEARCH_MAX_LEN="32"),
        clone("longer-context", INTENT_AUTORESEARCH_MAX_LEN="64"),
    ]
    proposal = candidates[run_idx % len(candidates)]
    proposal["_planner_source"] = "fallback"
    proposal["_planner_note"] = "heuristic-candidate"
    return proposal


def summarize_rows(rows: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    recent = rows[-limit:]
    return [
        {
            "description": row.get("description", ""),
            "status": row.get("status", ""),
            "val_intent_loss": row.get("val_intent_loss", ""),
            "val_accuracy": row.get("val_accuracy", ""),
            "val_macro_f1": row.get("val_macro_f1", ""),
        }
        for row in recent
    ]


def build_prompt(best: dict[str, str], rows: list[dict[str, str]]) -> str:
    return (
        "You are selecting the next intent classifier experiment for a local autoresearch loop.\n"
        "Goal: improve macro_f1 first, then accuracy, then lower val_intent_loss.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Allowed keys only:\n"
        f"{json.dumps(ALLOWED_KEYS)}\n"
        "All values must be strings.\n"
        "Constraints:\n"
        "- lr between 0.0001 and 0.004\n"
        "- batch size in {8,16,24,32}\n"
        "- embed dim in {96,128,160,192,224}\n"
        "- hidden dim in {128,160,192,224,256,320}\n"
        "- dropout between 0.05 and 0.25\n"
        "- weight decay between 0.0 and 0.08\n"
        "- max len in {32,48,64,80}\n"
        "- time budget in {20,30,45,60}\n"
        "- description should be short and specific\n"
        "Current best config:\n"
        f"{json.dumps(best, indent=2)}\n"
        "Recent results:\n"
        f"{json.dumps(summarize_rows(rows), indent=2)}\n"
        "Prefer a nearby but non-identical configuration unless recent results justify a larger jump."
    )


def extract_json(text: str) -> dict[str, str]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in planner response")
    candidate = json.loads(match.group(0))
    if not isinstance(candidate, dict):
        raise ValueError("planner response is not a JSON object")
    return candidate


def normalize(proposal: dict[str, object], best: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in ALLOWED_KEYS:
        value = proposal.get(key, best.get(key, ""))
        normalized[key] = str(value)
    if not normalized["description"].strip():
        normalized["description"] = "prometheus-proposal"
    return normalized


def call_llm(prompt: str) -> dict[str, object]:
    raw = chat(prompt, expect_json=True)
    return extract_json(raw)


def main() -> int:
    best_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "best_config.json"
    results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results.tsv"
    best = load_best(best_path)
    rows = load_rows(results_path)
    fallback = heuristic_candidate(best, rows)

    if not env_flag("INTENT_AUTORESEARCH_USE_LLM_PLANNER", "INTENT_AUTORESEARCH_USE_CODEX_PLANNER", True):
        print(json.dumps(fallback, indent=2))
        return 0

    try:
        proposal = call_llm(build_prompt(best, rows))
        normalized = normalize(proposal, best)
        normalized["_planner_source"] = provider_label()
        normalized["_planner_note"] = f"ollama:{provider_label()}"
        print(json.dumps(normalized, indent=2))
        return 0
    except Exception as exc:
        fallback["_planner_note"] = f"fallback-after-llm-error: {exc}"
        print(json.dumps(fallback, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
