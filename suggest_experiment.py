from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

from ollama_llm import chat, env_flag, provider_label


ROOT = Path(__file__).resolve().parent
PLANNER_SYSTEM_PROMPT = (
    "You are a strict JSON generator. "
    "Return exactly one JSON object with double-quoted keys and string values only. "
    "Do not add commentary, markdown, or explanations."
)
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
BASE_OPTION_SETS = {
    "INTENT_AUTORESEARCH_BATCH_SIZE": [8, 16, 24, 32],
    "INTENT_AUTORESEARCH_EMBED_DIM": [96, 128, 160, 192, 224],
    "INTENT_AUTORESEARCH_HIDDEN_DIM": [128, 160, 192, 224, 256, 320],
    "INTENT_AUTORESEARCH_MAX_LEN": [32, 48, 64, 80],
    "INTENT_AUTORESEARCH_TIME_BUDGET": [20, 30, 45, 60],
}
LOCAL_KEYS = [
    "INTENT_AUTORESEARCH_LR",
    "INTENT_AUTORESEARCH_BATCH_SIZE",
    "INTENT_AUTORESEARCH_EMBED_DIM",
    "INTENT_AUTORESEARCH_HIDDEN_DIM",
    "INTENT_AUTORESEARCH_DROPOUT",
    "INTENT_AUTORESEARCH_WEIGHT_DECAY",
    "INTENT_AUTORESEARCH_MAX_LEN",
    "INTENT_AUTORESEARCH_TIME_BUDGET",
]


def env_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def stagnation_profile() -> dict[str, int]:
    profile: dict[str, int] = {}
    for env_name, key in (
        ("INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM", "INTENT_AUTORESEARCH_EMBED_DIM"),
        ("INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM", "INTENT_AUTORESEARCH_HIDDEN_DIM"),
        ("INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN", "INTENT_AUTORESEARCH_MAX_LEN"),
        ("INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET", "INTENT_AUTORESEARCH_TIME_BUDGET"),
    ):
        value = env_int(env_name)
        if value is not None:
            profile[key] = value
    return profile


def dynamic_option_set(key: str) -> list[int]:
    values = list(BASE_OPTION_SETS.get(key, []))
    minimum = stagnation_profile().get(key)
    if minimum is None:
        return values
    while not values or values[-1] < minimum:
        if not values:
            values.append(minimum)
            break
        step = values[-1] - values[-2] if len(values) >= 2 else max(8, values[-1] // 4)
        step = max(step, 8)
        values.append(values[-1] + step)
    return values


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def adjust_choice(current: str, allowed: list[int], direction: int) -> str:
    try:
        value = int(float(current))
    except ValueError:
        value = allowed[0]
    choices = sorted(set(allowed))
    if value not in choices:
        choices.append(value)
        choices.sort()
    idx = choices.index(value)
    idx = max(0, min(len(choices) - 1, idx + direction))
    return str(choices[idx])


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def load_best(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_CONFIG)
    for key, value in data.items():
        text = str(value).strip()
        if text:
            merged[key] = text
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

    lr = float(best.get("INTENT_AUTORESEARCH_LR", "0.002"))
    dropout = float(best.get("INTENT_AUTORESEARCH_DROPOUT", "0.1"))
    weight_decay = float(best.get("INTENT_AUTORESEARCH_WEIGHT_DECAY", "0.01"))
    batch = best.get("INTENT_AUTORESEARCH_BATCH_SIZE", "16")
    embed = best.get("INTENT_AUTORESEARCH_EMBED_DIM", "128")
    hidden = best.get("INTENT_AUTORESEARCH_HIDDEN_DIM", "192")
    max_len = best.get("INTENT_AUTORESEARCH_MAX_LEN", "48")
    time_budget = best.get("INTENT_AUTORESEARCH_TIME_BUDGET", "30")

    batch_choices = dynamic_option_set("INTENT_AUTORESEARCH_BATCH_SIZE")
    embed_choices = dynamic_option_set("INTENT_AUTORESEARCH_EMBED_DIM")
    hidden_choices = dynamic_option_set("INTENT_AUTORESEARCH_HIDDEN_DIM")
    max_len_choices = dynamic_option_set("INTENT_AUTORESEARCH_MAX_LEN")
    budget_choices = dynamic_option_set("INTENT_AUTORESEARCH_TIME_BUDGET")

    candidates = [
        clone("best-lr-down", INTENT_AUTORESEARCH_LR=fmt_float(clamp_float(lr * 0.85, 0.0001, 0.004))),
        clone("best-lr-up", INTENT_AUTORESEARCH_LR=fmt_float(clamp_float(lr * 1.15, 0.0001, 0.004))),
        clone("best-batch-down", INTENT_AUTORESEARCH_BATCH_SIZE=adjust_choice(batch, batch_choices, -1)),
        clone("best-batch-up", INTENT_AUTORESEARCH_BATCH_SIZE=adjust_choice(batch, batch_choices, 1)),
        clone("best-embed-up", INTENT_AUTORESEARCH_EMBED_DIM=adjust_choice(embed, embed_choices, 1)),
        clone("best-hidden-up", INTENT_AUTORESEARCH_HIDDEN_DIM=adjust_choice(hidden, hidden_choices, 1)),
        clone("best-dropout-down", INTENT_AUTORESEARCH_DROPOUT=f"{clamp_float(dropout - 0.03, 0.05, 0.25):.2f}"),
        clone("best-dropout-up", INTENT_AUTORESEARCH_DROPOUT=f"{clamp_float(dropout + 0.03, 0.05, 0.25):.2f}"),
        clone(
            "best-weight-decay-down",
            INTENT_AUTORESEARCH_WEIGHT_DECAY=fmt_float(clamp_float(weight_decay - 0.005, 0.0, 0.08), 3),
        ),
        clone(
            "best-weight-decay-up",
            INTENT_AUTORESEARCH_WEIGHT_DECAY=fmt_float(clamp_float(weight_decay + 0.005, 0.0, 0.08), 3),
        ),
        clone(
            "best-context-up",
            INTENT_AUTORESEARCH_MAX_LEN=adjust_choice(max_len, max_len_choices, 1),
            INTENT_AUTORESEARCH_TIME_BUDGET=adjust_choice(time_budget, budget_choices, 1),
        ),
    ]
    proposal = candidates[run_idx % len(candidates)]
    proposal["_planner_source"] = "fallback"
    proposal["_planner_note"] = "best-neighborhood"
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
    stagnation = stagnation_profile()
    stagnation_text = ""
    if stagnation:
        stagnation_text = (
            "Stagnation profile:\n"
            f"{json.dumps(stagnation, indent=2)}\n"
            "The next proposal must meet or exceed these minimum capacity values because recent runs have stalled.\n"
        )
    return (
        "You are selecting the next intent classifier experiment for a local autoresearch loop.\n"
        "Goal: improve macro_f1 first, then accuracy, then lower val_intent_loss.\n"
        "Search in a tight neighborhood around the current best config.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Allowed keys only:\n"
        f"{json.dumps(ALLOWED_KEYS)}\n"
        "All values must be strings.\n"
        "Constraints:\n"
        "- lr between 0.0001 and 0.004\n"
        f"- batch size in {dynamic_option_set('INTENT_AUTORESEARCH_BATCH_SIZE')}\n"
        f"- embed dim in {dynamic_option_set('INTENT_AUTORESEARCH_EMBED_DIM')}\n"
        f"- hidden dim in {dynamic_option_set('INTENT_AUTORESEARCH_HIDDEN_DIM')}\n"
        "- dropout between 0.05 and 0.25\n"
        "- weight decay between 0.0 and 0.08\n"
        f"- max len in {dynamic_option_set('INTENT_AUTORESEARCH_MAX_LEN')}\n"
        f"- time budget in {dynamic_option_set('INTENT_AUTORESEARCH_TIME_BUDGET')}\n"
        "- description should be short and specific\n"
        "Current best config:\n"
        f"{json.dumps(best, indent=2)}\n"
        f"{stagnation_text}"
        "Recent results:\n"
        f"{json.dumps(summarize_rows(rows), indent=2)}\n"
        "Rules for locality:\n"
        "- change at most two hyperparameters from the current best config\n"
        "- prefer exact best or one-step neighbors over broad jumps\n"
        "- only increase capacity when the data has become harder or the stagnation profile requires it\n"
    )


def extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    candidates: list[str] = []
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend(fence_matches)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))
    for candidate_text in candidates:
        try:
            candidate = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, dict):
            return candidate
    preview = text[:240].replace("\n", "\\n")
    raise ValueError(f"no JSON object found in planner response: {preview}")


def repair_json(raw_text: str, best: dict[str, str]) -> dict[str, object]:
    repair_prompt = (
        "Rewrite the following planner output into exactly one valid JSON object.\n"
        "Allowed keys only:\n"
        f"{json.dumps(ALLOWED_KEYS)}\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- All values must be strings.\n"
        "- Preserve the intent of the candidate if possible.\n"
        f"- Use these defaults for any missing keys: {json.dumps(best, ensure_ascii=False)}\n"
        "Planner output to repair:\n"
        f"{raw_text}"
    )
    repaired = chat(repair_prompt, system=PLANNER_SYSTEM_PROMPT, expect_json=True)
    return extract_json(repaired)


def normalize(proposal: dict[str, object], best: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in ALLOWED_KEYS:
        value = proposal.get(key, best.get(key, ""))
        normalized[key] = str(value)
    for key, minimum in stagnation_profile().items():
        try:
            current = int(float(normalized.get(key, "0")))
        except ValueError:
            current = 0
        normalized[key] = str(max(current, minimum))
    if not normalized["description"].strip():
        normalized["description"] = "prometheus-proposal"
    return normalized


def enforce_best_neighborhood(
    proposal: dict[str, str],
    best: dict[str, str],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    changed = [key for key in LOCAL_KEYS if proposal.get(key, "") != best.get(key, "")]
    if len(changed) <= 2:
        return proposal
    fallback = heuristic_candidate(best, rows)
    fallback["_planner_note"] = f"forced-local-neighborhood-from-{len(changed)}-changes"
    return fallback


def enforce_stagnation_variation(
    proposal: dict[str, str],
    best: dict[str, str],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    if not stagnation_profile():
        return proposal

    varied = dict(proposal)
    same_description = varied.get("description", "").strip() == best.get("description", "").strip()
    same_regular_hparams = all(
        varied.get(key, "") == best.get(key, "")
        for key in (
            "INTENT_AUTORESEARCH_LR",
            "INTENT_AUTORESEARCH_BATCH_SIZE",
            "INTENT_AUTORESEARCH_DROPOUT",
            "INTENT_AUTORESEARCH_WEIGHT_DECAY",
        )
    )
    recent_rows = rows[-8:]
    recent_descriptions = {row.get("description", "").strip() for row in recent_rows}
    proposal_signature = tuple(
        varied.get(key, "")
        for key in (
            "INTENT_AUTORESEARCH_LR",
            "INTENT_AUTORESEARCH_BATCH_SIZE",
            "INTENT_AUTORESEARCH_DROPOUT",
            "INTENT_AUTORESEARCH_WEIGHT_DECAY",
            "INTENT_AUTORESEARCH_EMBED_DIM",
            "INTENT_AUTORESEARCH_HIDDEN_DIM",
            "INTENT_AUTORESEARCH_MAX_LEN",
            "INTENT_AUTORESEARCH_TIME_BUDGET",
        )
    )
    recent_signatures = {
        tuple(
            row.get(key, "")
            for key in (
                "INTENT_AUTORESEARCH_LR",
                "INTENT_AUTORESEARCH_BATCH_SIZE",
                "INTENT_AUTORESEARCH_DROPOUT",
                "INTENT_AUTORESEARCH_WEIGHT_DECAY",
                "INTENT_AUTORESEARCH_EMBED_DIM",
                "INTENT_AUTORESEARCH_HIDDEN_DIM",
                "INTENT_AUTORESEARCH_MAX_LEN",
                "INTENT_AUTORESEARCH_TIME_BUDGET",
            )
        )
        for row in recent_rows
    }
    recent_repeat = varied.get("description", "").strip() in recent_descriptions or proposal_signature in recent_signatures
    if not same_description and not same_regular_hparams and not recent_repeat:
        return varied

    jump_idx = len(rows) % 4
    lr = float(varied.get("INTENT_AUTORESEARCH_LR", best.get("INTENT_AUTORESEARCH_LR", "0.002")))
    dropout = float(varied.get("INTENT_AUTORESEARCH_DROPOUT", best.get("INTENT_AUTORESEARCH_DROPOUT", "0.1")))
    weight_decay = float(
        varied.get("INTENT_AUTORESEARCH_WEIGHT_DECAY", best.get("INTENT_AUTORESEARCH_WEIGHT_DECAY", "0.01"))
    )
    batch_choices = dynamic_option_set("INTENT_AUTORESEARCH_BATCH_SIZE")

    if jump_idx == 0:
        lr = clamp_float(lr * 0.5, 0.0001, 0.004)
        weight_decay = clamp_float(weight_decay + 0.01, 0.0, 0.08)
        varied["INTENT_AUTORESEARCH_BATCH_SIZE"] = adjust_choice(
            varied.get("INTENT_AUTORESEARCH_BATCH_SIZE", best.get("INTENT_AUTORESEARCH_BATCH_SIZE", "16")),
            batch_choices,
            1,
        )
    elif jump_idx == 1:
        lr = clamp_float(lr * 0.75, 0.0001, 0.004)
        dropout = clamp_float(dropout + 0.05, 0.05, 0.25)
        varied["INTENT_AUTORESEARCH_BATCH_SIZE"] = adjust_choice(
            varied.get("INTENT_AUTORESEARCH_BATCH_SIZE", best.get("INTENT_AUTORESEARCH_BATCH_SIZE", "16")),
            batch_choices,
            -1,
        )
    elif jump_idx == 2:
        lr = clamp_float(lr * 1.25, 0.0001, 0.004)
        weight_decay = clamp_float(weight_decay + 0.02, 0.0, 0.08)
    else:
        lr = clamp_float(lr * 0.6, 0.0001, 0.004)
        dropout = clamp_float(dropout + 0.08, 0.05, 0.25)
        weight_decay = clamp_float(weight_decay + 0.015, 0.0, 0.08)

    varied["INTENT_AUTORESEARCH_LR"] = f"{lr:.6f}".rstrip("0").rstrip(".")
    varied["INTENT_AUTORESEARCH_DROPOUT"] = f"{dropout:.2f}"
    varied["INTENT_AUTORESEARCH_WEIGHT_DECAY"] = f"{weight_decay:.3f}".rstrip("0").rstrip(".")
    varied["description"] = f"stagnation-jump-{len(rows) + 1}"
    return varied


def call_llm(prompt: str, best: dict[str, str]) -> dict[str, object]:
    raw = chat(prompt, system=PLANNER_SYSTEM_PROMPT, expect_json=True)
    try:
        return extract_json(raw)
    except ValueError:
        return repair_json(raw, best)


def main() -> int:
    best_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "best_config.json"
    results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results.tsv"
    best = load_best(best_path)
    rows = load_rows(results_path)
    fallback = enforce_stagnation_variation(normalize(heuristic_candidate(best, rows), best), best, rows)
    fallback["_planner_source"] = "fallback"

    if not env_flag("INTENT_AUTORESEARCH_USE_LLM_PLANNER", "INTENT_AUTORESEARCH_USE_CODEX_PLANNER", True):
        print(json.dumps(fallback, indent=2))
        return 0

    try:
        proposal = call_llm(build_prompt(best, rows), best)
        normalized = enforce_best_neighborhood(
            enforce_stagnation_variation(normalize(proposal, best), best, rows),
            best,
            rows,
        )
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
