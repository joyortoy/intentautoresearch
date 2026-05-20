from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys

from ollama_llm import chat, env_flag, provider_label


ROOT = Path(__file__).resolve().parent
SUMMARY_ROOT = ROOT / "logs" / "discord_summaries"
SUMMARY_PREFIX = "".join(
    ch for ch in os.getenv("INTENT_AUTORESEARCH_SUMMARY_PREFIX", "").strip()
    if ch.isalnum() or ch in {"_", "-"}
) or "default"
SUMMARY_LABEL = "".join(
    ch for ch in os.getenv("INTENT_AUTORESEARCH_SUMMARY_LABEL", "").strip()
    if ch.isalnum() or ch in {"_", "-"}
).lower() or SUMMARY_PREFIX.lower()
SUMMARY_TAG = f"[{SUMMARY_LABEL.upper()}-SUMMARY]"
MAX_REPORT_CHARS = 1600


def load_results(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    with results_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def kept_best(rows: list[dict[str, str]]) -> dict[str, str] | None:
    kept = [row for row in rows if row.get("status") == "keep"]
    if not kept:
        return None
    return max(
        kept,
        key=lambda row: (
            float(row.get("val_macro_f1") or "0"),
            float(row.get("val_accuracy") or "0"),
            -float(row.get("val_intent_loss") or "inf"),
        ),
    )


def clip(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def row_float(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key) or default)
    except Exception:
        return default


def row_score(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        row_float(row, "val_macro_f1", 0.0),
        row_float(row, "val_accuracy", 0.0),
        -row_float(row, "val_intent_loss", float("inf")),
    )


def format_row(row: dict[str, str], with_status: bool = True, desc_limit: int = 44) -> str:
    desc = clip(row.get("description", "-") or "-", desc_limit)
    parts = [desc]
    if with_status:
        parts.append(row.get("status", "-") or "-")
    parts.extend(
        [
            f"acc={row_float(row, 'val_accuracy', 0.0):.4f}",
            f"f1={row_float(row, 'val_macro_f1', 0.0):.4f}",
            f"loss={row_float(row, 'val_intent_loss', float('inf')):.6f}",
        ]
    )
    return " | ".join(parts)


def recent_rows(rows: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
    return rows[-limit:]


def tail_iteration_logs(iteration_dir: Path, limit: int = 4) -> list[dict[str, str]]:
    if not iteration_dir.exists():
        return []
    files = sorted(
        iteration_dir.glob("*_iteration_*.log"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )[-limit:]
    items: list[dict[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        items.append(
            {
                "file": str(path),
                "head": "\n".join(lines[:4])[:400],
                "tail": "\n".join(lines[-6:])[:500],
            }
        )
    return items


def infer_trend(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no-data"
    current = rows[-1]
    previous_kept = [row for row in rows[:-1] if row.get("status") == "keep"]
    if not previous_kept:
        return "new-baseline" if current.get("status") == "keep" else "cold-start"
    best_before = max(previous_kept, key=row_score)
    current_score = row_score(current)
    best_before_score = row_score(best_before)
    if current.get("status") == "keep" and current_score > best_before_score:
        return "improving"
    current_loss = row_float(current, "val_intent_loss", float("inf"))
    best_loss = row_float(best_before, "val_intent_loss", float("inf"))
    current_f1 = row_float(current, "val_macro_f1", 0.0)
    best_f1 = row_float(best_before, "val_macro_f1", 0.0)
    if current_f1 + 0.02 < best_f1 or current_loss > (best_loss * 1.5 if best_loss > 0 else current_loss + 1):
        return "regressing"
    return "flat"


def latest_log_signal(recent_logs: list[dict[str, str]]) -> str:
    if not recent_logs:
        return ""
    tail = recent_logs[-1].get("tail", "")
    for line in reversed([part.strip() for part in tail.splitlines() if part.strip()]):
        if line.startswith("Deck:"):
            continue
        return clip(line, 140)
    return ""


def heuristic_risk(rows: list[dict[str, str]], recent_logs: list[dict[str, str]], trend: str) -> str:
    if not rows:
        return "no completed runs yet"
    latest = rows[-1]
    if latest.get("status") == "crash":
        return "latest run crashed"
    if trend == "regressing":
        return "recent metrics slipped below the best kept run"
    if trend == "flat":
        return "recent runs are not beating the current best"
    signal = latest_log_signal(recent_logs)
    if signal:
        return signal
    return "need more iterations to confirm the trend"


def heuristic_next(rows: list[dict[str, str]], trend: str) -> str:
    if not rows:
        return "run the first training sweep"
    if trend in {"flat", "regressing"}:
        return "increase data difficulty or model capacity on the next stall"
    if trend == "improving":
        return "keep the current direction and validate with another run"
    return "collect more runs before changing the setup"


def deterministic_summary(
    iteration: int,
    rows: list[dict[str, str]],
    recent_logs: list[dict[str, str]],
    source: str = "metrics",
    trend: str | None = None,
    risk: str | None = None,
    next_step: str | None = None,
) -> str:
    best = kept_best(rows)
    recent = recent_rows(rows)
    trend = trend or infer_trend(rows)
    risk = clip(risk or heuristic_risk(rows, recent_logs, trend), 180)
    next_step = clip(next_step or heuristic_next(rows, trend), 180)
    if not best:
        lines = [
            f"{SUMMARY_TAG} iter={iteration} | trend={trend} | source={source}",
            "best: no kept runs yet",
            "recent:",
        ]
        lines.extend(
            f"- {format_row(row)}" for row in recent
        )
        lines.append(f"risk: {risk}")
        lines.append(f"next: {next_step}")
        return "\n".join(lines)[:MAX_REPORT_CHARS]
    lines = [
        f"{SUMMARY_TAG} iter={iteration} | trend={trend} | source={source}",
        f"best: {format_row(best, with_status=False, desc_limit=56)}",
        "recent:",
    ]
    lines.extend(f"- {format_row(row)}" for row in recent)
    signal = latest_log_signal(recent_logs)
    if signal:
        lines.append(f"latest: {signal}")
    lines.append(f"risk: {risk}")
    lines.append(f"next: {next_step}")
    return "\n".join(lines)[:MAX_REPORT_CHARS]


def build_prompt(iteration: int, rows: list[dict[str, str]], best: dict[str, str] | None, recent_logs: list[dict[str, str]]) -> str:
    report_rows = [
        {
            "description": row.get("description", ""),
            "status": row.get("status", ""),
            "val_accuracy": row.get("val_accuracy", ""),
            "val_macro_f1": row.get("val_macro_f1", ""),
            "val_intent_loss": row.get("val_intent_loss", ""),
        }
        for row in recent_rows(rows, limit=4)
    ]
    payload = {
        "iteration": iteration,
        "best_run": best or {},
        "recent_rows": report_rows,
        "recent_iteration_logs": recent_logs,
    }
    return (
        "You are summarizing intent autoresearch progress for Discord.\n"
        "Return JSON only.\n"
        "Schema:\n"
        '{"trend":"<one of improving|flat|regressing|cold-start|new-baseline>",'
        '"risk":"<short sentence>",'
        '"next":"<short sentence>"}\n'
        "Rules:\n"
        "- Keep risk under 120 characters.\n"
        "- Keep next under 120 characters.\n"
        "- Do not include any extra keys or commentary.\n"
        f"Data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_llm_json(raw: str) -> dict[str, str]:
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    payload = json.loads(match.group(1) if match else raw)
    if not isinstance(payload, dict):
        raise ValueError("summary payload was not a JSON object")
    return {
        "trend": clip(str(payload.get("trend", "")).strip().lower(), 24),
        "risk": clip(str(payload.get("risk", "")).strip(), 120),
        "next": clip(str(payload.get("next", "")).strip(), 120),
    }


def call_llm(prompt: str) -> dict[str, str]:
    return parse_llm_json(chat(prompt))


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: prometheus_progress_summary.py <iteration> <results_tsv> <iteration_log_dir>")
    iteration = int(sys.argv[1])
    results_path = Path(sys.argv[2])
    iteration_dir = Path(sys.argv[3])
    rows = load_results(results_path)
    best = kept_best(rows)
    recent_logs = tail_iteration_logs(iteration_dir)
    llm_fields: dict[str, str] = {}
    source = "metrics"
    error = ""
    if env_flag("INTENT_AUTORESEARCH_USE_LLM_SUMMARY", "INTENT_AUTORESEARCH_USE_CODEX_SUMMARY", True) and rows:
        try:
            llm_fields = call_llm(build_prompt(iteration, rows, best, recent_logs))
            source = provider_label()
        except Exception as exc:
            error = str(exc)
            source = "metrics"

    summary = deterministic_summary(
        iteration,
        rows,
        recent_logs,
        source=source,
        trend=llm_fields.get("trend"),
        risk=llm_fields.get("risk"),
        next_step=llm_fields.get("next"),
    )

    summary_dir = SUMMARY_ROOT / SUMMARY_PREFIX
    summary_dir.mkdir(parents=True, exist_ok=True)
    artifact = summary_dir / f"summary_iter_{iteration:04d}.json"
    artifact.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "prefix": SUMMARY_PREFIX,
                "label": SUMMARY_LABEL,
                "source": source,
                "summary": summary,
                "error": error,
                "llm_fields": llm_fields,
                "best": best or {},
                "recent_rows": recent_rows(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(str(artifact), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
