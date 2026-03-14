from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

from ollama_llm import chat, env_flag, provider_label


ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "logs" / "discord_summaries"


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


def tail_iteration_logs(iteration_dir: Path, limit: int = 4) -> list[dict[str, str]]:
    if not iteration_dir.exists():
        return []
    files = sorted(iteration_dir.glob("complex_forever_iteration_*.log"))[-limit:]
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


def deterministic_summary(iteration: int, rows: list[dict[str, str]]) -> str:
    best = kept_best(rows)
    recent = rows[-3:]
    if not best:
        return f"[AUTOSUMMARY] iter={iteration} | no kept runs yet | recent={len(recent)}"
    parts = [
        f"[AUTOSUMMARY] iter={iteration}",
        f"best={best.get('description','-')}",
        f"acc={float(best.get('val_accuracy') or '0'):.4f}",
        f"f1={float(best.get('val_macro_f1') or '0'):.4f}",
        f"loss={float(best.get('val_intent_loss') or 'inf'):.6f}",
    ]
    return " | ".join(parts)


def build_prompt(iteration: int, rows: list[dict[str, str]], best: dict[str, str] | None, recent_logs: list[dict[str, str]]) -> str:
    recent_rows = [
        {
            "description": row.get("description", ""),
            "status": row.get("status", ""),
            "val_accuracy": row.get("val_accuracy", ""),
            "val_macro_f1": row.get("val_macro_f1", ""),
            "val_intent_loss": row.get("val_intent_loss", ""),
        }
        for row in rows[-4:]
    ]
    payload = {
        "iteration": iteration,
        "best_run": best or {},
        "recent_rows": recent_rows,
        "recent_iteration_logs": recent_logs,
    }
    return (
        "You are summarizing intent autoresearch progress for Discord.\n"
        "Return plain text only, no markdown fences.\n"
        "Make it compact and factual.\n"
        "Format:\n"
        "[PROMETHEUS-SUMMARY] iter=<n> | trend=<short> | best=<run> | acc=<a> | f1=<f> | risk=<short> | next=<short>\n"
        "Rules:\n"
        "- Use one line only.\n"
        "- Under 280 characters.\n"
        "- Do not explain your reasoning.\n"
        "- Mention the best run and whether recent progress is improving, flat, or regressing.\n"
        "- Mention one concrete risk or blocker.\n"
        "- Mention one concrete next step.\n"
        f"Data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def call_llm(prompt: str) -> str:
    raw = " ".join(chat(prompt).splitlines()).strip()
    match = re.search(r"(\[PROMETHEUS-SUMMARY\][^\[]+)", raw)
    if match:
        return match.group(1).strip()[:280]
    return raw[:280]


def finalize_summary(summary: str, fallback: str, best: dict[str, str] | None) -> str:
    text = summary.strip()
    if best and "best=<run>" in text:
        text = text.replace("best=<run>", f"best={best.get('description', '-')}")
    if "<short>" in text or "<run>" in text:
        return fallback
    return text[:280]


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: prometheus_progress_summary.py <iteration> <results_tsv> <iteration_log_dir>")
    iteration = int(sys.argv[1])
    results_path = Path(sys.argv[2])
    iteration_dir = Path(sys.argv[3])
    rows = load_results(results_path)
    best = kept_best(rows)
    recent_logs = tail_iteration_logs(iteration_dir)
    fallback = deterministic_summary(iteration, rows)

    summary = fallback
    source = "fallback"
    error = ""
    if env_flag("INTENT_AUTORESEARCH_USE_LLM_SUMMARY", "INTENT_AUTORESEARCH_USE_CODEX_SUMMARY", True):
        try:
            summary = finalize_summary(call_llm(build_prompt(iteration, rows, best, recent_logs)), fallback, best)
            source = provider_label()
        except Exception as exc:
            error = str(exc)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    artifact = SUMMARY_DIR / f"summary_iter_{iteration:04d}.json"
    artifact.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "source": source,
                "summary": summary,
                "error": error,
                "best": best or {},
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
